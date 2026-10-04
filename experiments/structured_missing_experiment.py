"""Run the 3-pattern x 2-dataset x 4-method structured-missing experiment.

The default matrix contains 24 evaluations:

* patterns: multi_sensor, sensor_actuator, subsystem
* datasets: SWaT, WADI
* methods: SAITS, BRITS, Transformer, proposed

Models are trained once per dataset/method on the complete training prefix and
then reused for all selected test missingness patterns.  Test time and peak
resident memory are measured around imputation only (including test-time MSSA
decomposition for the proposed method).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import inspect
import json
import math
import os
import platform
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from spectraformer.config import DATA_DIR, STRUCTURED_EXPERIMENT_OUTPUT_DIR
from spectraformer.structured_missingness import (
    DATASET_SPECS,
    PATTERN_NAMES,
    apply_structured_missing,
    validate_pattern_columns,
)


BASELINE_METHODS = ("SAITS", "BRITS", "Transformer")
PROPOSED_METHODS = ("proposed", "without_mssa", "without_transformer_encoder")
METHODS = (*BASELINE_METHODS, *PROPOSED_METHODS)
DEFAULT_METHODS = (*BASELINE_METHODS, "proposed")
EXISTING_MISSING_FILES = {
    "swat": "train_continuous_missing_40.csv",
    "wadi": "train_wadi_continuous_missing_40.csv",
}
METHOD_DISPLAY_NAMES = {
    "proposed": "Full Model",
    "without_mssa": "w/o mSSA",
    "without_transformer_encoder": "w/o Transformer Encoder",
}


@dataclass
class PreparedData:
    dataset: str
    feature_columns: List[str]
    train_original: np.ndarray
    train_scaled: np.ndarray
    test_frame: pd.DataFrame
    test_original: np.ndarray
    test_scaled: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class TrainedMethod:
    name: str
    model: Any
    train_seconds: float
    extra: Dict[str, Any]


class PeakMemorySampler:
    """Sample process RSS in a lightweight background thread."""

    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = interval_seconds
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.start_rss = current_rss_bytes()
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss = max(self.peak_rss, current_rss_bytes())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.peak_rss = max(self.peak_rss, current_rss_bytes())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def current_rss_bytes() -> int:
    """Return current resident memory without requiring psutil."""

    if os.name == "nt":
        size_t = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else 0

    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    return 0


def _torch_module():
    try:
        import torch

        return torch
    except ImportError:
        return None


def synchronize_device() -> None:
    torch = _torch_module()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_inference(call: Callable[[], np.ndarray]) -> Tuple[np.ndarray, dict]:
    """Measure wall time, process RSS, and CUDA peaks for one test call."""

    gc.collect()
    torch = _torch_module()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    synchronize_device()

    with PeakMemorySampler() as memory:
        start = time.perf_counter()
        result = call()
        synchronize_device()
        elapsed = time.perf_counter() - start

    cuda_allocated = 0
    cuda_reserved = 0
    if torch is not None and torch.cuda.is_available():
        cuda_allocated = int(torch.cuda.max_memory_allocated())
        cuda_reserved = int(torch.cuda.max_memory_reserved())

    mib = 1024.0 ** 2
    measurements = {
        "test_seconds": elapsed,
        "peak_process_rss_mib": memory.peak_rss / mib,
        "process_rss_increase_mib": max(memory.peak_rss - memory.start_rss, 0) / mib,
        "peak_cuda_allocated_mib": cuda_allocated / mib,
        "peak_cuda_reserved_mib": cuda_reserved / mib,
    }
    return np.asarray(result), measurements


def cap_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit and len(frame) > limit:
        return frame.iloc[:limit].copy()
    return frame.copy()


def prepare_dataset(
    dataset: str,
    train_fraction: float,
    max_train_rows: int,
    max_test_rows: int,
    n_steps: int,
) -> PreparedData:
    spec = DATASET_SPECS[dataset]
    path = DATA_DIR / spec.filename
    frame = pd.read_csv(path)

    for pattern in PATTERN_NAMES:
        validate_pattern_columns(frame.columns, dataset, pattern)

    excluded = {spec.index_column, spec.label_column}
    feature_columns = [column for column in frame.columns if column not in excluded]
    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns in {dataset}: {non_numeric}")

    # Put any remainder into training so a full-data run (both caps zero)
    # consumes every source row while keeping baseline test windows aligned.
    usable_test_rows = (int(len(frame) * (1.0 - train_fraction)) // n_steps) * n_steps
    if usable_test_rows < n_steps:
        raise ValueError(f"{dataset} test split has fewer than n_steps={n_steps} rows")
    split = len(frame) - usable_test_rows
    train_frame = cap_rows(frame.iloc[:split], max_train_rows)
    test_frame = cap_rows(frame.iloc[split:], max_test_rows)
    test_frame = test_frame.iloc[:(len(test_frame) // n_steps) * n_steps].copy()

    train_original = train_frame[feature_columns].to_numpy(dtype=np.float32)
    test_original = test_frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(train_original).all() or not np.isfinite(test_original).all():
        raise ValueError(
            f"{dataset} complete reference contains NaN/inf in experiment features"
        )

    mean = train_original.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_original.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    train_scaled = (train_original - mean) / std
    test_scaled = (test_original - mean) / std

    return PreparedData(
        dataset=dataset,
        feature_columns=feature_columns,
        train_original=train_original,
        train_scaled=train_scaled.astype(np.float32),
        test_frame=test_frame,
        test_original=test_original,
        test_scaled=test_scaled.astype(np.float32),
        mean=mean,
        std=std,
    )


def prepare_existing_missing_dataset(dataset: str) -> Tuple[PreparedData, dict]:
    """Load the complete existing 40%-missing file without row subsampling."""

    spec = DATASET_SPECS[dataset]
    complete_path = DATA_DIR / spec.filename
    missing_path = DATA_DIR / EXISTING_MISSING_FILES[dataset]
    complete = pd.read_csv(complete_path)
    missing = pd.read_csv(missing_path)
    if len(complete) != len(missing):
        raise ValueError(
            f"Row mismatch for {dataset}: complete={len(complete)}, missing={len(missing)}"
        )

    excluded = {spec.index_column, spec.label_column}
    feature_columns = [
        column for column in complete.columns
        if column not in excluded and column in missing.columns
    ]
    truth = complete[feature_columns].to_numpy(dtype=np.float32)
    missing_values = missing[feature_columns].to_numpy(dtype=np.float32)
    mask = np.isnan(missing_values)
    if not mask.any():
        raise ValueError(f"No missing values found in {missing_path}")
    if not np.isfinite(truth[mask]).all():
        raise ValueError(f"Complete truth is invalid at masked positions for {dataset}")

    mean = np.nanmean(missing_values, axis=0).astype(np.float32)
    std = np.nanstd(missing_values, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    std[(~np.isfinite(std)) | (std < 1e-8)] = 1.0
    missing_scaled = ((missing_values - mean) / std).astype(np.float32)
    selected_columns = [
        feature_columns[index] for index in np.flatnonzero(mask.any(axis=0))
    ]
    metadata = {
        "dataset": dataset,
        "pattern": "existing_continuous_40",
        "description": "Existing project-provided continuous 40% missing dataset.",
        "requested_group_missing_rate": 0.4,
        "actual_group_missing_rate": float(mask[:, mask.any(axis=0)].mean()),
        "actual_overall_missing_rate": float(mask.mean()),
        "n_rows": len(missing),
        "selected_columns": selected_columns,
        "n_selected_columns": len(selected_columns),
        "source_path": str(missing_path),
        "complete_path": str(complete_path),
        "protected_columns": [spec.index_column, spec.label_column],
    }
    return PreparedData(
        dataset=dataset,
        feature_columns=feature_columns,
        train_original=truth,
        train_scaled=missing_scaled,
        test_frame=missing,
        test_original=truth,
        test_scaled=missing_scaled,
        mean=mean,
        std=std,
    ), metadata


def sliding_windows(array: np.ndarray, n_steps: int, stride: int) -> np.ndarray:
    if len(array) < n_steps:
        raise ValueError(f"Need at least {n_steps} rows, received {len(array)}")
    starts = range(0, len(array) - n_steps + 1, stride)
    return np.stack([array[start:start + n_steps] for start in starts]).astype(np.float32)


def nonoverlapping_windows(array: np.ndarray, n_steps: int) -> np.ndarray:
    usable = (len(array) // n_steps) * n_steps
    return array[:usable].reshape(-1, n_steps, array.shape[1]).astype(np.float32)


def _filter_constructor_args(model_class: type, params: dict) -> dict:
    signature = inspect.signature(model_class.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return params
    return {key: value for key, value in params.items() if key in signature.parameters}


def build_pypots_model(
    method: str, n_steps: int, n_features: int, epochs: int, batch_size: int
):
    try:
        from pypots.imputation import BRITS, SAITS, Transformer
    except ImportError as exc:
        raise RuntimeError(
            "PyPOTS is required for SAITS/BRITS/Transformer; install requirements.txt"
        ) from exc

    classes = {"SAITS": SAITS, "BRITS": BRITS, "Transformer": Transformer}
    common = {
        "n_steps": n_steps,
        "n_features": n_features,
        "epochs": epochs,
        "patience": max(2, min(10, epochs // 3)),
        "batch_size": batch_size,
        "verbose": True,
    }
    specialized = {
        "SAITS": {
            "n_layers": 2, "d_model": 64, "d_ffn": 128, "n_heads": 4,
            "d_k": 16, "d_v": 16, "dropout": 0.1,
        },
        "BRITS": {"rnn_hidden_size": 64},
        "Transformer": {
            "n_layers": 2, "d_model": 64, "d_inner": 128, "d_ffn": 128,
            "n_heads": 4, "d_k": 16, "d_v": 16, "dropout": 0.1,
            "attn_dropout": 0.1,
        },
    }
    params = {**common, **specialized[method]}
    model_class = classes[method]
    return model_class(**_filter_constructor_args(model_class, params))


def train_method(method: str, data: PreparedData, args: argparse.Namespace) -> TrainedMethod:
    print(f"  Training {method} once for {data.dataset.upper()}...")
    started = time.perf_counter()
    np.random.seed(args.seed)
    torch = _torch_module()
    if torch is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if method in BASELINE_METHODS:
        model = build_pypots_model(
            method,
            n_steps=args.n_steps,
            n_features=len(data.feature_columns),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        train_windows = sliding_windows(
            data.train_scaled, args.n_steps, args.train_stride
        )
        model.fit({"X": train_windows})
        extra = {"train_windows": len(train_windows)}
    elif method in PROPOSED_METHODS:
        try:
            import torch
            from spectraformer.mssa.mssa1 import MSSA
            from spectraformer.transformer.transformer_imputation_mssa_mlp_re_bias import (
                SpectralMSSATransformerImputer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The proposed model requires PyTorch, SciPy, and scikit-learn"
            ) from exc

        uses_mssa = method != "without_mssa"
        uses_transformer_encoder = method != "without_transformer_encoder"
        mssa = None
        if uses_mssa:
            mssa_training_data = (
                pd.DataFrame(data.train_scaled)
                .interpolate(method="linear", limit_direction="both")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            mssa = MSSA(
                window_length=args.mssa_window,
                n_components=args.mssa_components,
                verbose=False,
                use_sparse_svd=True,
            ).fit(mssa_training_data)
        model = SpectralMSSATransformerImputer(
            mssa_window_length=args.mssa_window,
            mssa_n_components=args.mssa_components,
            transformer_config={
                "d_model": args.d_model,
                "nhead": args.n_heads,
                "num_encoder_layers": args.n_layers,
                "dim_feedforward": args.d_ffn,
                "dropout": 0.1,
                "prediction_length": 1,
            },
            mlp_config={"hidden_dims": [args.d_ffn, args.d_model], "dropout": 0.1},
            spectral_config={
                "max_seq_len": max(512, args.n_steps),
                "top_k_freq": args.top_k_freq,
                "learnable_weights": True,
                "temperature": 1.0,
            },
            fusion_method="concat",
            use_modal_fusion=True,
            modal_fusion_type="adaptive",
            use_spectral_bias=True,
            use_transformer_encoder=uses_transformer_encoder,
        )
        model.train(
            data.train_scaled,
            val_data=None,
            mssa_model=mssa,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sequence_length=args.n_steps,
            patience=max(2, min(10, args.epochs // 3)),
            verbose=True,
        )
        extra = {
            "train_mssa": mssa,
            "uses_mssa": uses_mssa,
            "uses_transformer_encoder": uses_transformer_encoder,
            "uses_modal_fusion": uses_mssa,
            "uses_spectral_bias": uses_transformer_encoder,
            "parameter_count": sum(p.numel() for p in model.model.parameters()),
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return TrainedMethod(
        name=method,
        model=model,
        train_seconds=time.perf_counter() - started,
        extra=extra,
    )


def extract_pypots_imputation(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        for key in ("imputation", "X", "imputed_data"):
            if key in result:
                return np.asarray(result[key])
        raise ValueError(f"Unrecognized PyPOTS result keys: {sorted(result)}")
    return np.asarray(result)


def impute_method(
    trained: TrainedMethod,
    masked_scaled: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, dict]:
    if trained.name in BASELINE_METHODS:
        test_windows = nonoverlapping_windows(masked_scaled, args.n_steps)

        def call() -> np.ndarray:
            return extract_pypots_imputation(trained.model.impute({"X": test_windows}))

        result, performance = measure_inference(call)
        imputed_scaled = result.reshape(-1, result.shape[-1])
    else:
        def call() -> np.ndarray:
            from spectraformer.mssa.mssa1 import MSSA

            test_mssa = None
            if trained.extra.get("uses_mssa", False):
                preliminary = (
                    pd.DataFrame(masked_scaled)
                    .interpolate(method="linear", limit_direction="both")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                test_mssa = MSSA(
                    window_length=args.mssa_window,
                    n_components=args.mssa_components,
                    verbose=False,
                    use_sparse_svd=True,
                ).fit(preliminary)
            return trained.model.impute_robust(
                masked_scaled,
                mssa_model=test_mssa,
                iterations=args.imputation_iterations,
                min_sequence_length=min(10, args.n_steps),
                fallback_method="interpolate",
            )

        imputed_scaled, performance = measure_inference(call)

    if imputed_scaled.shape != masked_scaled.shape:
        raise ValueError(
            f"Imputed shape {imputed_scaled.shape} != input shape {masked_scaled.shape}"
        )
    return imputed_scaled.astype(np.float32), performance


def calculate_metrics(
    truth: np.ndarray,
    imputed: np.ndarray,
    mask: np.ndarray,
    feature_columns: Sequence[str],
) -> Tuple[dict, dict]:
    valid = mask & np.isfinite(truth) & np.isfinite(imputed)
    if not valid.any():
        raise ValueError("No finite imputed values at masked positions")
    y_true = truth[valid].astype(np.float64)
    y_pred = imputed[valid].astype(np.float64)
    error = y_pred - y_true
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
    variance = float(np.var(y_true))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "smape_percent": float(np.mean(2.0 * np.abs(error) / denominator) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "nrmse_by_std": float(np.sqrt(np.mean(error ** 2)) / np.sqrt(variance))
        if variance > 0 else math.nan,
        "evaluated_cells": int(valid.sum()),
        "finite_prediction_rate": float(valid.sum() / mask.sum()),
    }

    feature_metrics = {}
    for index, column in enumerate(feature_columns):
        column_mask = valid[:, index]
        if not column_mask.any():
            continue
        column_error = imputed[column_mask, index] - truth[column_mask, index]
        feature_metrics[column] = {
            "rmse": float(np.sqrt(np.mean(column_error ** 2))),
            "mae": float(np.mean(np.abs(column_error))),
            "evaluated_cells": int(column_mask.sum()),
        }
    return metrics, feature_metrics


def result_directory(root: Path, dataset: str, pattern: str, method: str) -> Path:
    path = root / dataset / pattern / method.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_imputed_csv(
    path: Path, data: PreparedData, imputed: np.ndarray
) -> None:
    output = data.test_frame.copy()
    output.loc[:, data.feature_columns] = imputed
    output.to_csv(path, index=False)


def hardware_metadata() -> dict:
    metadata = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    torch = _torch_module()
    if torch is not None:
        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["cuda_device"] = torch.cuda.get_device_name(0)
    return metadata


def run(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    for dataset in args.datasets:
        print(f"\nPreparing {dataset.upper()}...")
        if args.existing_missing:
            data, mask_metadata = prepare_existing_missing_dataset(dataset)
            feature_mask = np.isnan(data.test_scaled)
            pattern_inputs = {
                "existing_continuous_40": (
                    data.test_scaled, feature_mask, mask_metadata
                )
            }
        else:
            data = prepare_dataset(
                dataset,
                train_fraction=args.train_fraction,
                max_train_rows=args.max_train_rows,
                max_test_rows=args.max_test_rows,
                n_steps=args.n_steps,
            )

            pattern_inputs = {}
            for pattern_index, pattern in enumerate(args.patterns):
                masked_frame, mask_frame, mask_metadata = apply_structured_missing(
                    data.test_frame,
                    dataset=dataset,
                    pattern=pattern,
                    missing_rate=args.missing_rate,
                    n_blocks=args.blocks,
                    seed=args.seed + pattern_index,
                )
                masked_original = masked_frame[data.feature_columns].to_numpy(dtype=np.float32)
                masked_scaled = (masked_original - data.mean) / data.std
                feature_mask = mask_frame[data.feature_columns].to_numpy(dtype=bool)
                pattern_inputs[pattern] = (masked_scaled, feature_mask, mask_metadata)

        for method in args.methods:
            try:
                trained = train_method(method, data, args)
            except Exception as exc:
                print(f"  FAILED training {method}: {exc}")
                for pattern in args.patterns:
                    results.append(
                        {
                            "dataset": dataset,
                            "pattern": pattern,
                            "method": method,
                            "status": "training_failed",
                            "error": str(exc),
                        }
                    )
                if args.fail_fast:
                    raise
                continue

            for pattern in args.patterns:
                masked_scaled, feature_mask, mask_metadata = pattern_inputs[pattern]
                print(f"  Testing {method} on {pattern}...")
                output_dir = result_directory(args.output_dir, dataset, pattern, method)
                try:
                    imputed_scaled, performance = impute_method(
                        trained, masked_scaled, args
                    )
                    imputed = imputed_scaled * data.std + data.mean
                    metrics, per_feature = calculate_metrics(
                        data.test_original, imputed, feature_mask, data.feature_columns
                    )
                    row = {
                        "dataset": dataset,
                        "pattern": pattern,
                        "method": method,
                        "ablation_variant": METHOD_DISPLAY_NAMES.get(method, method),
                        "status": "ok",
                        "group_missing_rate": mask_metadata["actual_group_missing_rate"],
                        "overall_missing_rate": mask_metadata["actual_overall_missing_rate"],
                        "selected_columns": "|".join(mask_metadata["selected_columns"]),
                        "train_rows": len(data.train_scaled),
                        "test_rows": len(data.test_scaled),
                        "train_seconds": trained.train_seconds,
                        "use_mssa": trained.extra.get("uses_mssa", False),
                        "use_transformer_encoder": trained.extra.get(
                            "uses_transformer_encoder", method in BASELINE_METHODS
                        ),
                        "use_modal_fusion": trained.extra.get("uses_modal_fusion", False),
                        "use_spectral_bias": trained.extra.get("uses_spectral_bias", False),
                        "parameter_count": trained.extra.get("parameter_count", math.nan),
                        **metrics,
                        **performance,
                    }
                    row["imputed_cells_per_second"] = (
                        metrics["evaluated_cells"] / performance["test_seconds"]
                        if performance["test_seconds"] > 0 else math.nan
                    )
                    configuration = dict(vars(args))
                    configuration["output_dir"] = str(args.output_dir)
                    detail = {
                        "result": row,
                        "mask": mask_metadata,
                        "per_feature": per_feature,
                        "configuration": configuration,
                        "hardware": hardware_metadata(),
                    }
                    (output_dir / "metrics.json").write_text(
                        json.dumps(detail, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    if args.save_imputed:
                        write_imputed_csv(output_dir / "imputed.csv", data, imputed)
                    results.append(row)
                except Exception as exc:
                    print(f"  FAILED {dataset}/{pattern}/{method}: {exc}")
                    (output_dir / "error.txt").write_text(
                        f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8"
                    )
                    results.append(
                        {
                            "dataset": dataset,
                            "pattern": pattern,
                            "method": method,
                            "status": "test_failed",
                            "error": str(exc),
                        }
                    )
                    if args.fail_fast:
                        raise

            del trained
            gc.collect()

        summary = pd.DataFrame(results)
        summary.to_csv(args.output_dir / "summary.csv", index=False)

    summary = pd.DataFrame(results)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def print_plan(args: argparse.Namespace) -> None:
    rows = []
    for dataset in args.datasets:
        if args.existing_missing:
            data, metadata = prepare_existing_missing_dataset(dataset)
            for method in args.methods:
                rows.append(
                    {
                        "dataset": dataset,
                        "pattern": "existing_continuous_40",
                        "method": method,
                        "rows": len(data.test_original),
                        "masked_variables": metadata["n_selected_columns"],
                        "columns": ", ".join(metadata["selected_columns"]),
                    }
                )
            continue
        header = pd.read_csv(DATA_DIR / DATASET_SPECS[dataset].filename, nrows=0)
        for pattern in args.patterns:
            columns = validate_pattern_columns(header.columns, dataset, pattern)
            for method in args.methods:
                rows.append(
                    {
                        "dataset": dataset,
                        "pattern": pattern,
                        "method": method,
                        "masked_variables": len(columns),
                        "columns": ", ".join(columns),
                    }
                )
    plan = pd.DataFrame(rows)
    print(plan.to_string(index=False))
    print(f"\nTotal evaluations: {len(plan)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS),
                        default=sorted(DATASET_SPECS))
    parser.add_argument("--patterns", nargs="+", choices=PATTERN_NAMES,
                        default=list(PATTERN_NAMES))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run Full Model plus the two added ablations, overriding --methods.",
    )
    parser.add_argument(
        "--existing-missing", action="store_true",
        help="Use the complete existing SWaT/WADI 40%-missing files without subsampling.",
    )
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-train-rows", type=int, default=3000,
                        help="0 uses the full prefix; MSSA memory grows linearly with rows.")
    parser.add_argument("--max-test-rows", type=int, default=2000,
                        help="0 uses the full test suffix.")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ffn", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mssa-window", type=int, default=48)
    parser.add_argument("--mssa-components", type=int, default=8)
    parser.add_argument("--top-k-freq", type=int, default=10)
    parser.add_argument("--imputation-iterations", type=int, default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=STRUCTURED_EXPERIMENT_OUTPUT_DIR)
    parser.add_argument("--save-imputed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="Validate columns and print the evaluation matrix only.")
    args = parser.parse_args(argv)
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if not 0.0 < args.missing_rate < 1.0:
        parser.error("--missing-rate must be between 0 and 1")
    if args.mssa_window >= args.max_train_rows and args.max_train_rows:
        parser.error("--mssa-window must be smaller than --max-train-rows")
    if not 1 <= args.top_k_freq <= args.n_steps // 2:
        parser.error("--top-k-freq must be between 1 and half of --n-steps")
    if args.ablation:
        args.methods = list(PROPOSED_METHODS)
    if args.existing_missing:
        args.patterns = ["existing_continuous_40"]
    return args


def main() -> None:
    args = parse_args()
    if args.plan:
        print_plan(args)
        return
    summary = run(args)
    print("\nExperiment summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
