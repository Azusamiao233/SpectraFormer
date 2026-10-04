"""Structured missing-pattern generation for SWaT and WADI experiments.

The three patterns implemented here model increasingly coupled failures:

``multi_sensor``
    Four continuous sensor channels disappear over the same time blocks.
``sensor_actuator``
    A physically related sensor/actuator group disappears together.
``subsystem``
    Related variables in different process stages disappear together.

The requested missing rate is defined *within the selected variable group*.  In
other words, ``missing_rate=0.4`` masks 40% of the rows of every selected
variable.  All variables in a group share the same non-overlapping continuous
blocks.  Identifier, timestamp, and attack-label columns are never masked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternSpec:
    columns: Tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    index_column: str
    label_column: str
    patterns: Mapping[str, PatternSpec]


# These groups follow the process-stage/device naming in the SWaT and WADI
# testbed descriptions.  Keeping the choices explicit makes every run auditable
# and prevents data-dependent feature selection from leaking test information.
DATASET_SPECS: Dict[str, DatasetSpec] = {
    "swat": DatasetSpec(
        filename="train_swat.csv",
        index_column="Unnamed: 0",
        label_column="label",
        patterns={
            "multi_sensor": PatternSpec(
                columns=("FIT101", "LIT101", "AIT201", "FIT201"),
                description="Four continuous sensors spanning SWaT stages P1-P2.",
            ),
            "sensor_actuator": PatternSpec(
                columns=("FIT101", "MV101"),
                description="P1 inlet flow sensor and its directly related inlet valve.",
            ),
            "subsystem": PatternSpec(
                columns=("AIT201", "AIT501"),
                description="Related water-quality sensors across SWaT P2 and P5.",
            ),
        },
    ),
    "wadi": DatasetSpec(
        filename="train_wadi.csv",
        index_column="timestep",
        label_column="Attack",
        patterns={
            "multi_sensor": PatternSpec(
                columns=(
                    "1_AIT_001_PV", "1_AIT_002_PV", "1_AIT_003_PV",
                    "1_AIT_004_PV",
                ),
                description="Four continuous water-quality sensors in WADI stage P1.",
            ),
            "sensor_actuator": PatternSpec(
                columns=("1_FIT_001_PV", "1_MV_001_STATUS"),
                description="P1 inlet flow sensor and its directly related inlet valve.",
            ),
            "subsystem": PatternSpec(
                columns=("1_AIT_001_PV", "2A_AIT_001_PV"),
                description="Related water-quality sensors across WADI P1 and P2A.",
            ),
        },
    ),
}

PATTERN_NAMES = tuple(next(iter(DATASET_SPECS.values())).patterns)


def get_dataset_spec(dataset: str) -> DatasetSpec:
    """Return a dataset specification after normalizing its name."""

    key = dataset.strip().lower()
    if key not in DATASET_SPECS:
        valid = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of: {valid}")
    return DATASET_SPECS[key]


def validate_pattern_columns(
    columns: Iterable[str], dataset: str, pattern: str
) -> Tuple[str, ...]:
    """Validate configured columns against a dataframe column collection."""

    spec = get_dataset_spec(dataset)
    if pattern not in spec.patterns:
        valid = ", ".join(spec.patterns)
        raise ValueError(f"Unknown pattern {pattern!r}; expected one of: {valid}")

    available = set(columns)
    selected = spec.patterns[pattern].columns
    missing = [column for column in selected if column not in available]
    if missing:
        raise ValueError(
            f"{dataset}/{pattern} is missing configured columns: {missing}"
        )
    protected = {spec.index_column, spec.label_column}
    overlap = protected.intersection(selected)
    if overlap:
        raise ValueError(f"Protected columns cannot be masked: {sorted(overlap)}")
    return selected


def _positive_partition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly partition ``total`` into ``parts`` positive integers."""

    if parts < 1 or total < parts:
        raise ValueError("A positive partition requires total >= parts >= 1")
    return rng.multinomial(total - parts, np.full(parts, 1.0 / parts)) + 1


def generate_block_rows(
    n_rows: int,
    missing_rate: float = 0.4,
    n_blocks: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
    """Generate an exact-rate union of non-overlapping continuous row blocks.

    Returns a boolean row mask and half-open ``(start, end)`` block intervals.
    Internal blocks are separated by at least one observed row whenever the
    requested dimensions make that possible.
    """

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must be strictly between 0 and 1")
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")

    target = int(round(n_rows * missing_rate))
    target = min(max(target, 1), n_rows - 1)
    observed = n_rows - target
    # At most observed+1 blocks can be separated by observed rows.
    actual_blocks = min(n_blocks, target, observed + 1)
    rng = np.random.default_rng(seed)

    block_lengths = _positive_partition(target, actual_blocks, rng)
    gaps = np.zeros(actual_blocks + 1, dtype=int)
    if actual_blocks > 1:
        gaps[1:-1] = 1
    remaining_observed = observed - max(actual_blocks - 1, 0)
    if remaining_observed:
        gaps += rng.multinomial(
            remaining_observed, np.full(actual_blocks + 1, 1.0 / (actual_blocks + 1))
        )

    row_mask = np.zeros(n_rows, dtype=bool)
    intervals = []
    cursor = int(gaps[0])
    for block_index, length in enumerate(block_lengths):
        start = cursor
        end = start + int(length)
        row_mask[start:end] = True
        intervals.append((start, end))
        cursor = end + int(gaps[block_index + 1])

    if int(row_mask.sum()) != target:
        raise RuntimeError("Internal error: generated mask does not match target rate")
    return row_mask, tuple(intervals)


def make_structured_mask(
    frame: pd.DataFrame,
    dataset: str,
    pattern: str,
    missing_rate: float = 0.4,
    n_blocks: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """Return a boolean cell mask and serializable generation metadata."""

    selected = validate_pattern_columns(frame.columns, dataset, pattern)
    if frame.loc[:, selected].isna().any().any():
        raise ValueError(
            "Configured source columns already contain missing values; use a complete "
            "reference dataset so artificial and pre-existing missingness remain distinct"
        )

    row_mask, intervals = generate_block_rows(
        len(frame), missing_rate=missing_rate, n_blocks=n_blocks, seed=seed
    )
    mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    mask.loc[row_mask, list(selected)] = True

    spec = get_dataset_spec(dataset)
    actual_group_rate = float(mask.loc[:, list(selected)].to_numpy().mean())
    eligible_columns = [
        column for column in frame.columns if column not in {spec.index_column, spec.label_column}
    ]
    actual_overall_rate = float(mask.loc[:, eligible_columns].to_numpy().mean())
    metadata = {
        "dataset": dataset.lower(),
        "pattern": pattern,
        "description": spec.patterns[pattern].description,
        "missing_rate_definition": "masked cells / cells in selected variable group",
        "requested_group_missing_rate": float(missing_rate),
        "actual_group_missing_rate": actual_group_rate,
        "actual_overall_missing_rate": actual_overall_rate,
        "overall_missing_rate_definition": "masked cells / feature cells (index and label excluded)",
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "selected_columns": list(selected),
        "n_selected_columns": len(selected),
        "blocks": [{"start": start, "end": end, "length": end - start}
                   for start, end in intervals],
        "seed": int(seed),
        "protected_columns": [spec.index_column, spec.label_column],
    }
    return mask, metadata


def apply_structured_missing(
    frame: pd.DataFrame,
    dataset: str,
    pattern: str,
    missing_rate: float = 0.4,
    n_blocks: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Copy ``frame``, apply a structured mask, and return data/mask/metadata."""

    mask, metadata = make_structured_mask(
        frame,
        dataset=dataset,
        pattern=pattern,
        missing_rate=missing_rate,
        n_blocks=n_blocks,
        seed=seed,
    )
    masked = frame.copy()
    masked[mask] = np.nan
    return masked, mask, metadata


def specs_as_dict() -> dict:
    """Return all specifications in a JSON-serializable form."""

    return {name: asdict(spec) for name, spec in DATASET_SPECS.items()}
