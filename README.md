# SpectraFormer

**SpectraFormer** is a spectral-aware Transformer project for missing-value imputation in multivariate industrial time series. It combines multivariate singular spectrum analysis (MSSA), Transformer-based imputation, spectral attention bias, adaptive modality fusion, classical interpolation methods, and PyPOTS baselines.

> Current version: `0.3.0`. The root-level [`main.py`](main.py) is the unified entry point for selecting and running different model variants.

## Highlights

- Multiple SpectraFormer variants available through one command-line interface.
- Structured missingness generation for SWaT and WADI datasets.
- Experiments for 40% missing-data comparison and component ablation.
- Classical interpolation, SAITS, BRITS, iTransformer, and other baselines.
- Clear separation between reusable source code, experiments, examples, scripts, and tests.
- Local datasets, generated outputs, and model checkpoints are excluded from Git.

## Project Structure

```text
.
├── main.py                         # Unified model launcher
├── spectraformer/                  # Reusable Python package
│   ├── cli.py                      # Model registry and command-line interface
│   ├── config.py                   # Shared project paths
│   ├── structured_missingness.py   # Structured missingness generation
│   ├── mssa/                       # SSA and MSSA implementations
│   └── transformer/                # Transformer imputation models
├── experiments/                    # Training, baseline, and ablation experiments
├── examples/                       # MSSA and data-flow examples
├── scripts/data_preprocessing/     # CSV preprocessing utilities
├── tests/                          # Unit tests
└── docs/                           # Environment and experiment documentation
```

## Available Models and Experiments

| Key | Model or experiment | Description |
|---|---|---|
| `structured` | Structured missingness experiment | SWaT/WADI experiments, missingness patterns, baselines, and ablations |
| `transformer` | Base Transformer | Basic time-series imputation model |
| `transformer-kl` | Transformer + KL | Transformer imputation with KL-divergence regularization |
| `adaptive-transformer` | Adaptive SpectraFormer | Adaptively integrates MSSA information |
| `mssa-mlp` | MSSA + MLP + Transformer | Encodes the MSSA modality with an MLP before fusion |
| `modal-fusion` | Learnable modality fusion | Learns fusion weights for raw and MSSA modalities |
| `spectral-bias` | Spectral attention bias | Injects frequency-domain information into attention |
| `spectral-modal` | Full SpectraFormer variant | Combines MSSA, spectral bias, and adaptive modality fusion |
| `classical` | Classical baselines | Forward, backward, linear, and polynomial interpolation |
| `pypots` | PyPOTS baselines | SAITS, BRITS, iTransformer, and related models |
| `mssa-demo` | MSSA example | Runs an MSSA decomposition and visualization example |

List the registered entries or inspect one entry without loading training dependencies:

```bash
python main.py --list-models
python main.py --info spectral-modal
```

## Installation

Python 3.10 or 3.11 is recommended.

### Windows PowerShell

```powershell
cd "C:\path\to\SpectraFormer"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

### Linux or macOS

```bash
cd /path/to/SpectraFormer
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e . --no-deps
```

For NVIDIA GPU acceleration, install the PyTorch build that matches your CUDA environment before installing the remaining dependencies. Additional setup notes are available in [`docs/environment.md`](docs/environment.md).

## Usage

### Interactive model selection

```bash
python main.py
```

The launcher displays a numbered menu. Select a model by number or by its key.

### Direct model selection

```bash
# Recommended full model
python main.py --model spectral-modal

# Base Transformer
python main.py --model transformer

# Classical interpolation baselines
python main.py --model classical
```

After installing the project in editable mode, the console command is also available:

```bash
spectra-former --list-models
spectra-former --model spectral-modal
```

Arguments for an underlying experiment can be placed after `--`:

```bash
python main.py --model structured -- --plan
```

The original structured-experiment arguments remain supported directly:

```bash
python main.py --plan
python main.py --ablation --patterns multi_sensor --output-dir outputs/ablation
```

## Structured Missingness Experiments

The structured experiment supports SWaT and WADI, multiple missingness patterns, baseline comparisons, and ablation studies. Inspect the experiment matrix before starting a full run:

```bash
python main.py --model structured -- --plan
```

Run a selected dataset and missingness pattern:

```bash
python main.py --model structured -- \
  --datasets swat \
  --patterns multi_sensor \
  --methods linear proposed \
  --epochs 10
```

Run an ablation experiment:

```bash
python main.py --model structured -- \
  --ablation \
  --patterns multi_sensor \
  --output-dir outputs/ablation
```

See [`docs/structured_missing_experiments.md`](docs/structured_missing_experiments.md) for the experiment definitions.

## Local Data and Generated Files

Datasets, generated results, and model checkpoints are not distributed with this repository. Prepare them locally as required by the selected experiment. Shared paths are defined in [`spectraformer/config.py`](spectraformer/config.py); avoid hard-coding absolute paths in new code.

The following local directories are ignored by Git:

- `data/` — local datasets and generated masks
- `outputs/` — experiment results and plots
- `artifacts/` — checkpoints and other model artifacts

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q spectraformer experiments examples scripts tests
```

Quickly validate the launcher without importing training dependencies:

```bash
python main.py --list-models
python main.py --info structured
```

## Adding a Model

1. Add reusable implementation code to the appropriate `spectraformer/` subpackage.
2. Add the training or evaluation entry point under `experiments/`.
3. Register the entry in `MODEL_ENTRIES` inside [`spectraformer/cli.py`](spectraformer/cli.py).
4. Add tests and update the model table in this README.

## Repository

GitHub: [Azusamiao233/SpectraFormer](https://github.com/Azusamiao233/SpectraFormer)

Before redistributing datasets or third-party code, verify the applicable SWaT, WADI, and dependency licenses. Add a project license before publishing or redistributing the repository as open-source software.
