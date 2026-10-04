"""项目公共路径配置。

所有路径均以项目根目录为基准，避免脚本因当前工作目录不同而读取或写入错误位置。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MASKED_DATA_DIR = DATA_DIR / "masked"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMPUTATION_OUTPUT_DIR = OUTPUTS_DIR / "imputation"
BASELINE_OUTPUT_DIR = OUTPUTS_DIR / "baselines"
STRUCTURED_EXPERIMENT_OUTPUT_DIR = OUTPUTS_DIR / "structured_missing"
FIGURE_OUTPUT_DIR = OUTPUTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def ensure_runtime_directories() -> None:
    """创建实验运行时可能写入的目录。"""

    for directory in (
        OUTPUTS_DIR,
        IMPUTATION_OUTPUT_DIR,
        BASELINE_OUTPUT_DIR,
        STRUCTURED_EXPERIMENT_OUTPUT_DIR,
        FIGURE_OUTPUT_DIR,
        CHECKPOINT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def imputation_output_path(filename: str) -> Path:
    """返回插补实验输出路径，并确保目录存在。"""

    IMPUTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return IMPUTATION_OUTPUT_DIR / filename


def baseline_output_path(filename: str) -> Path:
    """返回基线实验输出路径，并确保目录存在。"""

    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINE_OUTPUT_DIR / filename
