"""MSSA-Transformer 项目的统一命令行入口。"""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from typing import Sequence

from . import __version__


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """一个可从统一入口启动的模型或实验。"""

    key: str
    name: str
    module: str
    category: str
    description: str
    recommended: bool = False


MODEL_ENTRIES: tuple[ModelEntry, ...] = (
    ModelEntry(
        "structured",
        "结构化缺失综合实验",
        "experiments.structured_missing_experiment",
        "综合实验",
        "SWaT/WADI 结构化缺失、基线对比与消融实验。",
        recommended=True,
    ),
    ModelEntry(
        "transformer",
        "基础 Transformer",
        "experiments.imputation.base_transformer_experiment",
        "插补模型",
        "不含额外融合模块的基础插补版本。",
    ),
    ModelEntry(
        "transformer-kl",
        "Transformer + KL",
        "experiments.imputation.kl_transformer_experiment",
        "插补模型",
        "加入 KL 散度约束的插补版本。",
    ),
    ModelEntry(
        "adaptive-transformer",
        "自适应 MSSA-Transformer",
        "experiments.imputation.adaptive_transformer_experiment",
        "插补模型",
        "带自适应融合策略的 MSSA-Transformer。",
    ),
    ModelEntry(
        "mssa-mlp",
        "MSSA + MLP + Transformer",
        "experiments.imputation.mssa_mlp_experiment",
        "插补模型",
        "使用 MLP 编码 MSSA 特征后与 Transformer 融合。",
    ),
    ModelEntry(
        "modal-fusion",
        "可学习模态融合",
        "experiments.imputation.modal_fusion_experiment",
        "插补模型",
        "学习原始时序模态和 MSSA 模态的融合权重。",
    ),
    ModelEntry(
        "spectral-bias",
        "谱注意力偏置",
        "experiments.imputation.spectral_bias_experiment",
        "插补模型",
        "将频谱信息作为 Transformer 注意力偏置。",
    ),
    ModelEntry(
        "spectral-modal",
        "谱感知模态融合（完整版本）",
        "experiments.imputation.spectral_modal_fusion_experiment",
        "插补模型",
        "结合 MSSA、谱注意力偏置和自适应模态融合。",
        recommended=True,
    ),
    ModelEntry(
        "classical",
        "传统插补基线",
        "experiments.baselines.classical_methods",
        "基线",
        "前向、后向、线性、多项式等传统插补方法。",
    ),
    ModelEntry(
        "pypots",
        "PyPOTS 深度学习基线",
        "experiments.baselines.baseline",
        "基线",
        "SAITS、BRITS、iTransformer 等 PyPOTS 基线。",
    ),
    ModelEntry(
        "gdn",
        "GDN 异常检测",
        "experiments.anomaly_detection.gdn_experiment",
        "异常检测",
        "基于图偏差网络的多变量时序异常检测。",
    ),
    ModelEntry(
        "gat-vae",
        "Masked GAT-VAE",
        "mssa_transformer.anomaly.gat_vae",
        "异常检测",
        "面向缺失观测的图注意力变分自编码器。",
    ),
    ModelEntry(
        "mssa-demo",
        "MSSA 示例",
        "examples.mssa_example",
        "示例",
        "仅运行 MSSA 分解和可视化示例。",
    ),
)

MODEL_REGISTRY = {entry.key: entry for entry in MODEL_ENTRIES}
LEGACY_DEFAULT_MODEL = "structured"


def configure_console_encoding() -> None:
    """优先使用 UTF-8 输出，避免 Windows 终端中的中文乱码。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一选择并运行 MSSA-Transformer 项目中的模型与实验。",
        epilog=(
            "模型参数会原样转发，例如："
            "python main.py --model structured -- --plan"
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=tuple(MODEL_REGISTRY),
        help="要运行的模型键；不提供时进入交互式菜单。",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出全部可用模型后退出。",
    )
    parser.add_argument(
        "--info",
        choices=tuple(MODEL_REGISTRY),
        metavar="MODEL",
        help="显示指定模型的说明和底层模块。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def format_model_list() -> str:
    """返回按类别分组的模型列表。"""

    lines: list[str] = []
    categories = dict.fromkeys(entry.category for entry in MODEL_ENTRIES)
    for category in categories:
        lines.append(f"\n[{category}]")
        for entry in MODEL_ENTRIES:
            if entry.category != category:
                continue
            marker = " *" if entry.recommended else ""
            lines.append(f"  {entry.key:<22} {entry.name}{marker}")
            lines.append(f"  {'':22} {entry.description}")
    lines.append("\n* 推荐入口")
    return "\n".join(lines).lstrip()


def print_model_info(entry: ModelEntry) -> None:
    marker = "是" if entry.recommended else "否"
    print(f"模型键：{entry.key}")
    print(f"名称：  {entry.name}")
    print(f"类别：  {entry.category}")
    print(f"推荐：  {marker}")
    print(f"模块：  {entry.module}")
    print(f"说明：  {entry.description}")


def choose_interactively() -> ModelEntry | None:
    """在终端中显示编号菜单并返回用户选择。"""

    print("MSSA-Transformer 模型选择器")
    print("=" * 48)
    for index, entry in enumerate(MODEL_ENTRIES, start=1):
        marker = "（推荐）" if entry.recommended else ""
        print(f"{index:>2}. {entry.name} [{entry.key}] {marker}")
    print(" q. 退出")

    while True:
        try:
            value = input("\n请选择模型编号或模型键：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if value.lower() in {"q", "quit", "exit"}:
            return None
        if value in MODEL_REGISTRY:
            return MODEL_REGISTRY[value]
        if value.isdigit() and 1 <= int(value) <= len(MODEL_ENTRIES):
            return MODEL_ENTRIES[int(value) - 1]
        print("输入无效，请重新选择。")


def run_model(entry: ModelEntry, forwarded_args: Sequence[str]) -> int:
    """延迟加载并运行模型模块。"""

    args = list(forwarded_args)
    if args[:1] == ["--"]:
        args = args[1:]

    print(f"正在启动：{entry.name} [{entry.key}]")
    print(f"底层模块：{entry.module}")
    if args:
        print(f"转发参数：{' '.join(args)}")
    print("-" * 48)

    original_argv = sys.argv[:]
    sys.argv = [entry.module, *args]
    try:
        runpy.run_module(entry.module, run_name="__main__")
    except ModuleNotFoundError as exc:
        print(
            f"\n缺少依赖模块 {exc.name!r}。请先执行：\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\n运行已由用户中止。", file=sys.stderr)
        return 130
    finally:
        sys.argv = original_argv
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析统一入口参数并启动所选模型。"""

    configure_console_encoding()
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)

    if args.list_models:
        print(format_model_list())
        return 0
    if args.info:
        print_model_info(MODEL_REGISTRY[args.info])
        return 0

    if args.model:
        entry = MODEL_REGISTRY[args.model]
    elif forwarded:
        # 兼容原入口：python main.py --plan / --ablation ...
        entry = MODEL_REGISTRY[LEGACY_DEFAULT_MODEL]
    elif sys.stdin.isatty():
        entry = choose_interactively()
        if entry is None:
            return 0
    else:
        parser.print_help()
        print("\n可用模型：\n" + format_model_list())
        return 2

    return run_model(entry, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
