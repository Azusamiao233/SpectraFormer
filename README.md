# SpectraFormer

**SpectraFormer** 是一个面向多变量工业时间序列缺失值补全的谱感知 Transformer 项目。项目包含 MSSA 分解、Transformer 插补、谱注意力偏置、自适应模态融合，以及传统方法和 PyPOTS 基线。

> 当前版本：`0.3.0`。根目录 [main.py](main.py) 是统一入口，可通过交互菜单或命令行选择不同模型版本。

## 功能概览

- 多个 SpectraFormer 演进版本，可从同一个入口启动。
- SWaT/WADI 结构化缺失生成、40% 缺失对比和消融实验。
- 传统插值、SAITS、BRITS、iTransformer 等基线。
- 源码、实验、示例和预处理工具分目录管理。
- 配套单元测试、环境文档和贡献规范。

## 项目结构

```text
.
├── main.py                         # 统一模型选择入口
├── spectraformer/                  # 可复用核心包
│   ├── cli.py                      # 模型注册表与命令行界面
│   ├── config.py                   # 项目公共路径
│   ├── structured_missingness.py   # 结构化缺失生成
│   ├── mssa/                       # SSA/MSSA 算法
│   └── transformer/                # Transformer 插补模型及变体
├── experiments/                    # 训练、基线、消融与综合实验
├── examples/                       # MSSA 和数据流示例
├── scripts/data_preprocessing/     # CSV 预处理工具
├── tests/                          # 单元测试
└── docs/                           # 环境及实验说明
```

## 模型版本

| 入口键 | 模型/实验 | 主要差异 |
|---|---|---|
| `structured` | 结构化缺失综合实验 | SWaT/WADI、多缺失模式、基线和消融实验 |
| `transformer` | 基础 Transformer | 基础时序插补版本 |
| `transformer-kl` | Transformer + KL | 加入 KL 散度约束 |
| `adaptive-transformer` | 自适应 SpectraFormer | 自适应融合 MSSA 信息 |
| `mssa-mlp` | MSSA + MLP + Transformer | 使用 MLP 编码 MSSA 模态 |
| `modal-fusion` | 可学习模态融合 | 学习原始模态与 MSSA 模态权重 |
| `spectral-bias` | 谱注意力偏置 | 频谱信息参与注意力计算 |
| `spectral-modal` | 完整插补版本 | MSSA + 谱偏置 + 自适应模态融合 |
| `classical` | 传统基线 | 前向、后向、线性和多项式插补 |
| `pypots` | PyPOTS 基线 | SAITS、BRITS、iTransformer 等 |

完整列表以命令输出为准：

```powershell
python main.py --list-models
python main.py --info spectral-modal
```

## 环境安装

推荐 Python 3.10 或 3.11。Windows PowerShell：

```powershell
cd "C:\path\to\SpectraFormer"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Linux/macOS：

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e . --no-deps
```

如需 NVIDIA GPU，请先按 PyTorch 官方说明安装与驱动匹配的 CUDA 版本，再安装其余依赖。更详细说明见 [docs/environment.md](docs/environment.md)。

## 使用统一入口

### 交互式选择

```powershell
python main.py
```

程序会显示编号菜单，输入编号或模型键即可启动。

### 命令行选择

```powershell
# 完整谱感知模态融合版本
python main.py --model spectral-modal

# 基础 Transformer
python main.py --model transformer

# 传统插补基线
python main.py --model classical
```

安装为可编辑包后，也可以使用：

```powershell
spectra-former --list-models
spectra-former --model spectral-modal
```

模型自己的参数放在 `--` 后面。例如只检查结构化实验计划：

```powershell
python main.py --model structured -- --plan
```

为兼容旧命令，下面的写法仍然有效：

```powershell
python main.py --plan
python main.py --ablation --patterns multi_sensor --output-dir outputs/ablation
```

## 结构化缺失实验

默认实验覆盖 SWaT/WADI、三种结构化缺失模式、多个基线与本文方法。建议先查看实验矩阵：

```powershell
python main.py --model structured -- --plan
```

运行指定数据集和模式：

```powershell
python main.py --model structured -- `
  --datasets swat `
  --patterns multi_sensor `
  --methods linear proposed `
  --epochs 10
```

消融实验：

```powershell
python main.py --model structured -- `
  --ablation `
  --patterns multi_sensor `
  --output-dir outputs/ablation
```

详细实验定义见 [docs/structured_missing_experiments.md](docs/structured_missing_experiments.md)。

## 本地数据与运行结果

数据集、运行结果和模型权重不随仓库发布，相关目录由使用者在本地准备或由程序运行时生成。公共路径定义在 `spectraformer/config.py`，新增代码不应硬编码绝对路径。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q spectraformer experiments examples scripts tests
```

只验证统一入口、不加载训练依赖：

```powershell
python main.py --list-models
python main.py --info structured
```

## 添加新模型

1. 将可复用模型放入 `spectraformer/` 对应子包。
2. 将训练或评估入口放入 `experiments/`。
3. 在 `spectraformer/cli.py` 的 `MODEL_ENTRIES` 中注册入口。
4. 添加测试，并更新本 README 的模型表。

更多开发约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 项目仓库

GitHub：[czh4994/SpectraFormer](https://github.com/czh4994/SpectraFormer)

公开使用前请补充合适的 `LICENSE`，并确认 SWaT/WADI 数据及第三方代码的授权要求。
