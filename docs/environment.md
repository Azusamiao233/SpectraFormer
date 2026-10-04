# Python 环境配置

## 1. 推荐环境

- 操作系统：Windows 10/11、Linux 或 macOS
- Python：3.10 或 3.11（项目元数据允许 3.10–3.12）
- 内存：建议至少 8 GB；完整 SWaT/WADI 实验建议 16 GB 以上
- GPU：可选。CPU 可运行代码，但神经网络训练会较慢

选择 Python 3.10/3.11 是为了兼顾 PyTorch、PyTorch Geometric、PyPOTS 和本项目较早的 pandas 写法。

## 2. 创建虚拟环境

在项目根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux/macOS 激活命令为：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. 安装依赖

### CPU 或不关心 CUDA 版本

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### NVIDIA GPU

先在 PyTorch 官方安装选择器中按操作系统、显卡驱动和 CUDA 版本生成安装命令，执行后再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

如果 `requirements.txt` 再次检查 `torch`，pip 会保留已满足版本范围的 CUDA 构建。PyTorch Geometric 2.3 及以上的基础功能可直接通过 `pip install torch-geometric` 安装；只有用到额外稀疏/采样算子时才需要按其官方文档安装对应 PyTorch/CUDA 的扩展 wheel。

## 4. 验证环境

```powershell
python -c "import numpy, pandas, scipy, sklearn, matplotlib, torch, torch_geometric, pypots; print('dependencies: OK'); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "from mssa_transformer.mssa import MSSA; from mssa_transformer.transformer import TransformerImputer; print('project imports: OK')"
```

## 5. 运行方式

安装完成后先查看统一入口中的模型列表：

```powershell
python main.py --list-models
python main.py --info spectral-modal
python main.py --model spectral-modal
```

直接执行 `python main.py` 可进入交互式菜单。安装为可编辑包后，也可以使用
`mssa-transformer --model spectral-modal`。需要复现单个底层实验时，仍可使用
`python -m experiments...` 的模块方式。

公共目录常量位于 `mssa_transformer/config.py`。新增脚本应从这里导入 `DATA_DIR`、`OUTPUTS_DIR` 或对应子目录，不要再写依赖当前工作目录的 `../data/...`。

## 6. 常见问题

- `ModuleNotFoundError: mssa_transformer`：确认当前目录是项目根目录，并执行过 `python -m pip install -e . --no-deps`。
- `torch.cuda.is_available()` 为 `False`：通常是安装了 CPU 版 PyTorch，或驱动与所选 CUDA wheel 不匹配；重新使用 PyTorch 官方安装选择器。
- PyG 扩展编译失败：先仅安装 `torch-geometric`。确需 `pyg-lib`、`torch-scatter` 或 `torch-sparse` 时，必须选择与 `torch.__version__` 和 `torch.version.cuda` 一致的 wheel。
- 数据文件不存在：确认文件位于 `data/`，并通过 `mssa_transformer.config.DATA_DIR` 构造路径，避免使用相对当前目录的 `../data/...`。

## 7. 官方安装参考

- [PyTorch 本地安装选择器](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric 安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [PyPOTS 安装说明](https://docs.pypots.com/en/dev/install.html)
