# 贡献指南

## 开发流程

1. 使用 Python 3.10 或 3.11 创建独立虚拟环境。
2. 执行 `python -m pip install -r requirements.txt` 和 `python -m pip install -e . --no-deps`。
3. 新功能放入 `mssa_transformer/`，实验入口放入 `experiments/`，不要在项目根目录堆放脚本。
4. 数据路径统一使用 `mssa_transformer.config`，生成文件写入 `outputs/` 或 `artifacts/`。
5. 提交前执行 `python -m unittest discover -s tests -v` 和 `python -m compileall -q mssa_transformer experiments examples scripts tests`。

## 新增模型版本

新增模型后，在 `mssa_transformer/cli.py` 的 `MODEL_ENTRIES` 中登记模型键、入口模块和说明，确保 `python main.py --list-models` 可以发现它。

## 提交规范

- 每个提交只处理一个明确问题。
- 不提交数据集、模型权重、缓存和实验输出。
- 对外部算法或代码注明来源与许可证。
- 行为变更应同时更新测试与 README。

