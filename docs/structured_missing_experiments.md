# 40% 结构化缺失补充实验

## 实验矩阵

缺失率固定为 `γ = 40%`，不再重复 20%/40%/60% 的 missing-rate robustness。
本实验只改变缺失结构，共包含 24 项测试：

| 维度 | 取值 |
|---|---|
| 数据集 | SWaT、WADI |
| 缺失模式 | Multi-Sensor、Sensor–Actuator、Subsystem |
| 方法 | SAITS、BRITS、Transformer、Proposed |

同一数据集上的每种插补模型只训练一次，再复用于三个缺失模式，避免因重复训练的
随机波动干扰模式间比较。

## 三种模式的严格定义

这里的 40% 指“所选变量组内部的缺失单元格数 / 该变量组全部单元格数”。同组变量
使用完全相同的连续时间块。默认生成 4 个互不重叠、内部至少间隔一个观测点的块，
块长度之和严格等于测试行数的 40%（取整后）。索引、时间戳和攻击标签始终保留。

- **Multi-Sensor**：同时遮蔽 4 个连续型传感器通道。
- **Sensor–Actuator**：同时遮蔽一个有直接工艺关系的传感器—执行器组。
- **Subsystem**：同时遮蔽同一处理阶段或控制回路内的多个变量。

变量组集中定义在 `mssa_transformer/structured_missingness.py`，运行产生的每个
`metrics.json` 也会保存实际变量名和半开区间 `[start, end)`，便于审计。

## 变量组

### SWaT

| 模式 | 变量组 | 工艺含义 |
|---|---|---|
| Multi-Sensor | FIT101, LIT101, AIT201, FIT201 | P1–P2 的 4 个连续传感器 |
| Sensor–Actuator | FIT101, LIT101, MV101, P101 | P1 进水流量/液位及阀门、泵 |
| Subsystem | AIT201, AIT202, AIT203, FIT201, MV201, P201–P206 | P2 化学加药阶段 |

### WADI

| 模式 | 变量组 | 工艺含义 |
|---|---|---|
| Multi-Sensor | 1_AIT_001_PV–1_AIT_004_PV | P1 的 4 个连续水质传感器 |
| Sensor–Actuator | 1_FIT_001_PV, 1_LT_001_PV, 1_MV_001_STATUS, 1_P_005_STATUS | P1 流量/液位及进水阀、P1 泵 005 |
| Subsystem | 2_FIC_101_CO/PV/SP, 2_FQ_101_PV, 2_LS_101_AH/AL, 2_MCV_101_CO, 2_MV_101_STATUS, 2_SV_101_STATUS | P2B 用户支路 101 |

选择采用公开的测试床工艺阶段和设备标签语义，而不是根据测试集效果反向挑选变量，
以免产生测试信息泄漏。若论文需要改组，只修改集中配置即可，生成器会在运行前检查
所有列是否真实存在。

## 输出指标

插补误差只在人工遮蔽位置计算，输出 RMSE、MAE、SMAPE、R²、按标准差归一化的
NRMSE、有效预测率以及逐变量指标。同时记录：

- `test_seconds`：纯测试阶段墙钟时间；本文方法包含测试时 MSSA 分解。
- `peak_process_rss_mib`：测试期间进程峰值常驻内存。
- `process_rss_increase_mib`：相对测试开始时新增的峰值内存。
- `peak_cuda_allocated_mib` / `peak_cuda_reserved_mib`：使用 CUDA 时的显存峰值。
- `imputed_cells_per_second`：每秒完成的有效遮蔽单元格数。

## 运行方式

验证 24 项矩阵和字段：

```powershell
python main.py --model structured -- --plan
```

运行默认可复现实验：

```powershell
python main.py --model structured
```

`main.py` 中的 `proposed` 方法使用当前最新的组合模型
`transformer_imputation_mssa_mlp_re_bias.SpectralMSSATransformerImputer`，默认从头
训练，不读取 `artifacts/checkpoints/best_model.pth`。

默认使用每个数据集前 3000 个训练点和测试段前 2000 个点，主要是控制 MSSA 轨迹
矩阵内存。正式报告时可在硬件允许范围内提高 `--max-train-rows` 和
`--max-test-rows`，并在所有方法上保持一致。例如：

```powershell
python main.py --model structured -- `
  --max-train-rows 10000 --max-test-rows 10000 --epochs 100 --save-imputed
```

生成六份独立的带缺失 CSV、压缩 mask 和元数据（可选；批量实验本身会在线生成）：

```powershell
python -m scripts.data_preprocessing.generate_structured_missing
```

若不希望复制大型 CSV，可仅保存压缩 mask 和元数据：

```powershell
python -m scripts.data_preprocessing.generate_structured_missing --metadata-only
```

汇总表保存在 `outputs/structured_missing/summary.csv`，各项详情保存在
`outputs/structured_missing/<dataset>/<pattern>/<method>/metrics.json`。

## 新增消融实验

在 Full Model 参考行之外补充以下两行：

| 表格名称 | mSSA | Modal fusion | Transformer Encoder | 谱偏置 | 输出头 |
|---|---:|---:|---:|---:|---|
| w/o mSSA | × | ×（无模态输入） | ✓ | ✓ | Transformer 输出投影 |
| w/o Transformer Encoder | ✓ | ✓ | × | ×（随编码器移除） | 轻量 MLP 重构头 |

轻量重构头连接融合序列的最后时刻特征和全窗口均值池化特征，再通过单隐藏层 MLP
预测下一时刻。它保留基本时间上下文，但不含自注意力或 Transformer 前馈层。

在两套数据的 Multi-Sensor 40% 设置上生成 Full Model 与两条消融结果：

```powershell
python main.py --model structured -- `
  --ablation --patterns multi_sensor --output-dir outputs/ablation
```

如需在全部三种结构化模式上消融，省略 `--patterns multi_sensor`。汇总表中的
`ablation_variant`、`use_mssa`、`use_modal_fusion`、
`use_transformer_encoder`、`use_spectral_bias` 和 `parameter_count` 字段可直接用于
论文消融表，测试耗时和内存统计与主实验保持一致。
