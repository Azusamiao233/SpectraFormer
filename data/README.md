# 数据目录

数据文件不会提交到普通 Git 仓库，以避免仓库体积过大。请在本目录按以下结构放置数据：

```text
data/
├── train_swat.csv
├── train_continuous_missing.csv
├── train_continuous_missing_40.csv
├── train_continuous_missing_60.csv
├── train_wadi.csv
├── train_wadi_continuous_missing_20.csv
├── train_wadi_continuous_missing_40.csv
├── train_wadi_continuous_missing_60.csv
├── gdn/
│   ├── train.csv
│   └── test.csv
└── masked/
    ├── swat_train.csv
    └── swat_test_random_missing.csv
```

如果需要公开数据，请确认 SWaT/WADI 的原始许可，并优先在 README 中提供合法下载地址或预处理说明，而不是直接提交数据文件。

