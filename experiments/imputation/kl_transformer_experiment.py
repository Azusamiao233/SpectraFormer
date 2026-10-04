"""
MSSA-Transformer缺失值填补完整示例 - 使用KL散度损失函数
演示如何使用MSSA分解和Transformer模型（KL损失）进行时间序列缺失值填补
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 导入必要的模块（假设已经有mssa.py和transformer_imputation_kl.py）
from mssa_transformer.mssa import MSSA, setup_chinese_fonts
from mssa_transformer.transformer.transformer_imputation_kl import (
    MSSATransformerImputerKLImproved,
)
from mssa_transformer.config import DATA_DIR, imputation_output_path

# 设置中文字体
setup_chinese_fonts()

# 设置随机种子
np.random.seed(42)


def load_data_with_missing(data_path: str,
                          true_data_path: str = None,
                          index_col: int = 0):
    """
    加载已经包含缺失值的数据

    参数:
        data_path: 包含缺失值的数据文件路径
        true_data_path: 完整数据文件路径（用于评估，可选）
        index_col: 索引列，默认为第一列

    返回:
        data_with_missing: 包含缺失值的DataFrame
        true_data: 完整数据DataFrame（如果提供了路径）
        missing_mask: 缺失值掩码
    """
    print(f"加载数据: {data_path}")

    # 加载包含缺失值的数据
    data_with_missing = pd.read_csv(data_path, index_col=index_col)
    print(f"数据形状: {data_with_missing.shape}")

    # 创建缺失值掩码
    missing_mask = data_with_missing.isna().values

    # 打印缺失值统计
    print("\n缺失值统计:")
    for col in data_with_missing.columns:
        missing_count = data_with_missing[col].isna().sum()
        missing_pct = missing_count / len(data_with_missing) * 100
        print(f"{col}: {missing_count} ({missing_pct:.1f}%)")

    # 如果提供了完整数据路径，加载它用于评估
    true_data = None
    if true_data_path:
        print(f"\n加载完整数据用于评估: {true_data_path}")
        true_data = pd.read_csv(true_data_path, index_col=index_col)

        # 确保列顺序一致
        true_data = true_data[data_with_missing.columns]
        print(f"完整数据形状: {true_data.shape}")

    return data_with_missing, true_data, missing_mask


def prepare_train_val_split(data_with_missing, split_ratio=0.8):
    """
    准备训练集和验证集

    参数:
        data_with_missing: 包含缺失值的DataFrame
        split_ratio: 训练集比例

    返回:
        train_data, val_data: 训练集和验证集
    """
    split_idx = int(len(data_with_missing) * split_ratio)

    train_data = data_with_missing.iloc[:split_idx]
    val_data = data_with_missing.iloc[split_idx:]

    print(f"\n数据集划分:")
    print(f"训练集: {train_data.shape}")
    print(f"验证集: {val_data.shape}")

    return train_data, val_data


def perform_mssa_analysis(data_for_mssa, window_length=100, n_components=20):
    """
    执行MSSA分析

    参数:
        data_for_mssa: 用于MSSA的数据（需要是完整的，可以是插值后的）
        window_length: MSSA窗口长度
        n_components: 保留的分量数
    """
    print(f"\n执行MSSA分析...")
    print(f"窗口长度: {window_length}")
    print(f"分量数: {n_components}")

    # 创建MSSA对象
    mssa = MSSA(
        window_length=window_length,
        n_components=n_components,
        verbose=True
    )

    # 拟合MSSA
    mssa.fit(data_for_mssa)

    # 绘制奇异谱
    print("\n绘制奇异谱...")
    mssa.plot_singular_spectrum(figsize=(12, 5))

    return mssa


def train_transformer_imputer_kl(train_data, val_data, mssa_model=None, config=None):
    """
    训练Transformer填补模型（使用KL散度损失）

    参数:
        train_data: 训练数据（DataFrame）
        val_data: 验证数据（DataFrame）
        mssa_model: MSSA模型（可选）
        config: 模型配置
    """
    print("\n训练Transformer填补模型（使用KL散度损失）...")

    # 默认配置
    if config is None:
        config = {
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        }

    # 创建填补器（使用KL损失版本）
    imputer = MSSATransformerImputerKLImproved(
        mssa_window_length=mssa_model.window_length if mssa_model else 48,
        mssa_n_components=mssa_model.n_components if mssa_model else None,
        transformer_config=config
    )

    # 训练
    print("开始训练（KL散度损失）...")
    train_losses, val_losses = imputer.train(
        train_data.values,
        val_data.values,
        mssa_model=mssa_model,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=50,
        patience=15,
        verbose=True
    )

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train KL Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence Loss')
    plt.title('训练过程中的KL散度损失变化')
    plt.legend()
    plt.grid(True)
    plt.show()

    return imputer


def impute_and_evaluate_kl(imputer, data_with_missing, true_data, missing_mask,
                          mssa_model=None, save_results=True):
    """
    使用训练好的模型（KL损失）进行填补并评估

    参数:
        imputer: 训练好的填补器（KL损失版本）
        data_with_missing: 包含缺失值的数据
        true_data: 真实完整数据（用于评估）
        missing_mask: 缺失值掩码
        mssa_model: MSSA模型（可选）
        save_results: 是否保存结果
    """
    print("\n使用训练好的模型（KL损失）填补缺失值...")

    # 填补缺失值
    imputed_data = imputer.impute(
        data_with_missing.values,
        mssa_model=mssa_model,
        iterations=3  # 迭代3次
    )

    # 如果有真实数据，进行评估
    if true_data is not None:
        print("\n评估填补效果（KL损失模型）...")
        metrics, feature_metrics = imputer.evaluate_imputation(
            true_data.values,
            imputed_data,
            missing_mask
        )

        print("\n整体填补性能指标（KL损失）:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:>10}: {value:>10.4f}")

        print("\n各特征填补性能（KL损失）:")
        print("-" * 60)
        print(f"{'特征':>15} {'缺失数':>10} {'RMSE':>10} {'MAE':>10}")
        print("-" * 60)

        for i, col in enumerate(data_with_missing.columns):
            if f'Feature_{i}' in feature_metrics:
                feat_metrics = feature_metrics[f'Feature_{i}']
                print(f"{col:>15} {feat_metrics['Missing_Count']:>10} "
                      f"{feat_metrics['RMSE']:>10.4f} {feat_metrics['MAE']:>10.4f}")
    else:
        metrics = None
        print("\n注意: 没有提供真实数据，无法评估填补效果")

    # 保存结果
    if save_results:
        print("\n保存填补后的数据（KL损失）...")
        imputed_df = pd.DataFrame(
            imputed_data,
            index=data_with_missing.index,
            columns=data_with_missing.columns
        )
        imputed_df.to_csv(imputation_output_path("imputed_data_kl.csv"))
        print("已保存到: imputed_data_kl.csv")

        # 保存评估报告
        if metrics:
            with open(imputation_output_path("imputation_report_kl.txt"), 'w', encoding='utf-8') as f:
                f.write("MSSA-Transformer缺失值填补报告（KL散度损失）\n")
                f.write("="*50 + "\n\n")
                f.write(f"数据集大小: {data_with_missing.shape}\n")
                f.write(f"缺失值总数: {missing_mask.sum()}\n")
                f.write(f"缺失比例: {missing_mask.sum() / missing_mask.size * 100:.2f}%\n")
                f.write(f"使用损失函数: KL散度损失\n\n")

                f.write("整体填补性能:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            print("评估报告已保存到: imputation_report_kl.txt")

    return imputed_data, metrics


def visualize_results_kl(data_with_missing, imputed_data, true_data, missing_mask,
                        time_range=(0, 500), features_to_plot=None):
    """
    可视化填补结果（KL损失）
    """
    print("\n生成填补结果可视化（KL损失）...")

    if features_to_plot is None:
        features_to_plot = data_with_missing.columns[:min(4, len(data_with_missing.columns))]

    n_features = len(features_to_plot)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 3*n_features))
    if n_features == 1:
        axes = [axes]

    start, end = time_range
    time_index = data_with_missing.index[start:end]

    for idx, (feat, ax) in enumerate(zip(features_to_plot, axes)):
        feat_idx = data_with_missing.columns.get_loc(feat)

        # 带缺失的数据
        ax.plot(time_index, data_with_missing[feat].iloc[start:end],
               'g--', label='带缺失值', alpha=0.5)

        # 真实数据（如果有）
        if true_data is not None:
            ax.plot(time_index, true_data[feat].iloc[start:end],
                   'b-', label='真实值', alpha=0.7, linewidth=1.5)

        # 填补的点
        mask_slice = missing_mask[start:end, feat_idx]
        if np.any(mask_slice):
            missing_times = time_index[mask_slice]
            imputed_values = imputed_data[start:end, feat_idx][mask_slice]
            ax.scatter(missing_times, imputed_values,
                      c='red', s=20, label='填补值(KL)', zorder=5, alpha=0.8)

        ax.set_title(f'{feat} - 缺失值填补结果（KL散度损失）')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：使用现成数据的MSSA-Transformer缺失值填补流程（KL损失）
    """
    print("="*60)
    print("MSSA-Transformer缺失值填补 - 使用KL散度损失")
    print("="*60)

    # 配置参数
    # 修改这些路径为您的实际数据文件路径
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing.csv")
    TRUE_DATA_PATH = str(DATA_DIR / "train_swat.csv")

    # MSSA参数
    MSSA_WINDOW_LENGTH = 48
    MSSA_N_COMPONENTS = 20

    # 是否使用MSSA
    USE_MSSA = True

    # 1. 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        DATA_WITH_MISSING_PATH,
        TRUE_DATA_PATH,
        index_col=0  # 假设第一列是索引
    )

    # 2. 准备训练和验证集
    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 3. MSSA分析（可选）
    mssa_model = None
    if USE_MSSA:
        # 对于MSSA，需要完整数据，这里使用插值填充
        print("\n对缺失值进行线性插值（仅用于MSSA）...")
        data_interpolated = train_data.interpolate(method='linear', limit_direction='both')

        # 执行MSSA
        mssa_model = perform_mssa_analysis(
            data_interpolated.values,
            window_length=MSSA_WINDOW_LENGTH,
            n_components=MSSA_N_COMPONENTS
        )

    # 4. 训练Transformer（使用KL损失）
    imputer = train_transformer_imputer_kl(
        train_data,
        val_data,
        mssa_model=mssa_model,
        config={
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        }
    )

    # 5. 填补和评估（KL损失）
    imputed_data, metrics = impute_and_evaluate_kl(
        imputer,
        data_with_missing,
        true_data,
        missing_mask,
        mssa_model=mssa_model,
        save_results=True
    )

    # 6. 可视化（KL损失）
    visualize_results_kl(
        data_with_missing,
        imputed_data,
        true_data,
        missing_mask,
        time_range=(200, 400),  # 可以调整显示的时间范围
        features_to_plot=None   # None表示显示前4个特征
    )

    print("\n处理完成！（使用KL散度损失）")

    return imputed_data, metrics


# 批量实验函数（可选）
def run_experiments_kl(data_path, true_data_path=None):
    """
    运行多组实验，比较不同配置的效果（使用KL损失）
    """
    print("="*60)
    print("批量实验：比较不同配置（KL散度损失）")
    print("="*60)

    # 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        data_path, true_data_path, index_col=0
    )

    # 实验配置
    experiments = [
        {
            'name': 'Baseline KL (No MSSA)',
            'use_mssa': False,
            'transformer_config': {
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'dim_feedforward': 256
            }
        },
        {
            'name': 'With MSSA KL (Small)',
            'use_mssa': True,
            'mssa_window': 24,
            'mssa_components': 10,
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512
            }
        },
        {
            'name': 'With MSSA KL (Large)',
            'use_mssa': True,
            'mssa_window': 48,
            'mssa_components': 20,
            'transformer_config': {
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 4,
                'dim_feedforward': 1024
            }
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n运行实验: {exp['name']}")
        print("-" * 40)

        # 准备数据
        train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

        # MSSA（如果需要）
        mssa_model = None
        if exp.get('use_mssa', False):
            data_interpolated = train_data.interpolate(method='linear', limit_direction='both')
            mssa_model = perform_mssa_analysis(
                data_interpolated.values,
                window_length=exp.get('mssa_window', 48),
                n_components=exp.get('mssa_components', 20)
            )

        # 训练（使用KL损失）
        imputer = train_transformer_imputer_kl(
            train_data,
            val_data,
            mssa_model=mssa_model,
            config=exp['transformer_config']
        )

        # 评估
        imputed_data, metrics = impute_and_evaluate_kl(
            imputer,
            data_with_missing,
            true_data,
            missing_mask,
            mssa_model=mssa_model,
            save_results=False
        )

        # 记录结果
        results.append({
            'experiment': exp['name'],
            **metrics
        })

    # 汇总结果
    results_df = pd.DataFrame(results)
    print("\n实验结果汇总（KL损失）:")
    print(results_df.to_string(index=False))

    # 保存结果
    results_df.to_csv(imputation_output_path("experiment_results_kl.csv"), index=False)
    print("\n结果已保存到: experiment_results_kl.csv")

    return results_df


# 对比不同损失函数的实验
def compare_loss_functions(data_path, true_data_path=None):
    """
    对比MSE和KL损失函数的效果
    """
    print("="*60)
    print("损失函数对比实验：MSE vs KL散度")
    print("="*60)

    # 这里需要同时导入MSE版本的填补器
    # from transformer.transformer_imputation import MSSATransformerImputer

    # 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        data_path, true_data_path, index_col=0
    )

    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 配置参数
    config = {
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'prediction_length': 1
    }

    results_comparison = []

    # KL损失实验
    print("\n训练KL损失模型...")
    imputer_kl = MSSATransformerImputerKLImproved(transformer_config=config)
    imputer_kl.train(train_data.values, val_data.values, epochs=50)
    imputed_data_kl = imputer_kl.impute(data_with_missing.values)
    metrics_kl, _ = imputer_kl.evaluate_imputation(true_data.values, imputed_data_kl, missing_mask)

    results_comparison.append({
        'Loss_Function': 'KL_Divergence',
        **metrics_kl
    })

    # 保存对比结果
    comparison_df = pd.DataFrame(results_comparison)
    print("\n损失函数对比结果:")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    # 运行主流程（KL损失）
    imputed_data, metrics = main()

    # 如果想运行批量实验，取消下面的注释
    # results = run_experiments_kl(
    #     "../data/train_continuous_missing.csv",
    #     "../data/test.csv"
    # )

    # 如果想对比不同损失函数，取消下面的注释
    # comparison_results = compare_loss_functions(
    #     "../data/train_continuous_missing.csv",
    #     "../data/test.csv"
    # )
