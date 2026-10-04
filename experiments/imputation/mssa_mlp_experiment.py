"""
MSSA-MLP-Transformer缺失值填补完整示例
演示如何使用MSSA分解+MLP处理+Transformer模型进行时间序列缺失值填补
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 导入必要的模块
from spectraformer.mssa import MSSA, setup_chinese_fonts
from spectraformer.transformer.transformer_imputation_mssa_mlp import (
    MSSATransformerImputerMLPFixed,
)
from spectraformer.config import DATA_DIR, imputation_output_path

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


def train_mssa_mlp_transformer(train_data, val_data, mssa_model=None,
                              transformer_config=None, mlp_config=None,
                              fusion_method='concat'):
    """
    训练MSSA+MLP+Transformer填补模型

    参数:
        train_data: 训练数据（DataFrame）
        val_data: 验证数据（DataFrame）
        mssa_model: MSSA模型（可选）
        transformer_config: Transformer配置参数
        mlp_config: MLP配置参数
        fusion_method: 特征融合方法
    """
    print(f"\n训练MSSA+MLP+Transformer填补模型...")
    print(f"融合方法: {fusion_method}")

    # 默认Transformer配置
    if transformer_config is None:
        transformer_config = {
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        }

    # 默认MLP配置
    if mlp_config is None:
        mlp_config = {
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1
        }

    # 创建填补器
    imputer = MSSATransformerImputerMLPFixed(
        mssa_window_length=mssa_model.window_length if mssa_model else 48,
        mssa_n_components=mssa_model.n_components if mssa_model else None,
        transformer_config=transformer_config,
        mlp_config=mlp_config,
        fusion_method=fusion_method
    )

    # 训练
    print("开始训练...")
    print(f"MSSA特征维度: {train_data.shape[1] * (mssa_model.n_components if mssa_model else 0)}")
    print(f"MLP隐藏层: {mlp_config['hidden_dims']}")

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
    plt.plot(train_losses, label='Train Loss (MSSA+MLP+Transformer)')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss (MSSA+MLP+Transformer)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程中的损失变化 (MSSA+MLP+Transformer)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return imputer


def impute_and_evaluate_mssa_mlp(imputer, data_with_missing, true_data, missing_mask,
                                mssa_model=None, save_results=True):
    """
    使用训练好的MSSA+MLP+Transformer模型进行填补并评估

    参数:
        imputer: 训练好的填补器
        data_with_missing: 包含缺失值的数据
        true_data: 真实完整数据（用于评估）
        missing_mask: 缺失值掩码
        mssa_model: MSSA模型（可选）
        save_results: 是否保存结果
    """
    print("\n使用训练好的MSSA+MLP+Transformer模型填补缺失值...")

    # 填补缺失值
    imputed_data = imputer.impute_robust(
        data_with_missing.values,
        mssa_model=mssa_model,
        iterations=3,  # 减少迭代次数
        min_sequence_length=10,
        fallback_method='interpolate'
    )

    # 如果有真实数据，进行评估
    if true_data is not None:
        print("\n评估填补效果（MSSA+MLP+Transformer）...")
        metrics, feature_metrics = imputer.evaluate_imputation(
            true_data.values,
            imputed_data,
            missing_mask
        )

        print("\n整体填补性能指标（MSSA+MLP+Transformer）:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric:>10}: {value:>10.4f}")

        print("\n各特征填补性能（MSSA+MLP+Transformer）:")
        print("-" * 70)
        print(f"{'特征':>15} {'缺失数':>10} {'RMSE':>10} {'MAE':>10}")
        print("-" * 70)

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
        print("\n保存填补后的数据（MSSA+MLP+Transformer）...")
        imputed_df = pd.DataFrame(
            imputed_data,
            index=data_with_missing.index,
            columns=data_with_missing.columns
        )
        imputed_df.to_csv(imputation_output_path("imputed_data_mssa_mlp.csv"))
        print("已保存到: imputed_data_mssa_mlp.csv")

        # 保存评估报告
        if metrics:
            with open(
                imputation_output_path("imputation_report_mssa_mlp.txt"),
                'w',
                encoding='utf-8',
            ) as f:
                f.write("MSSA+MLP+Transformer缺失值填补报告\n")
                f.write("="*50 + "\n\n")
                f.write(f"数据集大小: {data_with_missing.shape}\n")
                f.write(f"缺失值总数: {missing_mask.sum()}\n")
                f.write(f"缺失比例: {missing_mask.sum() / missing_mask.size * 100:.2f}%\n")
                f.write(f"使用方法: MSSA+MLP+Transformer\n")
                f.write(f"融合方法: {imputer.fusion_method}\n")
                f.write(f"MLP隐藏层: {imputer.mlp_config['hidden_dims']}\n\n")

                f.write("整体填补性能:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")

                f.write("\n模型架构信息:\n")
                f.write(f"使用MSSA: {imputer.model.use_mssa}\n")
                f.write(f"总参数数量: {sum(p.numel() for p in imputer.model.parameters()):,}\n")

            print("评估报告已保存到: imputation_report_mssa_mlp.txt")

    return imputed_data, metrics


def visualize_results_mssa_mlp(data_with_missing, imputed_data, true_data, missing_mask,
                              time_range=(0, 500), features_to_plot=None):
    """
    可视化MSSA+MLP+Transformer填补结果
    """
    print("\n生成MSSA+MLP+Transformer填补结果可视化...")

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
                      c='red', s=20, label='填补值(MSSA+MLP)', zorder=5, alpha=0.8)

        ax.set_title(f'{feat} - 缺失值填补结果（MSSA+MLP+Transformer）')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def compare_fusion_methods(data_with_missing, true_data, missing_mask, mssa_model=None):
    """
    比较不同特征融合方法的效果
    """
    print("\n比较不同特征融合方法...")

    fusion_methods = ['concat', 'add', 'gate']
    results = []

    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    for fusion_method in fusion_methods:
        print(f"\n测试融合方法: {fusion_method}")

        # 训练模型
        imputer = train_mssa_mlp_transformer(
            train_data, val_data, mssa_model,
            transformer_config={
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'prediction_length': 1
            },
            mlp_config={
                'hidden_dims': [512, 256],
                'dropout': 0.1
            },
            fusion_method=fusion_method
        )

        # 填补和评估
        imputed_data, metrics = impute_and_evaluate_mssa_mlp(
            imputer, data_with_missing, true_data, missing_mask,
            mssa_model=mssa_model, save_results=False
        )

        # 记录结果
        if metrics:
            results.append({
                'fusion_method': fusion_method,
                **metrics
            })

    # 显示对比结果
    if results:
        results_df = pd.DataFrame(results)
        print("\n融合方法对比结果:")
        print(results_df.to_string(index=False))

        # 保存对比结果
        results_df.to_csv(
            imputation_output_path("mssa_mlp_fusion_methods_comparison.csv"),
            index=False,
        )
        print("\n对比结果已保存到: fusion_methods_comparison.csv")

    return results


def main():
    """
    主函数：MSSA+MLP+Transformer缺失值填补流程
    """
    print("="*60)
    print("MSSA+MLP+Transformer缺失值填补")
    print("="*60)

    # 配置参数
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing_60.csv")
    TRUE_DATA_PATH = str(DATA_DIR / "train_swat.csv")

    # MSSA参数
    MSSA_WINDOW_LENGTH = 48
    MSSA_N_COMPONENTS = 20

    # 是否使用MSSA
    USE_MSSA = True

    # 融合方法
    FUSION_METHOD = 'concat'  # 'concat', 'add', 'gate'

    # 1. 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        DATA_WITH_MISSING_PATH,
        TRUE_DATA_PATH,
        index_col=0
    )

    # 2. 准备训练和验证集
    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 3. MSSA分析（可选）
    mssa_model = None
    if USE_MSSA:
        print("\n对缺失值进行线性插值（仅用于MSSA）...")
        data_interpolated = train_data.interpolate(method='linear', limit_direction='both')

        # 执行MSSA
        mssa_model = perform_mssa_analysis(
            data_interpolated.values,
            window_length=MSSA_WINDOW_LENGTH,
            n_components=MSSA_N_COMPONENTS
        )

    # 4. 训练MSSA+MLP+Transformer
    imputer = train_mssa_mlp_transformer(
        train_data,
        val_data,
        mssa_model=mssa_model,
        transformer_config={
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        },
        mlp_config={
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1
        },
        fusion_method=FUSION_METHOD
    )

    # 5. 填补和评估
    imputed_data, metrics = impute_and_evaluate_mssa_mlp(
        imputer,
        data_with_missing,
        true_data,
        missing_mask,
        mssa_model=mssa_model,
        save_results=True
    )

    # 6. 可视化
    visualize_results_mssa_mlp(
        data_with_missing,
        imputed_data,
        true_data,
        missing_mask,
        time_range=(200, 400),
        features_to_plot=None
    )

    # 7. 打印模型信息
    print("\n模型架构总结:")
    print(f"使用MSSA: {USE_MSSA}")
    if USE_MSSA:
        print(f"MSSA窗口长度: {MSSA_WINDOW_LENGTH}")
        print(f"MSSA分量数: {MSSA_N_COMPONENTS}")
    print(f"融合方法: {FUSION_METHOD}")
    print(f"MLP隐藏层: {imputer.mlp_config['hidden_dims']}")
    print(f"总参数数量: {sum(p.numel() for p in imputer.model.parameters()):,}")

    print("\n处理完成！（MSSA+MLP+Transformer）")

    return imputed_data, metrics


def run_ablation_study(data_path, true_data_path=None):
    """
    运行消融研究：比较不同组合的效果
    """
    print("="*60)
    print("消融研究：比较不同模型组合")
    print("="*60)

    # 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        data_path, true_data_path, index_col=0
    )

    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 准备MSSA模型
    data_interpolated = train_data.interpolate(method='linear', limit_direction='both')
    mssa_model = perform_mssa_analysis(data_interpolated.values, window_length=48, n_components=20)

    # 实验配置
    experiments = [
        {
            'name': 'Baseline (仅Transformer)',
            'use_mssa': False,
            'fusion_method': 'concat',
            'mlp_hidden_dims': [256]
        },
        {
            'name': 'MSSA+简单MLP+Transformer',
            'use_mssa': True,
            'fusion_method': 'concat',
            'mlp_hidden_dims': [256]
        },
        {
            'name': 'MSSA+深度MLP+Transformer',
            'use_mssa': True,
            'fusion_method': 'concat',
            'mlp_hidden_dims': [512, 256, 128]
        },
        {
            'name': 'MSSA+深度MLP+Transformer(Add融合)',
            'use_mssa': True,
            'fusion_method': 'add',
            'mlp_hidden_dims': [512, 256, 128]
        },
        {
            'name': 'MSSA+深度MLP+Transformer(Gate融合)',
            'use_mssa': True,
            'fusion_method': 'gate',
            'mlp_hidden_dims': [512, 256, 128]
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n运行实验: {exp['name']}")
        print("-" * 50)

        # 训练模型
        imputer = train_mssa_mlp_transformer(
            train_data,
            val_data,
            mssa_model=mssa_model if exp['use_mssa'] else None,
            transformer_config={
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'prediction_length': 1
            },
            mlp_config={
                'hidden_dims': exp['mlp_hidden_dims'],
                'dropout': 0.1
            },
            fusion_method=exp['fusion_method']
        )

        # 评估
        imputed_data, metrics = impute_and_evaluate_mssa_mlp(
            imputer,
            data_with_missing,
            true_data,
            missing_mask,
            mssa_model=mssa_model if exp['use_mssa'] else None,
            save_results=False
        )

        # 记录结果
        if metrics:
            results.append({
                'experiment': exp['name'],
                'params': sum(p.numel() for p in imputer.model.parameters()),
                **metrics
            })

    # 汇总结果
    if results:
        results_df = pd.DataFrame(results)
        print("\n消融研究结果:")
        print(results_df.to_string(index=False))

        # 保存结果
        results_df.to_csv(
            imputation_output_path("mssa_mlp_ablation_study_results.csv"),
            index=False,
        )
        print("\n结果已保存到: ablation_study_results.csv")

    return results


if __name__ == "__main__":
    # 运行主流程
    imputed_data, metrics = main()

    # 如果想运行消融研究，取消下面的注释
    # ablation_results = run_ablation_study(
    #     "../data/train_continuous_missing.csv",
    #     "../data/test.csv"
    # )

    # 如果想比较不同融合方法，取消下面的注释
    # fusion_comparison = compare_fusion_methods(
    #     data_with_missing,
    #     true_data,
    #     missing_mask,
    #     mssa_model=None  # 如果有MSSA模型，传入这里
    # )
