"""
可学习模态融合MSSA-Transformer缺失值填补完整测试示例
演示如何使用带有可学习模态融合的MSSA+Transformer模型进行时间序列缺失值填补
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Tuple, Optional
from mssa_transformer.config import DATA_DIR, imputation_output_path

warnings.filterwarnings('ignore')


# 导入必要的模块（假设这些模块在对应路径）
#from mssa.mssa import MSSA, setup_chinese_fonts
#from transformer.transformer_imputation_mssa_mlp_re import MSSATransformerImputerMLPFixed

# 如果没有实际的MSSA模块，创建一个模拟类
class MockMSSA:
    """模拟MSSA类用于测试"""

    def __init__(self, window_length=30, n_components=10, verbose=True):
        self.window_length = window_length
        self.n_components = n_components
        self.verbose = verbose

    def fit(self, data):
        """模拟fit方法"""
        self.data_shape = data.shape
        if self.verbose:
            print(f"MSSA fitted with data shape: {data.shape}")

    def reconstruct(self, groups):
        """模拟reconstruct方法，返回分解后的数据"""
        N, P = self.data_shape
        # 返回形状为 (N, P, K) 的模拟分解数据
        return np.random.randn(N, P, self.n_components)

    def plot_singular_spectrum(self, figsize=(12, 5)):
        """模拟绘图方法"""
        plt.figure(figsize=figsize)
        plt.plot(range(self.n_components), np.exp(-np.arange(self.n_components) * 0.2))
        plt.xlabel('Component')
        plt.ylabel('Singular Value')
        plt.title('Singular Spectrum (Simulated)')
        plt.grid(True)
        plt.show()


def create_synthetic_data(n_samples=1000, n_features=5, n_modes=3,
                          missing_ratio=0.2, missing_type='random'):
    """
    创建合成时间序列数据，包含不同模态的信号

    参数:
        n_samples: 样本数
        n_features: 特征数
        n_modes: 主要模态数
        missing_ratio: 缺失率
        missing_type: 缺失类型 ('random', 'continuous', 'mixed')
    """
    print(f"创建合成数据: {n_samples} 样本, {n_features} 特征, {n_modes} 主要模态")

    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))

    # 创建具有不同模态的数据
    for feat in range(n_features):
        # 趋势成分
        trend = 0.005 * t * (feat + 1)

        # 不同频率的季节性成分（模态）
        seasonal = 0
        for mode in range(n_modes):
            freq = 2 * np.pi * (mode + 1) / (50 + feat * 10)
            amplitude = (n_modes - mode) / n_modes  # 不同模态有不同重要性
            seasonal += amplitude * np.sin(freq * t + feat * np.pi / 4)

        # 噪声
        noise = 0.1 * np.random.randn(n_samples)

        data[:, feat] = trend + seasonal + noise

    # 创建缺失值
    data_with_missing = data.copy()
    missing_mask = np.zeros_like(data, dtype=bool)

    if missing_type == 'random':
        # 随机缺失
        missing_mask = np.random.random((n_samples, n_features)) < missing_ratio
    elif missing_type == 'continuous':
        # 连续缺失
        for feat in range(n_features):
            n_missing_blocks = int(n_samples * missing_ratio / 20)  # 每块平均20个点
            for _ in range(n_missing_blocks):
                start = np.random.randint(0, n_samples - 20)
                length = np.random.randint(10, 30)
                end = min(start + length, n_samples)
                missing_mask[start:end, feat] = True
    else:  # mixed
        # 混合缺失
        # 50%随机，50%连续
        half_ratio = missing_ratio / 2
        # 随机部分
        random_mask = np.random.random((n_samples, n_features)) < half_ratio
        missing_mask |= random_mask
        # 连续部分
        for feat in range(n_features):
            n_missing_blocks = int(n_samples * half_ratio / 20)
            for _ in range(n_missing_blocks):
                start = np.random.randint(0, n_samples - 20)
                length = np.random.randint(10, 30)
                end = min(start + length, n_samples)
                missing_mask[start:end, feat] = True

    data_with_missing[missing_mask] = np.nan

    # 转换为DataFrame
    columns = [f'Feature_{i}' for i in range(n_features)]
    index = pd.date_range('2020-01-01', periods=n_samples, freq='H')

    df_complete = pd.DataFrame(data, index=index, columns=columns)
    df_missing = pd.DataFrame(data_with_missing, index=index, columns=columns)

    print(f"缺失值统计:")
    for col in columns:
        n_missing = df_missing[col].isna().sum()
        pct_missing = n_missing / n_samples * 100
        print(f"  {col}: {n_missing} ({pct_missing:.1f}%)")

    return df_complete, df_missing, missing_mask


def test_modal_fusion_imputation(df_missing, df_complete, missing_mask,
                                 use_modal_fusion=True, modal_fusion_type='adaptive'):
    """
    测试带模态融合的填补方法

    参数:
        df_missing: 含缺失值的DataFrame
        df_complete: 完整的DataFrame（用于评估）
        missing_mask: 缺失值掩码
        use_modal_fusion: 是否使用模态融合
        modal_fusion_type: 模态融合类型
    """
    print(f"\n{'=' * 60}")
    print(f"测试模态融合填补")
    print(f"使用模态融合: {use_modal_fusion}")
    print(f"融合类型: {modal_fusion_type}")
    print(f"{'=' * 60}")

    # 1. 准备数据
    train_size = int(0.8 * len(df_missing))
    train_data = df_missing.iloc[:train_size]
    val_data = df_missing.iloc[train_size:]

    print(f"\n数据划分:")
    print(f"训练集: {train_data.shape}")
    print(f"验证集: {val_data.shape}")

    # 2. MSSA分析（使用插值数据）
    print("\n执行MSSA分析...")
    train_interpolated = train_data.interpolate(method='linear', limit_direction='both')

    # 使用模拟的MSSA或实际的MSSA
    mssa_model = MockMSSA(window_length=48, n_components=10)
    mssa_model.fit(train_interpolated.values)
    mssa_model.plot_singular_spectrum()

    # 3. 创建和训练模型
    print("\n创建并训练模型...")

    # 这里导入实际的模型类
    from mssa_transformer.transformer.transformer_imputation_mssa_mlp_re import (
        MSSATransformerImputerMLPFixed,
    )

    imputer = MSSATransformerImputerMLPFixed(
        mssa_window_length=48,
        mssa_n_components=10,
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
            'hidden_dims': [256, 128],
            'dropout': 0.1
        },
        fusion_method='concat',
        use_modal_fusion=use_modal_fusion,
        modal_fusion_type=modal_fusion_type
    )

    # 训练模型
    train_losses, val_losses = imputer.train(
        train_data.values,
        val_data.values,
        mssa_model=mssa_model,
        epochs=50,  # 减少epoch数用于测试
        batch_size=32,
        learning_rate=0.001,
        sequence_length=50,
        patience=10,
        verbose=True
    )

    # 4. 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'训练损失曲线 (Modal Fusion: {use_modal_fusion})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. 填补缺失值
    print("\n填补缺失值...")
    imputed_data = imputer.impute(
        df_missing.values,
        mssa_model=mssa_model,
        iterations=3,
        min_sequence_length=10
    )

    # 6. 评估结果
    print("\n评估填补效果...")
    overall_metrics, feature_metrics = imputer.evaluate_imputation(
        df_complete.values,
        imputed_data,
        missing_mask
    )

    print("\n整体性能指标:")
    print("-" * 40)
    for metric, value in overall_metrics.items():
        print(f"{metric:>10}: {value:>10.4f}")

    # 7. 可视化注意力权重（如果使用模态融合）
    if use_modal_fusion and modal_fusion_type != 'mlp':
        print("\n可视化模态注意力权重...")
        imputer.plot_modal_attention(n_samples=3)

    return imputed_data, overall_metrics, imputer


def compare_modal_fusion_methods(df_missing, df_complete, missing_mask):
    """
    比较不同模态融合方法的效果
    """
    print("\n" + "=" * 60)
    print("比较不同模态融合方法")
    print("=" * 60)

    methods = [
        ('不使用模态融合', False, 'mlp'),
        ('可学习模态融合', True, 'learnable'),
        ('自适应模态融合', True, 'adaptive'),
        ('原始MLP方法', True, 'mlp')
    ]

    results = []

    for name, use_fusion, fusion_type in methods:
        print(f"\n测试方法: {name}")
        print("-" * 40)

        try:
            imputed_data, metrics, imputer = test_modal_fusion_imputation(
                df_missing, df_complete, missing_mask,
                use_modal_fusion=use_fusion,
                modal_fusion_type=fusion_type
            )

            # 记录结果
            results.append({
                '方法': name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2'],
                '参数量': sum(p.numel() for p in imputer.model.parameters())
            })

        except Exception as e:
            print(f"错误: {e}")
            continue

    # 显示比较结果
    if results:
        results_df = pd.DataFrame(results)
        print("\n\n方法比较汇总:")
        print("=" * 80)
        print(results_df.to_string(index=False))

        # 保存结果
        results_df.to_csv(
            imputation_output_path("synthetic_modal_fusion_comparison.csv"),
            index=False,
        )
        print("\n结果已保存到: modal_fusion_comparison.csv")

        # 可视化比较
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2']
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = results_df[metric].values
            methods = results_df['方法'].values

            bars = ax.bar(range(len(methods)), values)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')

            # 添加数值标签
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    return results


def visualize_imputation_results(df_complete, df_missing, imputed_data, missing_mask,
                                 time_range=(100, 300), features=[0, 1]):
    """
    可视化填补结果
    """
    print("\n生成填补结果可视化...")

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    start, end = time_range
    time_index = df_complete.index[start:end]

    for idx, feat_idx in enumerate(features):
        ax = axes[idx]
        feat_name = df_complete.columns[feat_idx]

        # 真实完整数据
        ax.plot(time_index, df_complete.iloc[start:end, feat_idx],
                'b-', label='真实值', alpha=0.7, linewidth=2)

        # 带缺失的数据
        ax.plot(time_index, df_missing.iloc[start:end, feat_idx],
                'g--', label='观测值(含缺失)', alpha=0.5)

        # 填补的点
        mask_slice = missing_mask[start:end, feat_idx]
        if np.any(mask_slice):
            missing_times = time_index[mask_slice]
            imputed_values = imputed_data[start:end, feat_idx][mask_slice]
            ax.scatter(missing_times, imputed_values,
                       c='red', s=30, label='填补值', zorder=5, alpha=0.8)

        ax.set_title(f'{feat_name} - 模态融合填补结果')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def test_different_missing_patterns():
    """
    测试不同缺失模式下的填补效果
    """
    print("\n" + "=" * 60)
    print("测试不同缺失模式")
    print("=" * 60)

    missing_patterns = ['random', 'continuous', 'mixed']
    results = []

    for pattern in missing_patterns:
        print(f"\n测试缺失模式: {pattern}")
        print("-" * 40)

        # 创建数据
        df_complete, df_missing, missing_mask = create_synthetic_data(
            n_samples=1000,
            n_features=5,
            n_modes=3,
            missing_ratio=0.3,
            missing_type=pattern
        )

        # 测试自适应模态融合
        imputed_data, metrics, imputer = test_modal_fusion_imputation(
            df_missing, df_complete, missing_mask,
            use_modal_fusion=True,
            modal_fusion_type='adaptive'
        )

        results.append({
            '缺失模式': pattern,
            '缺失率': missing_mask.sum() / missing_mask.size * 100,
            **metrics
        })

        # 可视化部分结果
        visualize_imputation_results(
            df_complete, df_missing, imputed_data, missing_mask,
            time_range=(200, 400), features=[0, 1]
        )

    # 汇总结果
    results_df = pd.DataFrame(results)
    print("\n\n不同缺失模式结果汇总:")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df


def main():
    """
    主函数：MSSA+MLP+Transformer缺失值填补流程
    """
    print("="*60)
    print("MSSA+MLP+Transformer缺失值填补")
    print("="*60)

    # 配置参数
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing_40.csv")
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
            imputation_output_path("synthetic_ablation_study_results.csv"),
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
