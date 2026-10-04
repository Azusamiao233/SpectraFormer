"""
谱感知SpectraFormer缺失值填补测试代码
基于spectral attention bias的改进版本，结合MSSA特征处理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import torch
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# 导入基础模块
from spectraformer.mssa.mssa1 import MSSA, setup_chinese_fonts
from spectraformer.transformer.transformer_imputation_mssa_mlp_bias import (
    SpectralMSSATransformerImputer,
)
from spectraformer.config import DATA_DIR, imputation_output_path
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

# 设置中文字体
setup_chinese_fonts()

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


def create_synthetic_data(n_samples: int = 1000, n_features: int = 5,
                         missing_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建具有明显频率特征的合成数据"""

    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))

    # 为每个特征创建不同的频率模式
    for i in range(n_features):
        # 趋势分量
        trend = 0.01 * t + i * 0.5

        # 多个周期性分量（不同频率）
        seasonal1 = 2.0 * np.sin(2 * np.pi * t / 50 + i * 0.3)  # 主要周期
        seasonal2 = 1.5 * np.sin(2 * np.pi * t / 20 + i * 0.5)  # 次要周期
        seasonal3 = 1.0 * np.sin(2 * np.pi * t / 100 + i * 0.2) # 长周期
        seasonal4 = 0.8 * np.sin(2 * np.pi * t / 10 + i * 0.7)  # 短周期

        # 高频噪声
        noise = 0.3 * np.random.randn(n_samples)

        # 组合所有分量
        data[:, i] = trend + seasonal1 + seasonal2 + seasonal3 + seasonal4 + noise

    # 创建缺失值
    missing_mask = np.random.random((n_samples, n_features)) < missing_ratio
    data_with_missing = data.copy()
    data_with_missing[missing_mask] = np.nan

    return data, data_with_missing, missing_mask


def load_data_with_missing(data_path: str, true_data_path: str = None, index_col: int = 0):
    """加载包含缺失值的数据"""
    print(f"加载数据: {data_path}")

    data_with_missing = pd.read_csv(data_path, index_col=index_col)
    print(f"数据形状: {data_with_missing.shape}")

    missing_mask = data_with_missing.isna().values

    print("\n缺失值统计:")
    for col in data_with_missing.columns:
        missing_count = data_with_missing[col].isna().sum()
        missing_pct = missing_count / len(data_with_missing) * 100
        print(f"{col}: {missing_count} ({missing_pct:.1f}%)")

    true_data = None
    if true_data_path:
        print(f"\n加载完整数据用于评估: {true_data_path}")
        true_data = pd.read_csv(true_data_path, index_col=index_col)
        true_data = true_data[data_with_missing.columns]
        print(f"完整数据形状: {true_data.shape}")

    return data_with_missing, true_data, missing_mask


def visualize_results(data_with_missing, imputed_data, true_data, missing_mask,
                     time_range=(0, 500), features_to_plot=None):
    """可视化填补结果"""
    print("\n生成填补结果可视化...")

    if isinstance(data_with_missing, pd.DataFrame):
        data_with_missing_values = data_with_missing.values
        columns = data_with_missing.columns
        index = data_with_missing.index
    else:
        data_with_missing_values = data_with_missing
        columns = [f'Feature_{i}' for i in range(data_with_missing.shape[1])]
        index = range(len(data_with_missing))

    if features_to_plot is None:
        features_to_plot = columns[:min(4, len(columns))]

    n_features = len(features_to_plot)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 3*n_features))
    if n_features == 1:
        axes = [axes]

    start, end = time_range
    time_index = index[start:end] if hasattr(index, '__getitem__') else range(start, end)

    for idx, (feat, ax) in enumerate(zip(features_to_plot, axes)):
        if isinstance(feat, str) and feat in columns:
            feat_idx = list(columns).index(feat)
        else:
            feat_idx = idx

        # 带缺失的数据
        ax.plot(time_index, data_with_missing_values[start:end, feat_idx],
               'g--', label='带缺失值', alpha=0.5)

        # 真实数据（如果有）
        if true_data is not None:
            if isinstance(true_data, pd.DataFrame):
                true_values = true_data.values
            else:
                true_values = true_data
            ax.plot(time_index, true_values[start:end, feat_idx],
                   'b-', label='真实值', alpha=0.7, linewidth=1.5)

        # 填补的点
        mask_slice = missing_mask[start:end, feat_idx]
        if np.any(mask_slice):
            missing_times = np.array(time_index)[mask_slice]
            imputed_values = imputed_data[start:end, feat_idx][mask_slice]
            ax.scatter(missing_times, imputed_values,
                      c='red', s=20, label='谱感知填补值', zorder=5, alpha=0.8)

        feat_name = feat if isinstance(feat, str) else f'Feature_{feat_idx}'
        ax.set_title(f'{feat_name} - 谱感知SpectraFormer填补结果')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def visualize_spectral_analysis(data: np.ndarray, feature_indices: List[int] = [0, 1],
                               time_range: Tuple[int, int] = (0, 500)):
    """可视化频谱分析"""
    print("\n进行频谱分析可视化...")

    start, end = time_range
    n_features = len(feature_indices)

    fig, axes = plt.subplots(n_features, 2, figsize=(15, 4*n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    for idx, feat_idx in enumerate(feature_indices):
        feature_data = data[start:end, feat_idx]

        # 时域信号
        ax1 = axes[idx, 0]
        ax1.plot(range(len(feature_data)), feature_data, 'b-', alpha=0.8)
        ax1.set_title(f'特征 {feat_idx} - 时域信号')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('幅值')
        ax1.grid(True, alpha=0.3)

        # 频域分析
        ax2 = axes[idx, 1]

        # 计算功率谱
        fft_result = np.fft.rfft(feature_data)
        power_spectrum = np.abs(fft_result) ** 2
        freqs = np.fft.rfftfreq(len(feature_data))

        # 排除DC分量并绘制
        if len(power_spectrum) > 1:
            ax2.semilogy(freqs[1:], power_spectrum[1:], 'g-', alpha=0.8)

            # 标记主导频率
            top_k = 5
            if len(power_spectrum[1:]) >= top_k:
                dominant_indices = np.argsort(power_spectrum[1:])[::-1][:top_k] + 1
                ax2.scatter(freqs[dominant_indices], power_spectrum[dominant_indices],
                           c='red', s=50, zorder=5, label=f'Top {top_k} 主导频率')
                ax2.legend()

        ax2.set_title(f'特征 {feat_idx} - 功率谱')
        ax2.set_xlabel('频率')
        ax2.set_ylabel('功率')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_imputation_methods(true_data: np.ndarray, data_with_missing: np.ndarray,
                              missing_mask: np.ndarray, spectral_imputed: np.ndarray):
    """比较不同填补方法的效果"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("\n比较不同填补方法...")

    # 确保spectral_imputed没有NaN
    if np.any(np.isnan(spectral_imputed)):
        print("处理谱感知填补结果中的NaN值...")
        spectral_df = pd.DataFrame(spectral_imputed)
        spectral_imputed = spectral_df.interpolate(
            method='linear', limit_direction='both'
        ).fillna(method='bfill').fillna(method='ffill').values

        # 如果还有NaN，用均值填充
        if np.any(np.isnan(spectral_imputed)):
            for i in range(spectral_imputed.shape[1]):
                col_mean = np.nanmean(spectral_imputed[:, i])
                if not np.isnan(col_mean):
                    spectral_imputed[np.isnan(spectral_imputed[:, i]), i] = col_mean
                else:
                    spectral_imputed[np.isnan(spectral_imputed[:, i]), i] = 0.0

    # 简单插值填补作为基线
    df_missing = pd.DataFrame(data_with_missing)
    interpolated = df_missing.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill').values

    # 均值填补
    mean_filled = data_with_missing.copy()
    for i in range(data_with_missing.shape[1]):
        col_mean = np.nanmean(data_with_missing[:, i])
        mean_filled[missing_mask[:, i], i] = col_mean

    # 只在缺失位置计算指标
    true_missing = true_data[missing_mask]
    interpolated_missing = interpolated[missing_mask]
    mean_missing = mean_filled[missing_mask]
    spectral_missing = spectral_imputed[missing_mask]

    methods = {
        '线性插值': interpolated_missing,
        '均值填补': mean_missing,
        '谱感知Transformer': spectral_missing
    }

    print("\n各方法填补效果对比:")
    print("-" * 80)
    print(f"{'方法':>15} {'RMSE':>12} {'MAE':>12} {'MAPE(%)':>12} {'R²':>12}")
    print("-" * 80)

    for method_name, predictions in methods.items():
        try:
            # 检查是否有NaN
            if np.any(np.isnan(predictions)) or np.any(np.isnan(true_missing)):
                print(f"{method_name:>15} {'NaN detected':>48}")
                continue

            rmse = np.sqrt(mean_squared_error(true_missing, predictions))
            mae = mean_absolute_error(true_missing, predictions)

            # 计算MAPE时避免除零
            mape_values = np.abs((true_missing - predictions) / (np.abs(true_missing) + 1e-8))
            mape = np.mean(mape_values) * 100

            # 计算R²时处理可能的异常情况
            try:
                r2 = r2_score(true_missing, predictions)
            except:
                r2 = np.nan

            print(f"{method_name:>15} {rmse:>12.4f} {mae:>12.4f} {mape:>12.2f} {r2:>12.4f}")

        except Exception as e:
            print(f"{method_name:>15} {'Error: ' + str(e):>48}")

    return methods


def plot_training_progress(train_losses: List[float], val_losses: List[float]):
    """绘制训练过程"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', alpha=0.8)
    plt.plot(val_losses, label='验证损失', alpha=0.8)
    plt.title('谱感知模型训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='训练损失', alpha=0.8)
    plt.title('训练损失详细')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """主函数：谱感知SpectraFormer缺失值填补流程"""
    print("="*80)
    print("谱感知SpectraFormer缺失值填补系统测试")
    print("="*80)

    # ===== 配置参数 =====
    # 数据配置
    USE_SYNTHETIC_DATA = False  # 设为False使用CSV文件
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing_40.csv")
    TRUE_DATA_PATH = str(DATA_DIR / "train_swat.csv")

    # 合成数据参数
    N_SAMPLES = 1000
    N_FEATURES = 5
    MISSING_RATIO = 0.25

    # MSSA参数
    MSSA_WINDOW_LENGTH = 40
    MSSA_N_COMPONENTS = 20

    # Transformer配置
    TRANSFORMER_CONFIG = {
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'prediction_length': 1
    }

    # MLP配置
    MLP_CONFIG = {
        'hidden_dims': [512, 256, 128],
        'dropout': 0.1
    }

    # 谱偏置配置
    SPECTRAL_CONFIG = {
        'max_seq_len': 512,
        'top_k_freq': 10,
        'learnable_weights': True,
        'temperature': 1.0
    }

    # 训练配置
    FUSION_METHOD = 'concat'  # 'concat', 'add', 'gate'
    USE_SPECTRAL_BIAS = True

    # ===== 1. 数据准备 =====
    if USE_SYNTHETIC_DATA:
        print("\n创建合成数据...")
        true_data, data_with_missing, missing_mask = create_synthetic_data(
            N_SAMPLES, N_FEATURES, MISSING_RATIO
        )

        print(f"合成数据形状: {true_data.shape}")
        print(f"缺失值比例: {missing_mask.mean():.1%}")
        print(f"各特征缺失值数量: {missing_mask.sum(axis=0)}")

        # 转换为DataFrame以便后续处理
        columns = [f'Feature_{i}' for i in range(N_FEATURES)]
        data_with_missing_df = pd.DataFrame(data_with_missing, columns=columns)
        true_data_df = pd.DataFrame(true_data, columns=columns)

    else:
        print("\n加载真实数据...")
        data_with_missing_df, true_data_df, missing_mask = load_data_with_missing(
            DATA_WITH_MISSING_PATH, TRUE_DATA_PATH, index_col=0
        )
        true_data = true_data_df.values
        data_with_missing = data_with_missing_df.values

    # ===== 2. 数据集划分 =====
    split_idx = int(len(data_with_missing) * 0.8)
    train_data = data_with_missing[:split_idx]
    val_data = data_with_missing[split_idx:]

    print(f"\n数据集划分:")
    print(f"训练集: {train_data.shape}")
    print(f"验证集: {val_data.shape}")
    print(f"训练集缺失率: {np.isnan(train_data).mean():.1%}")
    print(f"验证集缺失率: {np.isnan(val_data).mean():.1%}")

    # ===== 3. MSSA预处理 =====
    print("\n进行MSSA预处理...")

    # 创建MSSA模型
    mssa_model = MSSA(
        window_length=MSSA_WINDOW_LENGTH,
        n_components=MSSA_N_COMPONENTS,
        verbose=True
    )

    # 准备MSSA训练数据（需要填补缺失值）
    train_data_for_mssa = pd.DataFrame(train_data).interpolate(
        method='linear', limit_direction='both'
    ).fillna(method='bfill').fillna(method='ffill').values

    # 拟合MSSA
    mssa_model.fit(train_data_for_mssa)
    print(f"MSSA模型拟合完成，分量数: {MSSA_N_COMPONENTS}")

    # 显示奇异值谱
    print("\n可视化MSSA奇异值谱...")
    mssa_model.plot_singular_spectrum(figsize=(12, 5))

    # ===== 4. 创建谱感知填补器 =====
    print("\n创建谱感知SpectraFormer填补器...")

    imputer = SpectralMSSATransformerImputer(
        mssa_window_length=MSSA_WINDOW_LENGTH,
        mssa_n_components=MSSA_N_COMPONENTS,
        transformer_config=TRANSFORMER_CONFIG,
        mlp_config=MLP_CONFIG,
        spectral_config=SPECTRAL_CONFIG,
        fusion_method=FUSION_METHOD,
        use_spectral_bias=USE_SPECTRAL_BIAS
    )

    # ===== 5. 训练模型 =====
    print("\n开始训练谱感知SpectraFormer模型...")
    print(f"使用融合方法: {FUSION_METHOD}")
    print(f"使用谱偏置: {USE_SPECTRAL_BIAS}")

    train_losses, val_losses = imputer.train(
        train_data,
        val_data,
        mssa_model=mssa_model,
        epochs=80,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=50,
        patience=15,
        verbose=True
    )

    # ===== 6. 可视化训练过程 =====
    if train_losses and val_losses:
        plot_training_progress(train_losses, val_losses)

    # ===== 7. 执行填补 =====
    print("\n执行谱感知缺失值填补...")
    imputed_data = imputer.impute_robust(
        data_with_missing,
        mssa_model=mssa_model,
        iterations=3,
        min_sequence_length=10,
        fallback_method='interpolate'
    )

    # ===== 8. 评估填补效果 =====
    if true_data is not None:
        print("\n评估谱感知填补效果...")

        # 检查填补后的数据是否还有NaN
        if np.any(np.isnan(imputed_data)):
            print(f"警告: 填补后仍有 {np.sum(np.isnan(imputed_data))} 个NaN值")
            print("使用线性插值填充剩余的NaN值...")

            # 对剩余NaN进行最终填充
            imputed_df = pd.DataFrame(imputed_data)
            imputed_data = imputed_df.interpolate(
                method='linear', limit_direction='both'
            ).fillna(method='bfill').fillna(method='ffill').values

            # 如果还有NaN，用均值填充
            if np.any(np.isnan(imputed_data)):
                for i in range(imputed_data.shape[1]):
                    col_mean = np.nanmean(imputed_data[:, i])
                    if not np.isnan(col_mean):
                        imputed_data[np.isnan(imputed_data[:, i]), i] = col_mean
                    else:
                        imputed_data[np.isnan(imputed_data[:, i]), i] = 0.0

        try:
            metrics, feature_metrics = imputer.evaluate_imputation(
                true_data,
                imputed_data,
                missing_mask
            )

            print("\n整体填补性能指标:")
            print("-" * 50)
            for metric, value in metrics.items():
                print(f"{metric:>10}: {value:>12.4f}")

            print("\n各特征填补性能:")
            print("-" * 70)
            print(f"{'特征':>15} {'缺失数':>10} {'RMSE':>12} {'MAE':>12}")
            print("-" * 70)

            for feat_name, feat_metrics in feature_metrics.items():
                print(f"{feat_name:>15} {feat_metrics['Missing_Count']:>10} "
                      f"{feat_metrics['RMSE']:>12.4f} {feat_metrics['MAE']:>12.4f}")

        except Exception as e:
            print(f"评估过程出错: {e}")
            print("跳过详细评估，继续后续步骤...")
            metrics = None
            feature_metrics = None

    # ===== 9. 方法对比 =====
    if true_data is not None and metrics is not None:
        comparison_results = compare_imputation_methods(
            true_data, data_with_missing, missing_mask, imputed_data
        )

    # ===== 10. 频谱分析 =====
    print("\n进行频谱分析...")
    visualize_spectral_analysis(
        true_data if true_data is not None else imputed_data,
        feature_indices=[0, 1] if N_FEATURES >= 2 else [0],
        time_range=(0, min(400, len(data_with_missing)))
    )

    # ===== 11. 可视化填补结果 =====
    print("\n可视化填补结果...")
    visualize_results(
        data_with_missing_df if USE_SYNTHETIC_DATA else data_with_missing_df,
        imputed_data,
        true_data_df if USE_SYNTHETIC_DATA else true_data_df,
        missing_mask,
        time_range=(100, min(400, len(data_with_missing))),
        features_to_plot=None
    )

    # ===== 12. 模型架构分析 =====
    print("\n模型架构分析:")
    model_info = imputer.get_model_info()

    print("-" * 60)
    for key, value in model_info.items():
        if key not in ["spectral_weights"]:
            print(f"{key:>25}: {value}")

    if "spectral_weights" in model_info and model_info["spectral_weights"]:
        print(f"\n谱偏置权重信息:")
        print(f"{'层数':>10}: {len(model_info['spectral_weights'])}")
        for i, weights in enumerate(model_info['spectral_weights']):
            print(f"{'层 ' + str(i+1):>10}: {weights[:5] if len(weights) >= 5 else weights}")

    # ===== 13. 保存结果 =====
    print("\n保存填补结果...")

    # 确保填补数据没有NaN再保存
    final_imputed_data = imputed_data.copy()
    if np.any(np.isnan(final_imputed_data)):
        print("最终清理NaN值...")
        final_df = pd.DataFrame(final_imputed_data)
        final_imputed_data = final_df.interpolate(
            method='linear', limit_direction='both'
        ).fillna(method='bfill').fillna(method='ffill').values

        if np.any(np.isnan(final_imputed_data)):
            for i in range(final_imputed_data.shape[1]):
                col_mean = np.nanmean(final_imputed_data[:, i])
                if not np.isnan(col_mean):
                    final_imputed_data[np.isnan(final_imputed_data[:, i]), i] = col_mean
                else:
                    final_imputed_data[np.isnan(final_imputed_data[:, i]), i] = 0.0

    # 保存填补后的数据
    if USE_SYNTHETIC_DATA:
        result_df = pd.DataFrame(
            final_imputed_data,
            columns=[f'Feature_{i}' for i in range(N_FEATURES)]
        )
    else:
        result_df = pd.DataFrame(
            final_imputed_data,
            index=data_with_missing_df.index,
            columns=data_with_missing_df.columns
        )

    result_df.to_csv(imputation_output_path("spectral_spectraformer_imputed.csv"))
    print("填补结果已保存到: spectral_spectraformer_imputed.csv")

    # 保存评估报告
    if true_data is not None and metrics is not None:
        report = {
            'overall_metrics': metrics,
            'feature_metrics': feature_metrics,
            'model_config': {
                'transformer_config': TRANSFORMER_CONFIG,
                'mlp_config': MLP_CONFIG,
                'spectral_config': SPECTRAL_CONFIG,
                'fusion_method': FUSION_METHOD,
                'use_spectral_bias': USE_SPECTRAL_BIAS
            }
        }

        import json
        with open(
            imputation_output_path("spectral_imputation_report.json"),
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print("评估报告已保存到: spectral_imputation_report.json")

    print("\n" + "="*80)
    print("谱感知SpectraFormer缺失值填补测试完成！")
    print("="*80)

    return final_imputed_data, metrics if true_data is not None else None, model_info


def demo_spectral_features():
    """演示谱感知特性的效果"""
    print("\n" + "="*60)
    print("谱感知特性演示")
    print("="*60)

    # 创建具有明显频率特征的信号
    t = np.arange(500)

    # 信号1：低频主导
    signal1 = 2 * np.sin(2 * np.pi * t / 100) + 0.5 * np.sin(2 * np.pi * t / 20) + 0.2 * np.random.randn(len(t))

    # 信号2：高频主导
    signal2 = 0.5 * np.sin(2 * np.pi * t / 100) + 2 * np.sin(2 * np.pi * t / 10) + 0.2 * np.random.randn(len(t))

    signals = np.column_stack([signal1, signal2])

    # 创建缺失值
    missing_ratio = 0.3
    missing_mask = np.random.random(signals.shape) < missing_ratio
    signals_with_missing = signals.copy()
    signals_with_missing[missing_mask] = np.nan

    print(f"演示数据形状: {signals.shape}")
    print(f"缺失值比例: {missing_mask.mean():.1%}")

    # 分析频谱特征
    visualize_spectral_analysis(signals, feature_indices=[0, 1], time_range=(0, 400))

    # 使用谱感知模型填补
    print("\n使用谱感知模型填补演示数据...")

    # 简单的MSSA模型
    demo_mssa = MSSA(window_length=30, n_components=15, verbose=False)
    demo_mssa.fit(pd.DataFrame(signals_with_missing).interpolate().fillna(method='bfill').values)

    # 谱感知填补器
    demo_imputer = SpectralMSSATransformerImputer(
        mssa_window_length=30,
        transformer_config={'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2},
        spectral_config={'top_k_freq': 5, 'temperature': 0.8},
        use_spectral_bias=True
    )

    # 训练
    train_size = int(0.8 * len(signals_with_missing))
    demo_imputer.train(
        signals_with_missing[:train_size],
        signals_with_missing[train_size:],
        mssa_model=demo_mssa,
        epochs=30,
        verbose=False
    )

    # 填补
    demo_imputed = demo_imputer.impute_robust(signals_with_missing, demo_mssa, iterations=2)

    # 可视化结果
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    for i, (signal_name, ax) in enumerate(zip(['低频主导信号', '高频主导信号'], axes)):
        # 原始信号
        ax.plot(t[:400], signals[:400, i], 'b-', label='真实信号', alpha=0.7, linewidth=1.5)

        # 缺失位置
        missing_points = missing_mask[:400, i]
        if np.any(missing_points):
            ax.scatter(t[:400][missing_points], demo_imputed[:400, i][missing_points],
                      c='red', s=15, label='谱感知填补', zorder=5, alpha=0.8)

        ax.set_title(f'{signal_name} - 谱感知填补演示')
        ax.set_ylabel('幅值')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()

    # 计算演示效果
    demo_rmse = np.sqrt(np.mean((signals[missing_mask] - demo_imputed[missing_mask]) ** 2))
    print(f"\n演示填补RMSE: {demo_rmse:.4f}")

    return signals, demo_imputed


if __name__ == "__main__":
    # 运行主测试流程
    print("开始谱感知SpectraFormer缺失值填补测试...")

    try:
        # 主要测试流程
        imputed_data, metrics, model_info = main()

        # 特性演示
        print("\n" + "="*80)
        demo_signals, demo_imputed = demo_spectral_features()

        print("\n所有测试完成！")

        # 总结报告
        print("\n" + "="*80)
        print("测试总结报告")
        print("="*80)

        if metrics:
            print(f"主要性能指标:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  R²:   {metrics['R2']:.4f}")

        if model_info:
            print(f"\n模型信息:")
            print(f"  总参数量: {model_info.get('total_parameters', 'N/A'):,}")
            print(f"  使用谱偏置: {model_info.get('use_spectral_bias', 'N/A')}")
            print(f"  融合方法: {model_info.get('fusion_method', 'N/A')}")

        print("\n关键优势:")
        print("✓ 谱感知注意力机制能够捕捉时间序列的频率特征")
        print("✓ MSSA+MLP特征处理提供丰富的时间模式信息")
        print("✓ 多种特征融合策略适应不同数据特点")
        print("✓ 稳健的填补策略处理各种缺失模式")
        print("✓ 端到端训练优化整体填补性能")

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

        print("\n故障排除建议:")
        print("1. 检查数据文件路径是否正确")
        print("2. 确认所需的Python包已安装")
        print("3. 检查GPU内存是否充足")
        print("4. 尝试减小batch_size或模型参数")
