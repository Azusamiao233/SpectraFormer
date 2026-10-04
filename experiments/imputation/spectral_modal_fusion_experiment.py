"""
谱感知+模态融合Transformer缺失值填补完整示例 - 使用现成数据版本
演示如何使用MSSA分解和谱感知Transformer模型进行时间序列缺失值填补
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import torch

warnings.filterwarnings('ignore')

# 导入必要的模块
from spectraformer.mssa.mssa1 import MSSA, setup_chinese_fonts
from spectraformer.transformer.transformer_imputation_mssa_mlp_re_bias import (
    SpectralMSSATransformerImputer,
)
from spectraformer.config import DATA_DIR, imputation_output_path

# 设置中文字体
setup_chinese_fonts()

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


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


def train_spectral_modal_transformer(train_data, val_data, mssa_model=None, config=None):
    """
    训练谱感知+模态融合Transformer填补模型

    参数:
        train_data: 训练数据（DataFrame）
        val_data: 验证数据（DataFrame）
        mssa_model: MSSA模型（可选）
        config: 模型配置
    """
    print("\n训练谱感知+模态融合Transformer填补模型...")

    # 默认配置
    if config is None:
        config = {
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'prediction_length': 1
            },
            'mlp_config': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.1
            },
            'spectral_config': {
                'max_seq_len': 512,
                'top_k_freq': 10,
                'learnable_weights': True,
                'temperature': 1.0
            },
            'fusion_method': 'concat',
            'use_modal_fusion': True,
            'modal_fusion_type': 'adaptive',
            'use_spectral_bias': True
        }

    # 创建谱感知+模态融合填补器
    imputer = SpectralMSSATransformerImputer(
        mssa_window_length=mssa_model.window_length if mssa_model else 48,
        mssa_n_components=mssa_model.n_components if mssa_model else None,
        transformer_config=config['transformer_config'],
        mlp_config=config['mlp_config'],
        spectral_config=config['spectral_config'],
        fusion_method=config['fusion_method'],
        use_modal_fusion=config['use_modal_fusion'],
        modal_fusion_type=config['modal_fusion_type'],
        use_spectral_bias=config['use_spectral_bias']
    )

    # 训练
    print("开始训练...")
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
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('谱感知+模态融合Transformer训练过程中的损失变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return imputer


def impute_and_evaluate(imputer, data_with_missing, true_data, missing_mask,
                       mssa_model=None, save_results=True):
    """
    使用训练好的谱感知+模态融合模型进行填补并评估

    参数:
        imputer: 训练好的填补器
        data_with_missing: 包含缺失值的数据
        true_data: 真实完整数据（用于评估）
        missing_mask: 缺失值掩码
        mssa_model: MSSA模型（可选）
        save_results: 是否保存结果
    """
    print("\n使用训练好的谱感知+模态融合模型填补缺失值...")

    # 使用稳健填补方法
    imputed_data = imputer.impute_robust(
        data_with_missing.values,
        mssa_model=mssa_model,
        iterations=3,  # 迭代3次
        min_sequence_length=10,
        fallback_method='interpolate'
    )

    # 如果有真实数据，进行评估
    if true_data is not None:
        print("\n评估谱感知+模态融合模型填补效果...")
        metrics, feature_metrics = imputer.evaluate_imputation(
            true_data.values,
            imputed_data,
            missing_mask
        )

        print("\n谱感知+模态融合模型整体填补性能指标:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric:>10}: {value:>10.4f}")

        print("\n各特征填补性能:")
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
        print("\n保存填补后的数据...")
        imputed_df = pd.DataFrame(
            imputed_data,
            index=data_with_missing.index,
            columns=data_with_missing.columns
        )
        imputed_df.to_csv(imputation_output_path("spectral_modal_imputed_data.csv"))
        print("已保存到: spectral_modal_imputed_data.csv")

        # 保存评估报告
        if metrics:
            with open(
                imputation_output_path("spectral_modal_imputation_report.txt"),
                'w',
                encoding='utf-8',
            ) as f:
                f.write("谱感知+模态融合Transformer缺失值填补报告\n")
                f.write("="*60 + "\n\n")
                f.write(f"数据集大小: {data_with_missing.shape}\n")
                f.write(f"缺失值总数: {missing_mask.sum()}\n")
                f.write(f"缺失比例: {missing_mask.sum() / missing_mask.size * 100:.2f}%\n\n")

                f.write("模型特性:\n")
                model_info = imputer.get_model_info()
                for key, value in model_info.items():
                    if key not in ["spectral_weights", "modal_fusion_params"]:
                        f.write(f"{key}: {value}\n")

                f.write("\n整体填补性能:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")

                if "spectral_weights" in model_info:
                    f.write(f"\n谱偏置权重层数: {len(model_info['spectral_weights'])}\n")

                if "modal_fusion_params" in model_info:
                    f.write(f"\n模态融合参数:\n")
                    for key, value in model_info["modal_fusion_params"].items():
                        f.write(f"  {key}: {value:,}\n")

            print("评估报告已保存到: spectral_modal_imputation_report.txt")

    return imputed_data, metrics


def visualize_comprehensive_results(imputer, data_with_missing, imputed_data, true_data, missing_mask,
                                   time_range=(0, 500), features_to_plot=None):
    """
    使用模型的可视化函数进行综合结果展示
    """
    print("\n生成综合填补结果可视化...")

    # 1. 基础填补结果可视化
    if features_to_plot is None:
        features_to_plot = list(range(min(4, data_with_missing.shape[1])))

    imputer.plot_imputation_results(
        true_data.values if true_data is not None else imputed_data,
        imputed_data,
        missing_mask,
        feature_indices=features_to_plot,
        time_range=time_range
    )

    # 2. 谱分析可视化
    print("\n生成谱分析可视化...")
    for feat_idx in features_to_plot[:2]:  # 只显示前两个特征的谱分析
        freqs, power_spectrum = imputer.plot_spectral_analysis(
            imputed_data,
            feature_idx=feat_idx,
            time_range=time_range
        )

    # 3. 模态注意力权重可视化
    print("\n生成模态注意力权重可视化...")
    imputer.plot_modal_attention(n_samples=3)

    # 4. 谱注意力权重可视化
    print("\n生成谱注意力权重可视化...")
    imputer.plot_spectral_attention(layer_idx=0, head_idx=0, n_samples=3)

    # 5. 传统风格的填补结果对比图
    print("\n生成传统风格填补结果对比图...")
    n_features = len(features_to_plot)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 3*n_features))
    if n_features == 1:
        axes = [axes]

    start, end = time_range
    time_index = data_with_missing.index[start:end]

    for idx, (feat_idx, ax) in enumerate(zip(features_to_plot, axes)):
        feat_name = data_with_missing.columns[feat_idx]

        # 带缺失的数据
        ax.plot(time_index, data_with_missing[feat_name].iloc[start:end],
               'g--', label='带缺失值', alpha=0.5)

        # 真实数据（如果有）
        if true_data is not None:
            ax.plot(time_index, true_data[feat_name].iloc[start:end],
                   'b-', label='真实值', alpha=0.7, linewidth=1.5)

        # 填补的点
        mask_slice = missing_mask[start:end, feat_idx]
        if np.any(mask_slice):
            missing_times = time_index[mask_slice]
            imputed_values = imputed_data[start:end, feat_idx][mask_slice]
            ax.scatter(missing_times, imputed_values,
                      c='red', s=25, label='谱感知+模态融合填补', zorder=5, alpha=0.8)

        ax.set_title(f'{feat_name} - 谱感知+模态融合填补结果')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def print_model_architecture_info(imputer):
    """
    打印模型架构详细信息
    """
    print("\n" + "="*60)
    print("谱感知+模态融合Transformer模型架构信息")
    print("="*60)

    model_info = imputer.get_model_info()

    print(f"模型类型: {model_info.get('model_type', 'Unknown')}")
    print(f"使用谱偏置: {model_info.get('use_spectral_bias', False)}")
    print(f"使用模态融合: {model_info.get('use_modal_fusion', False)}")
    print(f"模态融合类型: {model_info.get('modal_fusion_type', 'None')}")
    print(f"特征融合方法: {model_info.get('fusion_method', 'None')}")
    print(f"使用MSSA: {model_info.get('use_mssa', False)}")
    print(f"总参数数量: {model_info.get('total_parameters', 0):,}")
    print(f"可训练参数数量: {model_info.get('trainable_parameters', 0):,}")

    # 谱配置信息
    spectral_config = model_info.get('spectral_config', {})
    if spectral_config:
        print(f"\n谱偏置配置:")
        print(f"  最大序列长度: {spectral_config.get('max_seq_len', 'Unknown')}")
        print(f"  主导频率数: {spectral_config.get('top_k_freq', 'Unknown')}")
        print(f"  可学习权重: {spectral_config.get('learnable_weights', 'Unknown')}")
        print(f"  温度参数: {spectral_config.get('temperature', 'Unknown')}")

    # Transformer配置信息
    transformer_config = model_info.get('transformer_config', {})
    if transformer_config:
        print(f"\nTransformer配置:")
        print(f"  模型维度: {transformer_config.get('d_model', 'Unknown')}")
        print(f"  注意力头数: {transformer_config.get('nhead', 'Unknown')}")
        print(f"  编码器层数: {transformer_config.get('num_encoder_layers', 'Unknown')}")
        print(f"  前馈维度: {transformer_config.get('dim_feedforward', 'Unknown')}")

    # 谱偏置权重信息
    if "spectral_weights" in model_info:
        print(f"\n谱偏置权重信息:")
        print(f"  权重层数: {len(model_info['spectral_weights'])}")
        for i, weights in enumerate(model_info['spectral_weights']):
            print(f"  层 {i+1} 权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"  层 {i+1} 权重前5个: {weights[:5]}")

    # 模态融合参数信息
    if "modal_fusion_params" in model_info:
        print(f"\n模态融合参数信息:")
        for key, value in model_info["modal_fusion_params"].items():
            print(f"  {key}: {value:,}")


def main():
    """
    主函数：使用现成数据的谱感知+模态融合Transformer缺失值填补流程
    """
    print("="*70)
    print("谱感知+模态融合Transformer缺失值填补 - 使用现成数据")
    print("="*70)

    # 配置参数
    # 修改这些路径为您的实际数据文件路径
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing_40.csv")
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

    # 4. 训练谱感知+模态融合Transformer
    imputer = train_spectral_modal_transformer(
        train_data,
        val_data,
        mssa_model=mssa_model,
        config={
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'prediction_length': 1
            },
            'mlp_config': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.1
            },
            'spectral_config': {
                'max_seq_len': 512,
                'top_k_freq': 10,
                'learnable_weights': True,
                'temperature': 1.0
            },
            'fusion_method': 'concat',          # 可选: 'concat', 'add', 'gate'
            'use_modal_fusion': True,          # 启用模态融合
            'modal_fusion_type': 'adaptive',   # 使用自适应模态融合
            'use_spectral_bias': True          # 启用谱偏置
        }
    )

    # 5. 填补和评估
    imputed_data, metrics = impute_and_evaluate(
        imputer,
        data_with_missing,
        true_data,
        missing_mask,
        mssa_model=mssa_model,
        save_results=True
    )

    # 6. 打印模型架构信息
    print_model_architecture_info(imputer)

    # 7. 综合可视化
    visualize_comprehensive_results(
        imputer,
        data_with_missing,
        imputed_data,
        true_data,
        missing_mask,
        time_range=(200, 400),  # 可以调整显示的时间范围
        features_to_plot=None   # None表示显示前4个特征
    )

    print("\n处理完成！")
    print("\n模型特性总结:")
    print(f"✓ 谱感知注意力偏置: {imputer.use_spectral_bias}")
    print(f"✓ 自适应模态融合: {imputer.use_modal_fusion}")
    print(f"✓ 融合策略: {imputer.fusion_method}")
    print(f"✓ 模态融合类型: {imputer.modal_fusion_type}")
    print(f"✓ 使用MSSA: {imputer.model.use_mssa}")

    return imputed_data, metrics


# 批量实验函数：比较不同谱感知和模态融合配置的效果
def run_spectral_modal_experiments(data_path, true_data_path=None):
    """
    运行多组实验，比较不同谱感知和模态融合配置的效果
    """
    print("="*70)
    print("批量实验：比较不同谱感知+模态融合配置")
    print("="*70)

    # 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        data_path, true_data_path, index_col=0
    )

    # 实验配置
    experiments = [
        {
            'name': 'Baseline (No MSSA, No Spectral)',
            'use_mssa': False,
            'use_spectral_bias': False,
            'use_modal_fusion': False,
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512
            }
        },
        {
            'name': 'With MSSA + Modal Fusion',
            'use_mssa': True,
            'use_spectral_bias': False,
            'use_modal_fusion': True,
            'modal_fusion_type': 'adaptive',
            'fusion_method': 'concat',
            'mssa_window': 48,
            'mssa_components': 20,
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512
            }
        },
        {
            'name': 'With MSSA + Spectral Bias',
            'use_mssa': True,
            'use_spectral_bias': True,
            'use_modal_fusion': False,
            'mssa_window': 48,
            'mssa_components': 20,
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512
            },
            'spectral_config': {
                'max_seq_len': 512,
                'top_k_freq': 10,
                'learnable_weights': True,
                'temperature': 1.0
            }
        },
        {
            'name': 'Full Model (MSSA + Spectral + Modal)',
            'use_mssa': True,
            'use_spectral_bias': True,
            'use_modal_fusion': True,
            'modal_fusion_type': 'adaptive',
            'fusion_method': 'concat',
            'mssa_window': 48,
            'mssa_components': 20,
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512
            },
            'spectral_config': {
                'max_seq_len': 512,
                'top_k_freq': 10,
                'learnable_weights': True,
                'temperature': 1.0
            }
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n运行实验: {exp['name']}")
        print("-" * 50)

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

        # 训练
        config = {
            'transformer_config': exp.get('transformer_config', {}),
            'mlp_config': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.1
            },
            'spectral_config': exp.get('spectral_config', {}),
            'fusion_method': exp.get('fusion_method', 'concat'),
            'use_modal_fusion': exp.get('use_modal_fusion', False),
            'modal_fusion_type': exp.get('modal_fusion_type', 'adaptive'),
            'use_spectral_bias': exp.get('use_spectral_bias', False)
        }

        imputer = train_spectral_modal_transformer(
            train_data,
            val_data,
            mssa_model=mssa_model,
            config=config
        )

        # 评估
        imputed_data, metrics = impute_and_evaluate(
            imputer,
            data_with_missing,
            true_data,
            missing_mask,
            mssa_model=mssa_model,
            save_results=False
        )

        # 记录结果
        if metrics:
            results.append({
                'experiment': exp['name'],
                'use_mssa': exp.get('use_mssa', False),
                'use_spectral_bias': exp.get('use_spectral_bias', False),
                'use_modal_fusion': exp.get('use_modal_fusion', False),
                **metrics
            })

    # 汇总结果
    results_df = pd.DataFrame(results)
    print("\n实验结果汇总:")
    print(results_df.to_string(index=False))

    # 保存结果
    results_df.to_csv(
        imputation_output_path("spectral_modal_experiment_results_wadi_20.csv"),
        index=False,
    )
    print("\n结果已保存到: spectral_modal_experiment_results_wadi_20.csv")

    # 绘制对比图
    if len(results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2']

        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                ax = axes[i]
                values = results_df[metric].values
                names = results_df['experiment'].values

                bars = ax.bar(range(len(values)), values, alpha=0.7)
                ax.set_xlabel('实验配置')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} 对比')
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')

                # 添加数值标签
                for j, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return results_df


def compare_fusion_methods(data_path, true_data_path=None):
    """
    比较不同融合方法的效果
    """
    print("="*60)
    print("比较不同融合方法效果")
    print("="*60)

    # 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        data_path, true_data_path, index_col=0
    )

    # 准备数据
    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 准备MSSA
    data_interpolated = train_data.interpolate(method='linear', limit_direction='both')
    mssa_model = perform_mssa_analysis(data_interpolated.values, 48, 20)

    # 不同融合方法
    fusion_methods = ['concat', 'add', 'gate']
    results = []

    for method in fusion_methods:
        print(f"\n测试融合方法: {method}")
        print("-" * 30)

        config = {
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'prediction_length': 1
            },
            'mlp_config': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.1
            },
            'spectral_config': {
                'max_seq_len': 512,
                'top_k_freq': 10,
                'learnable_weights': True,
                'temperature': 1.0
            },
            'fusion_method': method,
            'use_modal_fusion': True,
            'modal_fusion_type': 'adaptive',
            'use_spectral_bias': True
        }

        imputer = train_spectral_modal_transformer(
            train_data, val_data, mssa_model=mssa_model, config=config
        )

        imputed_data, metrics = impute_and_evaluate(
            imputer, data_with_missing, true_data, missing_mask,
            mssa_model=mssa_model, save_results=False
        )

        if metrics:
            results.append({
                'fusion_method': method,
                **metrics
            })

    # 显示结果
    results_df = pd.DataFrame(results)
    print("\n融合方法对比结果:")
    print(results_df.to_string(index=False))

    return results_df


def analyze_spectral_parameters(data_path, true_data_path=None):
    """
    分析不同谱参数配置的效果
    """
    print("="*60)
    print("分析谱参数配置效果")
    print("="*60)

    # 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        data_path, true_data_path, index_col=0
    )

    # 准备数据
    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 准备MSSA
    data_interpolated = train_data.interpolate(method='linear', limit_direction='both')
    mssa_model = perform_mssa_analysis(data_interpolated.values, 48, 20)

    # 不同谱参数配置
    spectral_configs = [
        {'top_k_freq': 5, 'temperature': 0.5},
        {'top_k_freq': 10, 'temperature': 1.0},
        {'top_k_freq': 15, 'temperature': 1.5},
        {'top_k_freq': 20, 'temperature': 2.0}
    ]

    results = []

    for i, spectral_config in enumerate(spectral_configs):
        print(f"\n测试谱配置 {i+1}: {spectral_config}")
        print("-" * 40)

        config = {
            'transformer_config': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'prediction_length': 1
            },
            'mlp_config': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.1
            },
            'spectral_config': {
                'max_seq_len': 512,
                'learnable_weights': True,
                **spectral_config
            },
            'fusion_method': 'concat',
            'use_modal_fusion': True,
            'modal_fusion_type': 'adaptive',
            'use_spectral_bias': True
        }

        imputer = train_spectral_modal_transformer(
            train_data, val_data, mssa_model=mssa_model, config=config
        )

        imputed_data, metrics = impute_and_evaluate(
            imputer, data_with_missing, true_data, missing_mask,
            mssa_model=mssa_model, save_results=False
        )

        if metrics:
            results.append({
                'config_id': f"Config_{i+1}",
                'top_k_freq': spectral_config['top_k_freq'],
                'temperature': spectral_config['temperature'],
                **metrics
            })

    # 显示结果
    results_df = pd.DataFrame(results)
    print("\n谱参数配置对比结果:")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    # 运行主流程
    print("开始运行谱感知+模态融合Transformer缺失值填补测试...")
    imputed_data, metrics = main()

    # 如果想运行批量实验，取消下面的注释
    # print("\n\n开始运行批量实验...")
    # results = run_spectral_modal_experiments(
    #     "../data/train_continuous_missing.csv",
    #     "../data/test.csv"
    # )

    # 如果想比较不同融合方法，取消下面的注释
    # print("\n\n开始比较融合方法...")
    # fusion_results = compare_fusion_methods(
    #     "../data/train_continuous_missing.csv",
    #     "../data/test.csv"
    # )

    # 如果想分析谱参数，取消下面的注释
    # print("\n\n开始分析谱参数...")
    # spectral_results = analyze_spectral_parameters(
    #     "../data/train_continuous_missing.csv",
    #     "../data/test.csv"
    # )

    print("\n" + "="*70)
    print("谱感知+模态融合Transformer缺失值填补测试完成!")
    print("="*70)
