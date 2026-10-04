"""
MSSA-Transformer可学习模态融合缺失值填补完整示例 - 使用现成数据版本
演示如何使用MSSA分解和Transformer模型结合可学习模态融合进行时间序列缺失值填补
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# 导入必要的模块（假设已经有mssa.py和transformer_imputation_mssa_mlp_re.py）
from mssa_transformer.mssa import MSSA, setup_chinese_fonts
from mssa_transformer.transformer.transformer_imputation_mssa_mlp_re import (
    MSSATransformerImputerMLPFixed,
)
from mssa_transformer.config import DATA_DIR, imputation_output_path

# 设置中文字体
setup_chinese_fonts()

# 设置随机种子
np.random.seed(42)


def calculate_safe_metrics(true_data, imputed_data, missing_mask, column_names):
    """
    安全的指标计算函数，处理NaN值
    """
    try:
        # 确保没有NaN值
        true_missing = true_data[missing_mask]
        imputed_missing = imputed_data[missing_mask]

        # 移除任何仍然是NaN的值
        valid_mask = ~(np.isnan(true_missing) | np.isnan(imputed_missing))

        if np.sum(valid_mask) == 0:
            print("警告: 没有有效的数据点用于评估")
            return {
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'MAPE': float('inf'),
                'R2': -float('inf')
            }, {}

        true_valid = true_missing[valid_mask]
        imputed_valid = imputed_missing[valid_mask]

        # 计算指标
        rmse = np.sqrt(mean_squared_error(true_valid, imputed_valid))
        mae = mean_absolute_error(true_valid, imputed_valid)
        mape = np.mean(np.abs((true_valid - imputed_valid) / (np.abs(true_valid) + 1e-8))) * 100
        r2 = r2_score(true_valid, imputed_valid)

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }

        # 计算每个特征的指标
        feature_metrics = {}
        for i in range(true_data.shape[1]):
            feature_mask = missing_mask[:, i]
            if np.any(feature_mask):
                true_feat = true_data[feature_mask, i]
                imputed_feat = imputed_data[feature_mask, i]

                # 移除NaN
                feat_valid_mask = ~(np.isnan(true_feat) | np.isnan(imputed_feat))
                if np.sum(feat_valid_mask) > 0:
                    true_feat_valid = true_feat[feat_valid_mask]
                    imputed_feat_valid = imputed_feat[feat_valid_mask]

                    feature_metrics[f'Feature_{i}'] = {
                        'RMSE': np.sqrt(mean_squared_error(true_feat_valid, imputed_feat_valid)),
                        'MAE': mean_absolute_error(true_feat_valid, imputed_feat_valid),
                        'Missing_Count': np.sum(feature_mask)
                    }

        return metrics, feature_metrics

    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {
            'RMSE': float('inf'),
            'MAE': float('inf'),
            'MAPE': float('inf'),
            'R2': -float('inf')
        }, {}


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


def train_modal_fusion_transformer(train_data, val_data, mssa_model=None,
                                 transformer_config=None, mlp_config=None,
                                 fusion_method='concat', modal_fusion_type='adaptive'):
    """
    训练带模态融合的Transformer填补模型

    参数:
        train_data: 训练数据（DataFrame）
        val_data: 验证数据（DataFrame）
        mssa_model: MSSA模型（可选）
        transformer_config: Transformer配置
        mlp_config: MLP配置
        fusion_method: 特征融合方法
        modal_fusion_type: 模态融合类型
    """
    print(f"\n训练带{modal_fusion_type}模态融合的Transformer填补模型...")

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
            'hidden_dims': [512, 256],
            'dropout': 0.1
        }

    # 创建填补器
    imputer = MSSATransformerImputerMLPFixed(
        mssa_window_length=mssa_model.window_length if mssa_model else 48,
        mssa_n_components=mssa_model.n_components if mssa_model else None,
        transformer_config=transformer_config,
        mlp_config=mlp_config,
        fusion_method=fusion_method,
        use_modal_fusion=True,
        modal_fusion_type=modal_fusion_type
    )

    # 训练
    print("开始训练...")
    try:
        train_losses, val_losses = imputer.train(
            train_data.values,
            val_data.values,
            mssa_model=mssa_model,
            epochs=50,  # 减少训练轮数用于测试
            batch_size=32,
            learning_rate=0.001,
            sequence_length=50,
            patience=10,  # 减少patience用于测试
            verbose=True
        )

        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'训练过程中的损失变化 - {modal_fusion_type.capitalize()} Modal Fusion')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"训练过程出错: {e}")
        print("使用默认配置重新尝试...")
        # 简化配置重新训练
        simplified_config = {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'prediction_length': 1
        }

        imputer = MSSATransformerImputerMLPFixed(
            mssa_window_length=30,
            mssa_n_components=10,
            transformer_config=simplified_config,
            mlp_config={'hidden_dims': [256], 'dropout': 0.1},
            fusion_method=fusion_method,
            use_modal_fusion=True,
            modal_fusion_type=modal_fusion_type
        )

        train_losses, val_losses = imputer.train(
            train_data.values,
            val_data.values,
            mssa_model=mssa_model,
            epochs=20,
            batch_size=16,
            learning_rate=0.001,
            sequence_length=30,
            patience=5,
            verbose=True
        )

    return imputer


def impute_and_evaluate_modal(imputer, data_with_missing, true_data, missing_mask,
                            mssa_model=None, save_results=True, method_name="Modal_Fusion"):
    """
    使用训练好的模型进行填补并评估

    参数:
        imputer: 训练好的填补器
        data_with_missing: 包含缺失值的数据
        true_data: 真实完整数据（用于评估）
        missing_mask: 缺失值掩码
        mssa_model: MSSA模型（可选）
        save_results: 是否保存结果
        method_name: 方法名称标识
    """
    print(f"\n使用训练好的{method_name}模型填补缺失值...")

    # 填补缺失值
    try:
        imputed_data = imputer.impute(
            data_with_missing.values,
            mssa_model=mssa_model,
            iterations=3  # 迭代3次
        )

        # 检查填补结果中是否还有NaN
        if np.any(np.isnan(imputed_data)):
            print(f"警告: {method_name}填补后仍有 {np.sum(np.isnan(imputed_data))} 个NaN值，使用插值处理...")
            # 使用pandas的插值方法处理剩余的NaN
            imputed_df = pd.DataFrame(imputed_data)
            imputed_df = imputed_df.interpolate(method='linear', limit_direction='both')
            # 如果还有NaN，用前向填充和后向填充
            imputed_df = imputed_df.fillna(method='ffill').fillna(method='bfill')
            # 如果还有NaN，用均值填充
            imputed_df = imputed_df.fillna(imputed_df.mean())
            imputed_data = imputed_df.values

        print(f"{method_name}填补完成，检查最终结果...")
        print(f"填补后NaN数量: {np.sum(np.isnan(imputed_data))}")

    except Exception as e:
        print(f"填补过程出错: {e}")
        # 如果填补失败，使用简单的插值作为备选
        print("使用线性插值作为备选方案...")
        imputed_df = pd.DataFrame(data_with_missing.values)
        imputed_df = imputed_df.interpolate(method='linear', limit_direction='both')
        imputed_df = imputed_df.fillna(method='ffill').fillna(method='bfill')
        imputed_df = imputed_df.fillna(imputed_df.mean())
        imputed_data = imputed_df.values

    # 如果有真实数据，进行评估
    if true_data is not None:
        print(f"\n评估{method_name}填补效果...")
        try:
            metrics, feature_metrics = imputer.evaluate_imputation(
                true_data.values,
                imputed_data,
                missing_mask
            )
        except Exception as e:
            print(f"评估过程出错: {e}")
            # 手动计算评估指标
            print("使用备选评估方法...")
            metrics, feature_metrics = calculate_safe_metrics(
                true_data.values, imputed_data, missing_mask, data_with_missing.columns
            )

        print(f"\n{method_name}整体填补性能指标:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric:>10}: {value:>10.4f}")

        print(f"\n{method_name}各特征填补性能:")
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
        print(f"\n注意: 没有提供真实数据，无法评估{method_name}填补效果")

    # 保存结果
    if save_results:
        print(f"\n保存{method_name}填补后的数据...")
        imputed_df = pd.DataFrame(
            imputed_data,
            index=data_with_missing.index,
            columns=data_with_missing.columns
        )
        filename = imputation_output_path(f'imputed_data_{method_name.lower()}.csv')
        imputed_df.to_csv(filename)
        print(f"已保存到: {filename}")

        # 保存评估报告
        if metrics:
            report_filename = imputation_output_path(
                f'imputation_report_{method_name.lower()}.txt'
            )
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(f"MSSA-Transformer {method_name} 缺失值填补报告\n")
                f.write("="*60 + "\n\n")
                f.write(f"数据集大小: {data_with_missing.shape}\n")
                f.write(f"缺失值总数: {missing_mask.sum()}\n")
                f.write(f"缺失比例: {missing_mask.sum() / missing_mask.size * 100:.2f}%\n\n")

                f.write(f"{method_name} 整体填补性能:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            print(f"评估报告已保存到: {report_filename}")

    return imputed_data, metrics


def visualize_modal_fusion_results(data_with_missing, imputed_data_dict, true_data,
                                 missing_mask, time_range=(0, 500),
                                 features_to_plot=None):
    """
    可视化多种模态融合方法的填补结果比较

    参数:
        data_with_missing: 原始带缺失的数据
        imputed_data_dict: 不同方法的填补结果字典
        true_data: 真实数据
        missing_mask: 缺失值掩码
        time_range: 显示时间范围
        features_to_plot: 要显示的特征列表
    """
    print("\n生成模态融合填补结果对比可视化...")

    if features_to_plot is None:
        features_to_plot = data_with_missing.columns[:min(3, len(data_with_missing.columns))]

    n_features = len(features_to_plot)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
    if n_features == 1:
        axes = [axes]

    start, end = time_range
    time_index = data_with_missing.index[start:end]

    colors = ['red', 'orange', 'purple', 'brown']

    for idx, (feat, ax) in enumerate(zip(features_to_plot, axes)):
        feat_idx = data_with_missing.columns.get_loc(feat)

        # 带缺失的数据
        ax.plot(time_index, data_with_missing[feat].iloc[start:end],
               'g--', label='带缺失值', alpha=0.5)

        # 真实数据（如果有）
        if true_data is not None:
            ax.plot(time_index, true_data[feat].iloc[start:end],
                   'b-', label='真实值', alpha=0.7, linewidth=2)

        # 不同方法的填补结果
        for method_idx, (method_name, imputed_data) in enumerate(imputed_data_dict.items()):
            # 填补的点
            mask_slice = missing_mask[start:end, feat_idx]
            if np.any(mask_slice):
                missing_times = time_index[mask_slice]
                imputed_values = imputed_data[start:end, feat_idx][mask_slice]
                color = colors[method_idx % len(colors)]
                ax.scatter(missing_times, imputed_values,
                          c=color, s=25, label=f'{method_name}填补值',
                          zorder=5, alpha=0.8)

        ax.set_title(f'{feat} - 不同模态融合方法填补结果对比')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def visualize_modal_attention_comparison(imputers_dict):
    """
    比较不同方法的模态注意力权重
    """
    print("\n可视化模态注意力权重对比...")

    n_methods = len(imputers_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for idx, (method_name, imputer) in enumerate(imputers_dict.items()):
        ax = axes[idx]

        if hasattr(imputer, 'model') and hasattr(imputer.model, 'last_attention_weights'):
            attention_weights = imputer.model.last_attention_weights
            if attention_weights is not None:
                # 显示第一个样本的注意力权重
                weights = attention_weights[0].cpu().numpy()
                im = ax.imshow(weights.T, aspect='auto', cmap='hot')
                ax.set_xlabel('时间步')
                ax.set_ylabel('模态')
                ax.set_title(f'{method_name}\n模态注意力权重')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'{method_name}\n无注意力权重数据',
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f'{method_name}\n不支持注意力可视化',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def compare_modal_fusion_methods(data_with_missing, true_data, missing_mask,
                               mssa_model, train_data, val_data):
    """
    比较不同模态融合方法的效果
    """
    print("="*70)
    print("模态融合方法对比实验")
    print("="*70)

    # 定义要比较的方法
    methods = {
        'Adaptive_Fusion': {
            'modal_fusion_type': 'adaptive',
            'fusion_method': 'concat'
        },
        'Learnable_Fusion': {
            'modal_fusion_type': 'learnable',
            'fusion_method': 'concat'
        },
        'MLP_Only': {
            'modal_fusion_type': 'mlp',
            'fusion_method': 'concat'
        }
    }

    results = {}
    imputers = {}
    imputed_data_dict = {}

    # 训练和评估每种方法
    for method_name, config in methods.items():
        print(f"\n{'='*20} {method_name} {'='*20}")

        # 训练模型
        imputer = train_modal_fusion_transformer(
            train_data,
            val_data,
            mssa_model=mssa_model,
            modal_fusion_type=config['modal_fusion_type'],
            fusion_method=config['fusion_method']
        )

        # 填补和评估
        imputed_data, metrics = impute_and_evaluate_modal(
            imputer,
            data_with_missing,
            true_data,
            missing_mask,
            mssa_model=mssa_model,
            save_results=True,
            method_name=method_name
        )

        # 保存结果
        results[method_name] = metrics
        imputers[method_name] = imputer
        imputed_data_dict[method_name] = imputed_data

    # 汇总对比结果
    print("\n" + "="*70)
    print("方法对比结果汇总")
    print("="*70)

    if all(result is not None for result in results.values()):
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.round(4))

        # 保存对比结果
        comparison_df.to_csv(imputation_output_path("modal_fusion_comparison.csv"))
        print("\n对比结果已保存到: modal_fusion_comparison.csv")

        # 找出最佳方法
        best_method_rmse = comparison_df['RMSE'].idxmin()
        best_method_mae = comparison_df['MAE'].idxmin()
        best_method_r2 = comparison_df['R2'].idxmax()

        print(f"\n最佳方法:")
        print(f"RMSE最低: {best_method_rmse} ({comparison_df.loc[best_method_rmse, 'RMSE']:.4f})")
        print(f"MAE最低: {best_method_mae} ({comparison_df.loc[best_method_mae, 'MAE']:.4f})")
        print(f"R2最高: {best_method_r2} ({comparison_df.loc[best_method_r2, 'R2']:.4f})")

    # 可视化对比
    visualize_modal_fusion_results(
        data_with_missing,
        imputed_data_dict,
        true_data,
        missing_mask,
        time_range=(200, 400)
    )

    # 可视化注意力权重对比
    visualize_modal_attention_comparison(imputers)

    return results, imputers, imputed_data_dict


def main():
    """
    主函数：MSSA-Transformer模态融合缺失值填补流程
    """
    print("="*70)
    print("MSSA-Transformer可学习模态融合缺失值填补实验")
    print("="*70)

    # 配置参数
    # 修改这些路径为您的实际数据文件路径
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing_60.csv")
    TRUE_DATA_PATH = str(DATA_DIR / "train_swat.csv")

    # MSSA参数
    MSSA_WINDOW_LENGTH = 48
    MSSA_N_COMPONENTS = 20

    # 实验选项
    RUN_COMPARISON = True  # 是否运行方法对比实验
    RUN_SINGLE_METHOD = False  # 是否只运行单一方法

    # 1. 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        DATA_WITH_MISSING_PATH,
        TRUE_DATA_PATH,
        index_col=0
    )

    # 2. 准备训练和验证集
    train_data, val_data = prepare_train_val_split(data_with_missing, split_ratio=0.8)

    # 3. MSSA分析
    print("\n对缺失值进行线性插值（仅用于MSSA）...")
    data_interpolated = train_data.interpolate(method='linear', limit_direction='both')

    mssa_model = perform_mssa_analysis(
        data_interpolated.values,
        window_length=MSSA_WINDOW_LENGTH,
        n_components=MSSA_N_COMPONENTS
    )

    if RUN_COMPARISON:
        # 4. 运行方法对比实验
        results, imputers, imputed_data_dict = compare_modal_fusion_methods(
            data_with_missing,
            true_data,
            missing_mask,
            mssa_model,
            train_data,
            val_data
        )

        return results, imputers, imputed_data_dict

    elif RUN_SINGLE_METHOD:
        # 4. 训练单一方法（自适应模态融合）
        imputer = train_modal_fusion_transformer(
            train_data,
            val_data,
            mssa_model=mssa_model,
            modal_fusion_type='adaptive',
            fusion_method='concat'
        )

        # 5. 填补和评估
        imputed_data, metrics = impute_and_evaluate_modal(
            imputer,
            data_with_missing,
            true_data,
            missing_mask,
            mssa_model=mssa_model,
            method_name="Adaptive_Modal_Fusion"
        )

        # 6. 可视化结果
        single_result_dict = {'Adaptive_Modal_Fusion': imputed_data}
        visualize_modal_fusion_results(
            data_with_missing,
            single_result_dict,
            true_data,
            missing_mask,
            time_range=(200, 400)
        )

        # 7. 可视化模态注意力
        imputer.plot_modal_attention(n_samples=3)

        print("\n处理完成！")
        return imputed_data, metrics, imputer


def create_synthetic_data_experiment():
    """
    创建合成数据进行实验演示
    """
    print("="*70)
    print("合成数据模态融合实验")
    print("="*70)

    # 1. 创建合成数据
    N = 500   # 减少数据量用于测试
    P = 3     # 减少特征数用于测试
    K = 6     # 减少MSSA模态数

    print(f"生成合成数据: N={N}, P={P}, K={K}")

    # 生成示例数据
    t = np.arange(N)
    data = np.zeros((N, P))
    for i in range(P):
        trend = 0.01 * t + i
        seasonal = 2 * np.sin(2 * np.pi * t / 50 + i)
        noise = 0.3 * np.random.randn(N)  # 减少噪声
        data[:, i] = trend + seasonal + noise

    # 2. 创建缺失值
    missing_ratio = 0.15  # 减少缺失比例
    missing_mask = np.random.random((N, P)) < missing_ratio
    data_with_missing = data.copy()
    data_with_missing[missing_mask] = np.nan

    print(f"缺失值比例: {missing_ratio:.2%}")
    print(f"总缺失值数量: {np.sum(missing_mask)}")

    # 转换为DataFrame
    data_df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(P)])
    data_with_missing_df = pd.DataFrame(data_with_missing, columns=[f'Feature_{i}' for i in range(P)])

    # 3. 准备数据
    train_data, val_data = prepare_train_val_split(data_with_missing_df, split_ratio=0.8)

    # 4. MSSA分析
    print("\n进行MSSA分析...")
    data_interpolated = train_data.interpolate(method='linear', limit_direction='both')

    # 确保没有NaN值
    data_interpolated = data_interpolated.fillna(method='ffill').fillna(method='bfill')
    data_interpolated = data_interpolated.fillna(data_interpolated.mean())

    try:
        mssa_model = perform_mssa_analysis(
            data_interpolated.values,
            window_length=min(30, len(data_interpolated)//3),  # 适应数据长度
            n_components=K
        )
    except Exception as e:
        print(f"MSSA分析出错: {e}")
        print("使用简化的MSSA参数...")
        mssa_model = perform_mssa_analysis(
            data_interpolated.values,
            window_length=20,
            n_components=4
        )

    # 5. 运行对比实验
    print("\n开始模态融合方法对比...")
    try:
        results, imputers, imputed_data_dict = compare_modal_fusion_methods(
            data_with_missing_df,
            data_df,
            missing_mask,
            mssa_model,
            train_data,
            val_data
        )
    except Exception as e:
        print(f"对比实验出错: {e}")
        print("尝试运行简化实验...")

        # 简化实验：只测试一种方法
        print("\n运行简化的单一方法测试...")
        imputer = train_modal_fusion_transformer(
            train_data,
            val_data,
            mssa_model=mssa_model,
            modal_fusion_type='adaptive',
            fusion_method='concat'
        )

        imputed_data, metrics = impute_and_evaluate_modal(
            imputer,
            data_with_missing_df,
            data_df,
            missing_mask,
            mssa_model=mssa_model,
            method_name="Adaptive_Modal_Fusion"
        )

        results = {'Adaptive_Modal_Fusion': metrics}
        imputers = {'Adaptive_Modal_Fusion': imputer}
        imputed_data_dict = {'Adaptive_Modal_Fusion': imputed_data}

    print("\n合成数据实验完成！")
    return results, imputers, imputed_data_dict


if __name__ == "__main__":
    # 选择运行模式
    use_real_data = True  # 设置为True使用真实数据，False使用合成数据

    if use_real_data:
        # 使用真实数据
        results = main()
    else:
        # 使用合成数据进行演示
        results = create_synthetic_data_experiment()

    print("\n所有实验完成！")
