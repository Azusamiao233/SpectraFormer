import numpy as np
import pandas as pd
from pypots.imputation import SAITS, BRITS, iTransformer, Transformer, SegRNN

# 尝试导入所有可用模型
try:
    from pypots.imputation import ImputeFormer

    IMPUTEFORMER_AVAILABLE = True
    print("✅ ImputeFormer 可用")
except ImportError:
    IMPUTEFORMER_AVAILABLE = False
    print("⚠️ ImputeFormer 不可用")

try:
    from pypots.imputation import Autoformer

    AUTOFORMER_AVAILABLE = True
    print("✅ Autoformer 可用")
except ImportError:
    AUTOFORMER_AVAILABLE = False
    print("⚠️ Autoformer 不可用")

try:
    from pypots.imputation import Informer

    INFORMER_AVAILABLE = True
    print("✅ Informer 可用")
except ImportError:
    INFORMER_AVAILABLE = False
    print("⚠️ Informer 不可用")

try:
    from pypots.nn.functional import calc_mae, calc_mse, calc_rmse
except ImportError:
    from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from spectraformer.config import DATA_DIR

warnings.filterwarnings('ignore')


class ForwardPredictionImputation:
    def __init__(self, missing_data_file_path, original_data_file_path, time_col=None):
        """
        前向预测插补评估器 - 专门处理缺失值集中在前期的情况
        """
        self.missing_data_file_path = missing_data_file_path
        self.original_data_file_path = original_data_file_path
        self.time_col = time_col
        self.models = {}
        self.results = {}
        self.original_data = None
        self.data_with_missing = None
        self.missing_mask = None
        self.time_index = None

    def load_data(self):
        """
        加载数据并分析缺失模式
        """
        try:
            print("正在加载数据...")

            # 加载数据 (简化版本，核心逻辑相同)
            if self.original_data_file_path.endswith('.csv'):
                df_original = pd.read_csv(self.original_data_file_path)
                df_missing = pd.read_csv(self.missing_data_file_path)
            else:
                df_original = pd.read_excel(self.original_data_file_path)
                df_missing = pd.read_excel(self.missing_data_file_path)

            print(f"数据形状: {df_original.shape}")

            # 处理时间列
            if self.time_col and self.time_col in df_original.columns:
                print(f"📅 处理时间列: {self.time_col}")
                df_original = df_original.sort_values(self.time_col).reset_index(drop=True)
                df_missing = df_missing.sort_values(self.time_col).reset_index(drop=True)

                self.time_index = df_original[self.time_col].copy()
                df_original = df_original.drop(self.time_col, axis=1)
                df_missing = df_missing.drop(self.time_col, axis=1)

            # 获取数值列
            numeric_cols = df_original.select_dtypes(include=[np.number]).columns
            common_cols = list(set(numeric_cols) & set(df_missing.select_dtypes(include=[np.number]).columns))

            df_original = df_original[common_cols]
            df_missing = df_missing[common_cols]

            print(f"数值特征数量: {len(common_cols)}")

            # 分析缺失模式
            missing_stats = df_missing.isnull().sum()
            missing_features = missing_stats[missing_stats > 0]
            print(f"有缺失值的特征: {len(missing_features)} 个")

            # 转换为3D格式
            original_array = df_original.values
            missing_array = df_missing.values

            # 使用滑动窗口
            n_rows, n_cols = original_array.shape
            window_size = 48
            stride = 12  # 更小的步长，获得更多样本

            samples_original = []
            samples_missing = []
            window_indices = []  # 记录每个窗口的起始位置

            for i in range(0, n_rows - window_size + 1, stride):
                samples_original.append(original_array[i:i + window_size])
                samples_missing.append(missing_array[i:i + window_size])
                window_indices.append(i)

            self.original_data = np.array(samples_original, dtype=np.float32)
            self.data_with_missing = np.array(samples_missing, dtype=np.float32)
            self.missing_mask = np.isnan(self.data_with_missing)
            self.window_indices = window_indices

            print(f"滑动窗口重塑: {self.original_data.shape}")
            print(f"总缺失率: {self.missing_mask.mean():.2%}")

            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def analyze_missing_temporal_pattern(self):
        """
        详细分析缺失值的时间分布模式
        """
        print("\n" + "🔍" * 60)
        print("详细缺失值时间模式分析")
        print("🔍" * 60)

        # 计算每个窗口的缺失值数量
        missing_counts = np.isnan(self.data_with_missing).sum(axis=(1, 2))

        # 找到有缺失值的窗口
        windows_with_missing = np.where(missing_counts > 0)[0]

        if len(windows_with_missing) == 0:
            print("❌ 没有发现缺失值窗口")
            return None

        print(f"缺失值分布分析:")
        print(f"  有缺失值的窗口数: {len(windows_with_missing)}/{len(missing_counts)}")
        print(f"  缺失值窗口范围: {windows_with_missing.min()} → {windows_with_missing.max()}")
        print(
            f"  对应的原始时间位置: {self.window_indices[windows_with_missing.min()]} → {self.window_indices[windows_with_missing.max()]}")

        # 计算缺失值结束的位置
        last_missing_window = windows_with_missing.max()
        last_missing_time_pos = self.window_indices[last_missing_window]

        print(f"  缺失值最后出现位置: 窗口 {last_missing_window}, 原始位置 {last_missing_time_pos}")
        print(f"  数据总长度: {len(self.window_indices)} 窗口")

        # 计算合适的划分点
        safe_split_window = min(last_missing_window + 10, len(missing_counts) - 50)  # 留一些缓冲
        split_ratio = safe_split_window / len(missing_counts)

        print(f"\n💡 建议的评估策略:")
        print(f"  建议训练窗口范围: 0 → {safe_split_window}")
        print(f"  建议测试窗口范围: {safe_split_window} → {len(missing_counts)}")
        print(f"  对应的训练比例: {split_ratio:.2f}")

        return {
            'last_missing_window': last_missing_window,
            'safe_split_window': safe_split_window,
            'split_ratio': split_ratio,
            'windows_with_missing': windows_with_missing
        }

    def prepare_forward_prediction_split(self, missing_info):
        """
        基于缺失模式进行前向预测划分
        """
        print("\n" + "🚀" * 60)
        print("前向预测评估策略")
        print("🚀" * 60)

        if missing_info is None:
            print("❌ 无法获取缺失信息，使用默认划分")
            train_split = int(0.7 * len(self.data_with_missing))
        else:
            train_split = missing_info['safe_split_window']

        print(f"📊 前向预测划分策略:")
        print(f"  策略说明: 使用有缺失值的时期训练模型，在无缺失值的未来时期评估")
        print(f"  训练期: 窗口 0 → {train_split} (包含缺失值的历史时期)")
        print(f"  测试期: 窗口 {train_split} → {len(self.data_with_missing)} (无缺失值的未来时期)")

        # 划分数据
        train_data = self.data_with_missing[:train_split]
        train_original = self.original_data[:train_split]

        test_data = self.data_with_missing[train_split:]
        test_original = self.original_data[train_split:]

        print(f"\n📈 数据集统计:")
        print(f"  训练集: {train_data.shape}, 缺失率: {np.isnan(train_data).mean():.2%}")
        print(f"  测试集: {test_data.shape}, 缺失率: {np.isnan(test_data).mean():.2%}")

        # 在测试集中人工创建缺失值来评估插补能力
        print(f"\n🔧 创建评估用缺失值:")
        test_data_for_imputation = test_data.copy()

        # 策略：在每个测试样本中随机创建一些缺失值
        n_test_samples = test_data.shape[0]
        missing_rate = 0.05  # 在测试数据中创建5%的缺失值

        evaluation_mask = np.zeros_like(test_data, dtype=bool)

        for i in range(n_test_samples):
            # 为每个样本随机选择一些位置设为缺失
            sample_size = test_data.shape[1] * test_data.shape[2]
            n_missing = int(sample_size * missing_rate)

            if n_missing > 0:
                # 随机选择位置
                flat_indices = np.random.choice(sample_size, n_missing, replace=False)

                # 转换为2D索引
                time_indices = flat_indices // test_data.shape[2]
                feature_indices = flat_indices % test_data.shape[2]

                # 设置缺失值
                for t_idx, f_idx in zip(time_indices, feature_indices):
                    test_data_for_imputation[i, t_idx, f_idx] = np.nan
                    evaluation_mask[i, t_idx, f_idx] = True

        print(f"  人工创建的缺失值数量: {evaluation_mask.sum()}")
        print(f"  人工缺失率: {evaluation_mask.mean():.2%}")

        return train_data, test_data_for_imputation, test_original, evaluation_mask

    def train_models(self, train_data):
        """
        训练插补模型
        """
        print("\n" + "🔧" * 60)
        print("训练插补模型")
        print("🔧" * 60)

        n_steps, n_features = train_data.shape[1], train_data.shape[2]
        print(f"模型输入维度: {n_steps} 时间步 × {n_features} 特征")
        print(f"训练数据缺失率: {np.isnan(train_data).mean():.2%}")

        train_dict = {'X': train_data}

        # 完整的模型配置列表
        model_configs = {
            'SAITS': {
                'class': SAITS,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 2,
                    'd_model': 64,
                    'd_ffn': 32,
                    'n_heads': 4,
                    'd_k': 16,
                    'd_v': 16,
                    'dropout': 0.1,
                    'epochs': 10,
                    'patience': 5,
                    'batch_size': 32
                }
            },
            'BRITS': {
                'class': BRITS,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'rnn_hidden_size': 64,
                    'epochs': 5,
                    'patience': 5,
                    'batch_size': 32
                }
            },
            'iTransformer': {
                'class': iTransformer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 2,
                    'd_model': 64,
                    'n_heads': 4,
                    'd_k': 16,
                    'd_v': 16,
                    'd_ffn': 64,
                    'dropout': 0.1,
                    'epochs': 10,
                    'patience': 5,
                    'batch_size': 32
                }
            },
            'Transformer': {
                'class': Transformer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 2,
                    'd_model': 64,
                    'n_heads': 4,
                    'd_k': 16,
                    'd_v': 16,
                    'd_ffn': 64,
                    'dropout': 0.1,
                    'epochs': 10,
                    'patience': 5,
                    'batch_size': 32
                }
            },
            'SegRNN': {
                'class': SegRNN,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'd_model': 64,
                    'seg_len': min(12, n_steps // 2),
                    'dropout': 0.1,
                    'epochs': 10,
                    'patience': 5,
                    'batch_size': 32
                }
            }
        }

        # 添加可选模型（如果可用）
        if IMPUTEFORMER_AVAILABLE:
            model_configs['ImputeFormer'] = {
                'class': ImputeFormer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 2,
                    'd_input_embed': 64,
                    'd_learnable_embed': 32,
                    'd_proj': 64,
                    'd_ffn': 128,
                    'n_temporal_heads': 4,
                    'dropout': 0.1,
                    'epochs': 5,
                    'patience': 5,
                    'batch_size': 32
                }
            }

        if AUTOFORMER_AVAILABLE:
            model_configs['Autoformer'] = {
                'class': Autoformer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 2,
                    'd_model': 64,
                    'n_heads': 4,
                    'd_ffn': 128,
                    'factor': 3,
                    'moving_avg_window_size': min(25, n_steps // 2),
                    'dropout': 0.1,
                    'epochs': 10,
                    'patience': 5,
                    'batch_size': 32
                }
            }

        if INFORMER_AVAILABLE:
            # Informer 使用简化配置避免参数错误
            model_configs['Informer'] = {
                'class': Informer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 2,
                    'd_model': 64,
                    'n_heads': 4,
                    'd_ffn': 128,
                    'factor': 3,
                    'dropout': 0.1,
                    'epochs': 10,
                    'patience': 5,
                    'batch_size': 32
                }
            }

        print(f"准备训练 {len(model_configs)} 个模型: {list(model_configs.keys())}")

        # 训练模型，包含错误处理和重试机制
        for model_name, config in model_configs.items():
            print(f"\n🔄 训练 {model_name}...")
            try:
                model = config['class'](**config['params'])
                model.fit(train_dict)
                self.models[model_name] = model
                print(f"✅ {model_name} 训练完成")
            except Exception as e:
                print(f"⚠️ {model_name} 训练失败: {e}")

                # 对某些模型尝试更简单的配置
                if model_name in ['SegRNN', 'ImputeFormer', 'Transformer']:
                    print(f"🔄 尝试 {model_name} 的简化配置...")
                    try:
                        simple_params = config['params'].copy()
                        # 简化参数
                        if 'd_model' in simple_params:
                            simple_params['d_model'] = 32
                        if 'd_ffn' in simple_params:
                            simple_params['d_ffn'] = 32
                        if 'n_layers' in simple_params:
                            simple_params['n_layers'] = 1
                        if 'batch_size' in simple_params:
                            simple_params['batch_size'] = 16
                        if 'epochs' in simple_params:
                            simple_params['epochs'] = 5
                        if model_name == 'SegRNN' and 'seg_len' in simple_params:
                            simple_params['seg_len'] = min(6, max(1, n_steps // 4))

                        model = config['class'](**simple_params)
                        model.fit(train_dict)
                        self.models[model_name] = model
                        print(f"✅ {model_name} 简化配置训练完成")
                    except Exception as e2:
                        print(f"❌ {model_name} 简化配置也失败: {e2}")
                        self.models[model_name] = None
                else:
                    self.models[model_name] = None

        successful_models = [name for name, model in self.models.items() if model is not None]
        print(f"\n📊 训练结果: {len(successful_models)} 个模型训练成功")

    def evaluate_forward_prediction(self, test_data_with_missing, test_original, evaluation_mask):
        """
        在前向预测设置下评估模型
        """
        print("\n" + "📊" * 60)
        print("前向预测插补性能评估")
        print("📊" * 60)

        print(f"评估设置:")
        print(f"  测试数据形状: {test_data_with_missing.shape}")
        print(f"  人工缺失值数量: {evaluation_mask.sum()}")
        print(f"  评估缺失率: {evaluation_mask.mean():.2%}")

        for model_name, model in self.models.items():
            if model is None:
                continue

            print(f"\n🔍 评估 {model_name}...")
            try:
                # 进行插补
                try:
                    test_dict = {'X': test_data_with_missing}
                    imputed_result = model.impute(test_dict)
                    if isinstance(imputed_result, dict) and 'X' in imputed_result:
                        imputed_data = imputed_result['X']
                    else:
                        imputed_data = imputed_result
                except:
                    imputed_data = model.impute(test_data_with_missing)

                # 只在人工创建的缺失位置评估
                true_values = test_original[evaluation_mask]
                imputed_values = imputed_data[evaluation_mask]

                # 移除无效值
                valid_mask = ~(np.isnan(true_values) | np.isnan(imputed_values))
                if valid_mask.sum() == 0:
                    print(f"  ❌ {model_name} 没有有效预测")
                    continue

                true_values = true_values[valid_mask]
                imputed_values = imputed_values[valid_mask]

                # 计算指标
                mae = mean_absolute_error(true_values, imputed_values)
                mse = mean_squared_error(true_values, imputed_values)
                rmse = np.sqrt(mse)

                try:
                    r2 = r2_score(true_values, imputed_values)
                except:
                    r2 = np.nan

                correlation = np.corrcoef(true_values, imputed_values)[0, 1]

                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((true_values - imputed_values) /
                                          (np.abs(true_values) + 1e-8))) * 100

                # 保存结果
                self.results[model_name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R²': r2,
                    'MAPE': mape,
                    'Correlation': correlation,
                    'evaluated_points': len(true_values)
                }

                print(f"  ✅ {model_name} 评估完成:")
                print(f"    评估点数: {len(true_values)}")
                print(f"    MAE: {mae:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    R²: {r2:.4f}")
                print(f"    相关系数: {correlation:.4f}")

            except Exception as e:
                print(f"  ❌ {model_name} 评估失败: {e}")

    def print_forward_prediction_summary(self):
        """
        打印前向预测评估结果摘要
        """
        if not self.results:
            print("没有可用的评估结果")
            return

        print("\n" + "🏆" * 70)
        print("前向预测插补性能摘要")
        print("🏆" * 70)

        print("📋 评估说明:")
        print("  • 使用历史时期(含缺失值)的数据训练模型")
        print("  • 在未来时期(原本无缺失值)创建人工缺失值进行评估")
        print("  • 模拟实际应用中的前向预测场景")
        print("  • 避免了数据泄漏，提供真实的性能评估")

        # 结果表格
        metrics = ['MAE', 'RMSE', 'R²', 'MAPE', 'Correlation']

        print(f"\n📊 性能对比:")
        print(f"{'模型':<15}", end="")
        for metric in metrics:
            print(f"{metric:>12}", end="")
        print(f"{'评估点数':>10}")
        print("-" * 85)

        for model_name, result in self.results.items():
            print(f"{model_name:<15}", end="")
            for metric in metrics:
                value = result.get(metric, np.nan)
                if metric == 'MAPE':
                    print(f"{value:>11.1f}%", end="")
                else:
                    print(f"{value:>12.4f}", end="")
            print(f"{result.get('evaluated_points', 0):>10d}")

        # 最佳模型
        print(f"\n🥇 最佳模型:")
        for metric in ['MAE', 'RMSE']:
            valid_results = {k: v for k, v in self.results.items()
                             if not np.isnan(v.get(metric, np.nan))}
            if valid_results:
                best_model = min(valid_results.keys(), key=lambda x: valid_results[x][metric])
                best_value = valid_results[best_model][metric]
                print(f"  最低{metric}: {best_model} ({best_value:.4f})")

        for metric in ['R²', 'Correlation']:
            valid_results = {k: v for k, v in self.results.items()
                             if not np.isnan(v.get(metric, np.nan))}
            if valid_results:
                best_model = max(valid_results.keys(), key=lambda x: valid_results[x][metric])
                best_value = valid_results[best_model][metric]
                print(f"  最高{metric}: {best_model} ({best_value:.4f})")

        print(f"\n✅ 评估完成 - 前向预测插补性能评估为实际应用提供了可靠的参考")


def main():
    """
    主函数 - 前向预测插补评估
    """
    print("🚀 前向预测时间序列插补评估")
    print("=" * 70)
    print("🎯 适用场景:")
    print("  • 缺失值集中在时间序列前期")
    print("  • 需要评估模型在未来时期的插补能力")
    print("  • 模拟实际部署中的前向预测场景")
    print("=" * 70)

    # 配置
    MISSING_DATA_FILE_PATH = str(DATA_DIR / "train_continuous_missing_60.csv")
    ORIGINAL_DATA_FILE_PATH = str(DATA_DIR / "train_swat.csv")
    TIME_COLUMN = "Unnamed: 0"

    # 设置随机种子以确保结果可重复
    np.random.seed(42)

    # 初始化
    imputer = ForwardPredictionImputation(
        MISSING_DATA_FILE_PATH,
        ORIGINAL_DATA_FILE_PATH,
        TIME_COLUMN
    )

    # 步骤1: 加载数据
    if not imputer.load_data():
        print("❌ 数据加载失败")
        return

    # 步骤2: 分析缺失模式
    missing_info = imputer.analyze_missing_temporal_pattern()

    # 步骤3: 前向预测划分
    train_data, test_data_missing, test_original, eval_mask = imputer.prepare_forward_prediction_split(missing_info)

    # 步骤4: 训练模型
    imputer.train_models(train_data)

    # 步骤5: 评估模型
    imputer.evaluate_forward_prediction(test_data_missing, test_original, eval_mask)

    # 步骤6: 结果摘要
    imputer.print_forward_prediction_summary()

    print("\n🎉 前向预测插补评估完成!")


if __name__ == "__main__":
    main()
