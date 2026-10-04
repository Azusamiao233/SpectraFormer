import  numpy as np

import pandas as pd
from pypots.imputation import SAITS, BRITS, iTransformer

try:
    from pypots.nn.functional import calc_mae, calc_mse, calc_rmse
except ImportError:
    from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from spectraformer.config import DATA_DIR

warnings.filterwarnings('ignore')


class TimeSeriesImputation:
    def __init__(self, missing_data_file_path, original_data_file_path, time_col=None):
        """
        初始化插补器
        Args:
            missing_data_file_path: 带缺失值的数据文件路径
            original_data_file_path: 原始完整数据文件路径
            time_col: 时间列名（可选）
        """
        self.missing_data_file_path = missing_data_file_path
        self.original_data_file_path = original_data_file_path
        self.time_col = time_col
        self.models = {}
        self.results = {}
        self.original_data = None
        self.data_with_missing = None
        self.missing_mask = None

    def load_data(self):
        """
        加载原始完整数据和带缺失值的数据
        """
        try:
            print("正在加载原始完整数据...")
            # 加载原始完整数据
            if self.original_data_file_path.endswith('.csv'):
                df_original = pd.read_csv(self.original_data_file_path)
            elif self.original_data_file_path.endswith(('.xlsx', '.xls')):
                df_original = pd.read_excel(self.original_data_file_path)
            else:
                raise ValueError("支持的文件格式: CSV, Excel")

            print("正在加载带缺失值数据...")
            # 加载带缺失值的数据
            if self.missing_data_file_path.endswith('.csv'):
                df_missing = pd.read_csv(self.missing_data_file_path)
            elif self.missing_data_file_path.endswith(('.xlsx', '.xls')):
                df_missing = pd.read_excel(self.missing_data_file_path)
            else:
                raise ValueError("支持的文件格式: CSV, Excel")

            print(f"原始数据形状: {df_original.shape}")
            print(f"缺失数据形状: {df_missing.shape}")
            print(f"原始数据列名: {df_original.columns.tolist()}")
            print(f"缺失数据列名: {df_missing.columns.tolist()}")

            # 处理时间列
            if self.time_col and self.time_col in df_original.columns:
                df_original[self.time_col] = pd.to_datetime(df_original[self.time_col])
                df_original = df_original.sort_values(self.time_col)
                df_original = df_original.drop(self.time_col, axis=1)

            if self.time_col and self.time_col in df_missing.columns:
                df_missing[self.time_col] = pd.to_datetime(df_missing[self.time_col])
                df_missing = df_missing.sort_values(self.time_col)
                df_missing = df_missing.drop(self.time_col, axis=1)

            # 只保留数值列
            numeric_cols_original = df_original.select_dtypes(include=[np.number]).columns
            numeric_cols_missing = df_missing.select_dtypes(include=[np.number]).columns

            # 确保两个数据集有相同的列
            common_cols = list(set(numeric_cols_original) & set(numeric_cols_missing))
            if len(common_cols) == 0:
                raise ValueError("原始数据和缺失数据没有共同的数值列")

            df_original = df_original[common_cols]
            df_missing = df_missing[common_cols]

            print(f"共同数值列: {common_cols}")
            print(f"原始数据缺失值统计:")
            print(df_original.isnull().sum())
            print(f"带缺失值数据缺失值统计:")
            print(df_missing.isnull().sum())

            # 确保两个数据集形状一致
            min_rows = min(df_original.shape[0], df_missing.shape[0])
            df_original = df_original.iloc[:min_rows]
            df_missing = df_missing.iloc[:min_rows]

            print(f"调整后形状 - 原始: {df_original.shape}, 缺失: {df_missing.shape}")

            # 转换为PyPOTS需要的格式 (n_samples, n_steps, n_features)
            original_array = df_original.values
            missing_array = df_missing.values

            # 如果数据是2D，需要reshape为3D
            if len(original_array.shape) == 2:
                n_rows, n_cols = original_array.shape

                # 根据数据量自动确定合适的维度
                if n_rows >= 48:  # 如果有足够的时间步
                    n_steps = min(48, n_rows)
                    n_features = n_cols
                    n_samples = max(1, n_rows // n_steps)

                    # 截取数据以适应形状
                    original_array = original_array[:n_samples * n_steps]
                    missing_array = missing_array[:n_samples * n_steps]

                    original_array = original_array.reshape(n_samples, n_steps, n_features)
                    missing_array = missing_array.reshape(n_samples, n_steps, n_features)
                else:
                    # 假设每列是一个时间序列
                    n_samples = 1
                    n_steps = n_rows
                    n_features = n_cols
                    original_array = original_array.reshape(n_samples, n_steps, n_features)
                    missing_array = missing_array.reshape(n_samples, n_steps, n_features)

            print(f"重塑后数据形状: {original_array.shape}")

            # 存储数据
            self.original_data = original_array.astype(np.float32)
            self.data_with_missing = missing_array.astype(np.float32)

            # 创建缺失值掩码
            self.missing_mask = np.isnan(self.data_with_missing)

            # 验证原始数据中对应位置的值
            actual_missing_positions = self.missing_mask.sum()
            original_missing_positions = np.isnan(self.original_data).sum()

            print(f"缺失值位置数量: {actual_missing_positions}")
            print(f"原始数据缺失值数量: {original_missing_positions}")
            print(f"缺失率: {self.missing_mask.mean():.2%}")

            # 检查数据一致性
            non_missing_mask = ~self.missing_mask
            if non_missing_mask.sum() > 0:
                original_non_missing = self.original_data[non_missing_mask]
                missing_non_missing = self.data_with_missing[non_missing_mask]
                consistency = np.allclose(original_non_missing, missing_non_missing, equal_nan=True)
                print(f"非缺失位置数据一致性: {'✓' if consistency else '✗'}")

            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def prepare_data(self, train_ratio=0.8):
        """
        准备训练和测试数据
        """
        n_samples = self.data_with_missing.shape[0]
        train_size = int(train_ratio * n_samples)

        train_data = self.data_with_missing[:train_size]
        test_data = self.data_with_missing[train_size:]
        test_original = self.original_data[train_size:]
        test_missing_mask = self.missing_mask[train_size:]

        return train_data, test_data, test_original, test_missing_mask

    def train_models(self, train_data):
        """
        训练所有模型
        """
        print("\n开始训练模型...")

        n_steps, n_features = train_data.shape[1], train_data.shape[2]
        print(f"模型输入维度: n_steps={n_steps}, n_features={n_features}")

        # 将数据转换为PyPOTS期望的字典格式
        train_dict = {
            'X': train_data
        }

        # SAITS模型
        print("\n训练SAITS模型...")
        try:
            self.models['SAITS'] = SAITS(
                n_steps=n_steps,
                n_features=n_features,
                n_layers=2,
                d_model=128,  # 减小模型大小
                d_ffn=64,
                n_heads=4,
                d_k=32,
                d_v=32,
                dropout=0.1,
                epochs=5,
                patience=3,
                batch_size=16
            )
            self.models['SAITS'].fit(train_dict)
            print("✓ SAITS训练完成")
        except Exception as e:
            print(f"✗ SAITS训练失败: {e}")
            print(f"尝试更简单的SAITS配置...")
            try:
                self.models['SAITS'] = SAITS(
                    n_steps=n_steps,
                    n_features=n_features,
                    n_layers=1,
                    d_model=64,
                    d_ffn=32,
                    n_heads=2,
                    d_k=32,
                    d_v=32,
                    epochs=3,
                    batch_size=8
                )
                self.models['SAITS'].fit(train_dict)
                print("✓ SAITS简单配置训练完成")
            except Exception as e2:
                print(f"✗ SAITS简单配置也失败: {e2}")
                self.models['SAITS'] = None

        # BRITS模型
        print("\n训练BRITS模型...")
        try:
            self.models['BRITS'] = BRITS(
                n_steps=n_steps,
                n_features=n_features,
                rnn_hidden_size=64,
                epochs=5,
                patience=3,
                batch_size=16
            )
            self.models['BRITS'].fit(train_dict)
            print("✓ BRITS训练完成")
        except Exception as e:
            print(f"✗ BRITS训练失败: {e}")
            print(f"尝试更简单的BRITS配置...")
            try:
                self.models['BRITS'] = BRITS(
                    n_steps=n_steps,
                    n_features=n_features,
                    rnn_hidden_size=32,
                    epochs=3,
                    batch_size=8
                )
                self.models['BRITS'].fit(train_dict)
                print("✓ BRITS简单配置训练完成")
            except Exception as e2:
                print(f"✗ BRITS简单配置也失败: {e2}")
                self.models['BRITS'] = None

        # iTransformer模型
        print("\n训练iTransformer模型...")
        try:
            self.models['iTransformer'] = iTransformer(
                n_steps=n_steps,
                n_features=n_features,
                n_layers=2,
                d_model=64,
                n_heads=4,
                d_k=16,
                d_v=16,
                d_ffn=128,
                dropout=0.1,
                epochs=5,
                patience=3,
                batch_size=16
            )
            self.models['iTransformer'].fit(train_dict)
            print("✓ iTransformer训练完成")
        except Exception as e:
            print(f"✗ iTransformer训练失败: {e}")
            print(f"尝试更简单的iTransformer配置...")
            try:
                self.models['iTransformer'] = iTransformer(
                    n_steps=n_steps,
                    n_features=n_features,
                    n_layers=1,
                    d_model=32,
                    n_heads=2,
                    d_k=16,
                    d_v=16,
                    d_ffn=64,
                    epochs=3,
                    batch_size=8
                )
                self.models['iTransformer'].fit(train_dict)
                print("✓ iTransformer简单配置训练完成")
            except Exception as e2:
                print(f"✗ iTransformer简单配置也失败: {e2}")
                self.models['iTransformer'] = None

        # 统计成功训练的模型
        successful_models = [name for name, model in self.models.items() if model is not None]
        print(f"\n成功训练的模型: {successful_models}")

    def evaluate_models(self, test_data, test_original, test_missing_mask):
        """
        评估所有模型 - 使用真实原始数据作为参考
        """
        print("\n" + "=" * 60)
        print("开始评估模型...")
        print("=" * 60)

        for model_name, model in self.models.items():
            if model is None:
                continue

            print(f"\n评估 {model_name}...")
            try:
                # 进行插补 - 尝试不同的数据格式
                try:
                    # 方法1: 使用字典格式
                    test_dict = {'X': test_data}
                    imputed_result = model.impute(test_dict)
                    if isinstance(imputed_result, dict) and 'X' in imputed_result:
                        imputed_data = imputed_result['X']
                    else:
                        # 如果返回的不是字典，直接使用结果
                        imputed_data = imputed_result
                except:
                    # 方法2: 直接传入numpy数组
                    imputed_data = model.impute(test_data)

                # 只在实际缺失位置计算指标
                actual_missing_positions = test_missing_mask

                if actual_missing_positions.sum() == 0:
                    print(f"警告: 测试集中没有缺失值，使用所有位置进行评估")
                    actual_missing_positions = np.ones_like(test_missing_mask, dtype=bool)

                # 使用真实原始数据作为参考
                true_values = test_original[actual_missing_positions]
                imputed_values = imputed_data[actual_missing_positions]

                # 移除可能的nan值
                valid_mask = ~(np.isnan(true_values) | np.isnan(imputed_values))
                if valid_mask.sum() == 0:
                    print(f"警告: {model_name} 没有有效的预测值")
                    continue

                true_values = true_values[valid_mask]
                imputed_values = imputed_values[valid_mask]

                # 计算各种指标
                mae = mean_absolute_error(true_values, imputed_values)
                mse = mean_squared_error(true_values, imputed_values)
                rmse = np.sqrt(mse)

                # 计算R²分数
                try:
                    r2 = r2_score(true_values, imputed_values)
                except:
                    r2 = np.nan

                # 计算MAPE (Mean Absolute Percentage Error)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((true_values - imputed_values) /
                                          (np.abs(true_values) + 1e-8))) * 100

                # 计算相关系数
                correlation = np.corrcoef(true_values, imputed_values)[0, 1]

                # 计算SMAPE (Symmetric Mean Absolute Percentage Error)
                smape = np.mean(2 * np.abs(true_values - imputed_values) /
                                (np.abs(true_values) + np.abs(imputed_values) + 1e-8)) * 100

                # 计算NRMSE (Normalized RMSE)
                data_range = np.max(true_values) - np.min(true_values)
                nrmse = rmse / (data_range + 1e-8) * 100

                self.results[model_name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R²': r2,
                    'MAPE': mape,
                    'SMAPE': smape,
                    'NRMSE': nrmse,
                    'Correlation': correlation,
                    'imputed_data': imputed_data,
                    'valid_predictions': valid_mask.sum(),
                    'missing_predictions': actual_missing_positions.sum()
                }

                print(f"✓ {model_name} 评估完成:")
                print(f"    缺失值预测数量: {actual_missing_positions.sum()}")
                print(f"    有效预测数量: {valid_mask.sum()}")
                print(f"    MAE: {mae:.4f}")
                print(f"    MSE: {mse:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    R²: {r2:.4f}")
                print(f"    MAPE: {mape:.2f}%")
                print(f"    SMAPE: {smape:.2f}%")
                print(f"    NRMSE: {nrmse:.2f}%")
                print(f"    Correlation: {correlation:.4f}")

            except Exception as e:
                print(f"✗ {model_name} 评估失败: {e}")
                import traceback
                traceback.print_exc()

    def plot_results(self, test_data, test_original, sample_idx=0, feature_idx=0):
        """
        绘制结果对比图 - 使用真实原始数据
        """
        if not self.results:
            print("没有可用的结果进行绘图")
            return

        if sample_idx >= test_data.shape[0] or feature_idx >= test_data.shape[2]:
            print(f"索引超出范围，使用默认值 sample_idx=0, feature_idx=0")
            sample_idx, feature_idx = 0, 0

        plt.figure(figsize=(15, 8))

        # 获取时间步长
        time_steps = range(test_data.shape[1])

        # 数据
        data_with_missing = test_data[sample_idx, :, feature_idx]
        true_original = test_original[sample_idx, :, feature_idx]

        # 绘制真实完整数据
        plt.plot(time_steps, true_original, 'k-', label='真实完整数据', linewidth=2, alpha=0.8)

        # 标记观测值（非缺失值）
        observed_mask = ~np.isnan(data_with_missing)
        plt.scatter(np.array(time_steps)[observed_mask],
                    data_with_missing[observed_mask],
                    color='black', s=30, label='观测值', zorder=5)

        # 标记缺失值位置
        missing_mask = np.isnan(data_with_missing)
        if missing_mask.sum() > 0:
            plt.scatter(np.array(time_steps)[missing_mask],
                        true_original[missing_mask],
                        color='red', s=30, marker='x', label='缺失位置真实值', zorder=5)

        # 绘制各模型的插补结果
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, result) in enumerate(self.results.items()):
            if 'imputed_data' in result:
                imputed_values = result['imputed_data'][sample_idx, :, feature_idx]
                plt.plot(time_steps, imputed_values, '--',
                         color=colors[i % len(colors)],
                         label=f'{model_name} (RMSE: {result["RMSE"]:.3f})',
                         linewidth=2, alpha=0.7)

        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.title(f'时间序列插补结果对比 (样本 {sample_idx}, 特征 {feature_idx})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """
        打印结果摘要
        """
        if not self.results:
            print("没有可用的结果")
            return

        print("\n" + "=" * 80)
        print("模型性能摘要")
        print("=" * 80)

        # 创建结果表格
        metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE', 'Correlation']

        print(f"{'模型':<15}", end="")
        for metric in metrics:
            print(f"{metric:>12}", end="")
        print()
        print("-" * 87)

        for model_name, result in self.results.items():
            print(f"{model_name:<15}", end="")
            for metric in metrics:
                value = result.get(metric, np.nan)
                if metric == 'MAPE':
                    print(f"{value:>11.2f}%", end="")
                else:
                    print(f"{value:>12.4f}", end="")
            print()

        # 找出最佳模型
        print("\n最佳模型:")
        if len(self.results) > 0:
            # 最低RMSE
            valid_rmse_results = {k: v for k, v in self.results.items()
                                  if not np.isnan(v.get('RMSE', np.nan))}
            if valid_rmse_results:
                best_rmse_model = min(valid_rmse_results.keys(),
                                      key=lambda x: valid_rmse_results[x]['RMSE'])
                print(f"  最低RMSE: {best_rmse_model} ({valid_rmse_results[best_rmse_model]['RMSE']:.4f})")

            # 最高R²
            valid_r2_results = {k: v for k, v in self.results.items()
                                if not np.isnan(v.get('R²', np.nan))}
            if valid_r2_results:
                best_r2_model = max(valid_r2_results.keys(),
                                    key=lambda x: valid_r2_results[x]['R²'])
                print(f"  最高R²: {best_r2_model} ({valid_r2_results[best_r2_model]['R²']:.4f})")

            # 最高相关系数
            valid_corr_results = {k: v for k, v in self.results.items()
                                  if not np.isnan(v.get('Correlation', np.nan))}
            if valid_corr_results:
                best_corr_model = max(valid_corr_results.keys(),
                                      key=lambda x: valid_corr_results[x]['Correlation'])
                print(f"  最高相关系数: {best_corr_model} ({valid_corr_results[best_corr_model]['Correlation']:.4f})")


def main():
    """
    主函数 - 在这里指定数据文件路径
    """
    # ====== 在这里修改数据文件路径 ======
    MISSING_DATA_FILE_PATH = str(DATA_DIR / "train_continuous_missing_60.csv")
    ORIGINAL_DATA_FILE_PATH = str(DATA_DIR / "train_swat.csv")
    TIME_COLUMN = "Unnamed: 0"  # 如果有时间列，请指定列名，例如 "timestamp" 或 "date"

    print("时间序列插补模型测试")
    print("=" * 50)
    print(f"缺失数据文件: {MISSING_DATA_FILE_PATH}")
    print(f"原始数据文件: {ORIGINAL_DATA_FILE_PATH}")
    print(f"时间列: {TIME_COLUMN}")

    # 初始化插补器
    imputer = TimeSeriesImputation(MISSING_DATA_FILE_PATH, ORIGINAL_DATA_FILE_PATH, TIME_COLUMN)

    # 加载数据
    if not imputer.load_data():
        print("数据加载失败，程序退出")
        return

    # 准备数据
    train_data, test_data, test_original, test_missing_mask = imputer.prepare_data()

    print(f"\n数据划分:")
    print(f"  训练集形状: {train_data.shape}")
    print(f"  测试集形状: {test_data.shape}")
    print(f"  训练集缺失率: {np.isnan(train_data).mean():.2%}")
    print(f"  测试集缺失率: {np.isnan(test_data).mean():.2%}")
    print(f"  测试集缺失值数量: {test_missing_mask.sum()}")

    # 训练模型
    imputer.train_models(train_data)

    # 评估模型
    imputer.evaluate_models(test_data, test_original, test_missing_mask)

    # 显示结果摘要
    imputer.print_summary()

    # 绘制结果（可选）
    try:
        imputer.plot_results(test_data, test_original, sample_idx=0, feature_idx=0)
    except Exception as e:
        print(f"绘图失败: {e}")

    print("\n程序执行完成!")

def print_summary(self):
    """
    打印结果摘要
    """
    if not self.results:
        print("没有可用的结果")
        return

    print("\n" + "=" * 100)
    print("模型性能摘要")
    print("=" * 100)

    # 创建结果表格
    metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE', 'SMAPE', 'NRMSE', 'Correlation']

    print(f"{'模型':<15}", end="")
    for metric in metrics:
        print(f"{metric:>12}", end="")
    print()
    print("-" * 111)

    for model_name, result in self.results.items():
        print(f"{model_name:<15}", end="")
        for metric in metrics:
            value = result.get(metric, np.nan)
            if metric in ['MAPE', 'SMAPE', 'NRMSE']:
                print(f"{value:>11.2f}%", end="")
            else:
                print(f"{value:>12.4f}", end="")
        print()

    # 找出最佳模型
    print("\n最佳模型:")
    if len(self.results) > 0:
        # 最低RMSE
        valid_rmse_results = {k: v for k, v in self.results.items()
                              if not np.isnan(v.get('RMSE', np.nan))}
        if valid_rmse_results:
            best_rmse_model = min(valid_rmse_results.keys(),
                                  key=lambda x: valid_rmse_results[x]['RMSE'])
            print(f"  最低RMSE: {best_rmse_model} ({valid_rmse_results[best_rmse_model]['RMSE']:.4f})")

        # 最高R²
        valid_r2_results = {k: v for k, v in self.results.items()
                            if not np.isnan(v.get('R²', np.nan))}
        if valid_r2_results:
            best_r2_model = max(valid_r2_results.keys(),
                                key=lambda x: valid_r2_results[x]['R²'])
            print(f"  最高R²: {best_r2_model} ({valid_r2_results[best_r2_model]['R²']:.4f})")

        # 最高相关系数
        valid_corr_results = {k: v for k, v in self.results.items()
                              if not np.isnan(v.get('Correlation', np.nan))}
        if valid_corr_results:
            best_corr_model = max(valid_corr_results.keys(),
                                  key=lambda x: valid_corr_results[x]['Correlation'])
            print(f"  最高相关系数: {best_corr_model} ({valid_corr_results[best_corr_model]['Correlation']:.4f})")

        # 最低MAPE
        valid_mape_results = {k: v for k, v in self.results.items()
                              if not np.isnan(v.get('MAPE', np.nan))}
        if valid_mape_results:
            best_mape_model = min(valid_mape_results.keys(),
                                  key=lambda x: valid_mape_results[x]['MAPE'])
            print(f"  最低MAPE: {best_mape_model} ({valid_mape_results[best_mape_model]['MAPE']:.2f}%)")

    # 打印详细统计信息
    print(f"\n详细统计:")
    for model_name, result in self.results.items():
        missing_preds = result.get('missing_predictions', 0)
        valid_preds = result.get('valid_predictions', 0)
        print(f"  {model_name}: 缺失值预测 {missing_preds}, 有效预测 {valid_preds}")


if __name__ == "__main__":
    main()
