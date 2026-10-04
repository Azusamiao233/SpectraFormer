import numpy as np
import pandas as pd
from pypots.imputation import SAITS, BRITS, iTransformer, Transformer, SegRNN

# 尝试导入其他可用模型
try:
    from pypots.imputation import ImputeFormer

    IMPUTEFORMER_AVAILABLE = True
except ImportError:
    IMPUTEFORMER_AVAILABLE = False
    print("注意: ImputeFormer在当前PyPOTS版本中不可用")

try:
    from pypots.imputation import Autoformer

    AUTOFORMER_AVAILABLE = True
except ImportError:
    AUTOFORMER_AVAILABLE = False

try:
    from pypots.imputation import Informer

    INFORMER_AVAILABLE = True
except ImportError:
    INFORMER_AVAILABLE = False

try:
    from pypots.nn.functional import calc_mae, calc_mse, calc_rmse
except ImportError:
    from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from mssa_transformer.config import DATA_DIR

warnings.filterwarnings('ignore')


class ExtendedTimeSeriesImputation:
    def __init__(self, missing_data_file_path, original_data_file_path, time_col=None):
        """
        初始化扩展插补器，包含更多先进模型
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

    def prepare_data(self, train_ratio=0.8, ensure_missing_in_test=True):
        """
        准备训练和测试数据 - 修正版本，确保测试集包含缺失值
        Args:
            train_ratio: 训练集比例
            ensure_missing_in_test: 是否确保测试集包含缺失值
        """
        n_samples = self.data_with_missing.shape[0]

        if ensure_missing_in_test:
            print("🔧 启用智能数据划分，确保测试集包含缺失值...")

            # 找到包含缺失值的样本
            samples_with_missing = []
            samples_without_missing = []

            for i in range(n_samples):
                if np.isnan(self.data_with_missing[i]).any():
                    samples_with_missing.append(i)
                else:
                    samples_without_missing.append(i)

            print(f"  有缺失值的样本数: {len(samples_with_missing)}")
            print(f"  无缺失值的样本数: {len(samples_without_missing)}")

            if len(samples_with_missing) == 0:
                print("⚠️ 警告: 没有发现缺失值样本，使用常规划分")
                train_size = int(train_ratio * n_samples)
                train_indices = list(range(train_size))
                test_indices = list(range(train_size, n_samples))
            else:
                # 确保测试集至少包含30%的缺失值样本
                missing_for_test = max(1, int(len(samples_with_missing) * 0.3))
                missing_for_train = len(samples_with_missing) - missing_for_test

                # 随机选择缺失值样本
                np.random.seed(42)  # 为了可重复性
                np.random.shuffle(samples_with_missing)

                train_missing = samples_with_missing[:missing_for_train]
                test_missing = samples_with_missing[missing_for_train:]

                # 分配无缺失值的样本
                remaining_train_slots = int(train_ratio * n_samples) - len(train_missing)
                remaining_train_slots = max(0, remaining_train_slots)

                np.random.shuffle(samples_without_missing)
                train_complete = samples_without_missing[:remaining_train_slots]
                test_complete = samples_without_missing[remaining_train_slots:]

                train_indices = train_missing + train_complete
                test_indices = test_missing + test_complete

                print(f"  训练集缺失值样本: {len(train_missing)}")
                print(f"  测试集缺失值样本: {len(test_missing)}")
                print(f"  训练集完整样本: {len(train_complete)}")
                print(f"  测试集完整样本: {len(test_complete)}")
        else:
            # 传统的顺序划分
            train_size = int(train_ratio * n_samples)
            train_indices = list(range(train_size))
            test_indices = list(range(train_size, n_samples))

        # 提取数据
        train_data = self.data_with_missing[train_indices]
        test_data = self.data_with_missing[test_indices]
        test_original = self.original_data[test_indices]
        test_missing_mask = self.missing_mask[test_indices]

        # 添加详细的数据诊断
        print(f"\n🔍 数据诊断:")
        print(f"  训练集缺失率: {np.isnan(train_data).mean():.4f}")
        print(f"  测试集缺失率: {np.isnan(test_data).mean():.4f}")
        print(f"  测试集缺失值数量: {np.isnan(test_data).sum()}")
        print(f"  测试集总元素数: {test_data.size}")
        print(f"  missing_mask中缺失值数量: {test_missing_mask.sum()}")

        # 检查数据一致性
        computed_missing_mask = np.isnan(test_data)
        mask_consistency = np.array_equal(test_missing_mask, computed_missing_mask)
        print(f"  缺失值掩码一致性: {'✓' if mask_consistency else '✗'}")

        if not mask_consistency:
            print(f"  实际缺失值: {computed_missing_mask.sum()}")
            print(f"  掩码标记缺失值: {test_missing_mask.sum()}")

        # 检查原始数据与缺失数据的差异
        non_missing_positions = ~test_missing_mask
        if non_missing_positions.sum() > 0:
            original_values = test_original[non_missing_positions]
            missing_values = test_data[non_missing_positions]
            values_match = np.allclose(original_values, missing_values, equal_nan=True)
            print(f"  非缺失位置数值一致性: {'✓' if values_match else '✗'}")

        # ✨ 关键修正：如果测试集仍然没有缺失值，强制创建一些
        if np.isnan(test_data).sum() == 0 and ensure_missing_in_test:
            print("🚨 测试集仍无缺失值，强制创建缺失值用于测试...")

            # 在测试集中随机创建一些缺失值
            test_data_copy = test_data.copy()
            n_missing = max(100, int(test_data.size * 0.05))  # 创建5%的缺失值

            # 随机选择位置
            flat_indices = np.random.choice(test_data.size, n_missing, replace=False)
            flat_test = test_data_copy.flatten()
            flat_test[flat_indices] = np.nan
            test_data = flat_test.reshape(test_data.shape)

            # 更新缺失值掩码
            test_missing_mask = np.isnan(test_data)

            print(f"  强制创建缺失值数量: {test_missing_mask.sum()}")
            print(f"  新的测试集缺失率: {test_missing_mask.mean():.4f}")

        return train_data, test_data, test_original, test_missing_mask

    def train_models(self, train_data):
        """
        训练所有模型，包括新增的Transformer、ImputeFormer和SegRNN
        """
        print("\n开始训练扩展模型集合...")

        n_steps, n_features = train_data.shape[1], train_data.shape[2]
        print(f"模型输入维度: n_steps={n_steps}, n_features={n_features}")

        # 将数据转换为PyPOTS期望的字典格式
        train_dict = {
            'X': train_data
        }

        # ================== 原有模型 ==================

        # SAITS模型
        print("\n训练SAITS模型...")
        try:
            self.models['SAITS'] = SAITS(
                n_steps=n_steps,
                n_features=n_features,
                n_layers=2,
                d_model=128,
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
            self.models['iTransformer'] = None

        # ================== 新增模型 ==================

        # Transformer模型
        print("\n训练Transformer模型...")
        try:
            self.models['Transformer'] = Transformer(
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
            self.models['Transformer'].fit(train_dict)
            print("✓ Transformer训练完成")
        except Exception as e:
            print(f"✗ Transformer训练失败: {e}")
            print(f"尝试更简单的Transformer配置...")
            try:
                self.models['Transformer'] = Transformer(
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
                self.models['Transformer'].fit(train_dict)
                print("✓ Transformer简单配置训练完成")
            except Exception as e2:
                print(f"✗ Transformer简单配置也失败: {e2}")
                self.models['Transformer'] = None

        # ImputeFormer模型（如果可用）
        if IMPUTEFORMER_AVAILABLE:
            print("\n训练ImputeFormer模型...")
            try:
                self.models['ImputeFormer'] = ImputeFormer(
                    n_steps=n_steps,
                    n_features=n_features,
                    n_layers=2,
                    d_input_embed=64,  # 输入嵌入维度
                    d_learnable_embed=32,  # 可学习节点嵌入维度
                    d_proj=64,  # 可学习投影器维度
                    d_ffn=128,  # FFN层维度
                    n_temporal_heads=4,  # 时间注意力头数
                    dropout=0.1,
                    input_dim=1,  # 输入特征维度
                    output_dim=1,  # 输出特征维度
                    epochs=5,
                    patience=3,
                    batch_size=16
                )
                self.models['ImputeFormer'].fit(train_dict)
                print("✓ ImputeFormer训练完成")
            except Exception as e:
                print(f"✗ ImputeFormer训练失败: {e}")
                print(f"尝试更简单的ImputeFormer配置...")
                try:
                    self.models['ImputeFormer'] = ImputeFormer(
                        n_steps=n_steps,
                        n_features=n_features,
                        n_layers=1,
                        d_input_embed=32,  # 更小的嵌入维度
                        d_learnable_embed=16,  # 更小的节点嵌入
                        d_proj=32,  # 更小的投影器
                        d_ffn=64,  # 更小的FFN
                        n_temporal_heads=2,  # 更少的注意力头
                        dropout=0.1,
                        epochs=3,
                        batch_size=8
                    )
                    self.models['ImputeFormer'].fit(train_dict)
                    print("✓ ImputeFormer简单配置训练完成")
                except Exception as e2:
                    print(f"✗ ImputeFormer简单配置也失败: {e2}")
                    self.models['ImputeFormer'] = None
        else:
            print("\n跳过ImputeFormer (模型不可用)")
            self.models['ImputeFormer'] = None

        # Autoformer模型（如果可用）
        if AUTOFORMER_AVAILABLE:
            print("\n训练Autoformer模型...")
            try:
                self.models['Autoformer'] = Autoformer(
                    n_steps=n_steps,
                    n_features=n_features,
                    n_layers=2,
                    d_model=64,
                    n_heads=4,
                    d_ffn=128,
                    factor=3,
                    moving_avg_window_size=25,
                    dropout=0.1,
                    epochs=5,
                    patience=3,
                    batch_size=16
                )
                self.models['Autoformer'].fit(train_dict)
                print("✓ Autoformer训练完成")
            except Exception as e:
                print(f"✗ Autoformer训练失败: {e}")
                self.models['Autoformer'] = None
        else:
            print("\n跳过Autoformer (模型不可用)")
            self.models['Autoformer'] = None

        # SegRNN模型
        print("\n训练SegRNN模型...")
        try:
            self.models['SegRNN'] = SegRNN(
                n_steps=n_steps,
                n_features=n_features,
                d_model=64,
                seg_len=min(12, n_steps // 2),  # 段长度，不能超过序列长度的一半
                dropout=0.1,
                epochs=5,
                patience=3,
                batch_size=16
            )
            self.models['SegRNN'].fit(train_dict)
            print("✓ SegRNN训练完成")
        except Exception as e:
            print(f"✗ SegRNN训练失败: {e}")
            print(f"尝试更简单的SegRNN配置...")
            try:
                self.models['SegRNN'] = SegRNN(
                    n_steps=n_steps,
                    n_features=n_features,
                    d_model=32,
                    seg_len=min(6, max(1, n_steps // 4)),  # 更小的段长度
                    epochs=3,
                    batch_size=8
                )
                self.models['SegRNN'].fit(train_dict)
                print("✓ SegRNN简单配置训练完成")
            except Exception as e2:
                print(f"✗ SegRNN简单配置也失败: {e2}")
                self.models['SegRNN'] = None

        # 统计成功训练的模型
        successful_models = [name for name, model in self.models.items() if model is not None]
        failed_models = [name for name, model in self.models.items() if model is None]

        print(f"\n✓ 成功训练的模型 ({len(successful_models)}): {successful_models}")
        if failed_models:
            print(f"✗ 训练失败的模型 ({len(failed_models)}): {failed_models}")

    def evaluate_models(self, test_data, test_original, test_missing_mask):
        """
        评估所有模型 - 使用真实原始数据作为参考，只在真正的缺失位置评估
        """
        print("\n" + "=" * 70)
        print("开始评估扩展模型...")
        print("=" * 70)

        # 添加评估前的数据检查
        print(f"\n📋 评估数据概况:")
        print(f"  测试数据形状: {test_data.shape}")
        print(f"  缺失值掩码缺失数量: {test_missing_mask.sum()}")
        print(f"  实际NaN数量: {np.isnan(test_data).sum()}")

        # 确保我们有缺失值要预测
        if test_missing_mask.sum() == 0:
            print("❌ 错误: 测试集中没有缺失值可以评估！")
            return

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

                # 检查插补结果
                print(f"  插补前缺失值: {np.isnan(test_data).sum()}")
                print(f"  插补后缺失值: {np.isnan(imputed_data).sum()}")

                # ✨ 关键修正：只在真正的缺失位置评估
                actual_missing_positions = test_missing_mask

                if actual_missing_positions.sum() == 0:
                    print(f"警告: {model_name} 测试集中没有缺失值可评估")
                    continue

                # 使用真实原始数据作为参考，只在缺失位置
                true_values = test_original[actual_missing_positions]
                imputed_values = imputed_data[actual_missing_positions]

                # 移除可能的nan值
                valid_mask = ~(np.isnan(true_values) | np.isnan(imputed_values))
                if valid_mask.sum() == 0:
                    print(f"警告: {model_name} 没有有效的预测值")
                    continue

                true_values = true_values[valid_mask]
                imputed_values = imputed_values[valid_mask]

                print(f"  实际评估的缺失值数量: {len(true_values)}")

                # 检查是否所有值都相同（可能的作弊检测）
                if len(np.unique(imputed_values)) == 1:
                    print(f"  警告: 插补值全部相同 ({imputed_values[0]})")
                elif np.allclose(true_values, imputed_values):
                    print(f"  警告: 插补值与真实值完全一致，可能存在数据泄漏")

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

                # 计算MedAE (Median Absolute Error)
                medae = np.median(np.abs(true_values - imputed_values))

                self.results[model_name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R²': r2,
                    'MAPE': mape,
                    'SMAPE': smape,
                    'NRMSE': nrmse,
                    'MedAE': medae,
                    'Correlation': correlation,
                    'imputed_data': imputed_data,
                    'valid_predictions': valid_mask.sum(),
                    'missing_predictions': actual_missing_positions.sum()
                }

                print(f"✓ {model_name} 评估完成:")
                print(f"    缺失值预测数量: {actual_missing_positions.sum()}")
                print(f"    有效预测数量: {valid_mask.sum()}")
                print(f"    MAE: {mae:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    R²: {r2:.4f}")
                print(f"    MAPE: {mape:.2f}%")
                print(f"    Correlation: {correlation:.4f}")

            except Exception as e:
                print(f"✗ {model_name} 评估失败: {e}")
                import traceback
                traceback.print_exc()

    def plot_comprehensive_results(self, test_data, test_original, test_missing_mask, sample_idx=0, feature_idx=0):
        """
        绘制综合结果对比图，包含所有模型 - 修正版本，突出显示缺失值插补
        """
        if not self.results:
            print("没有可用的结果进行绘图")
            return

        if sample_idx >= test_data.shape[0] or feature_idx >= test_data.shape[2]:
            print(f"索引超出范围，使用默认值 sample_idx=0, feature_idx=0")
            sample_idx, feature_idx = 0, 0

        # 创建子图
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # 获取时间步长
        time_steps = range(test_data.shape[1])

        # 数据
        data_with_missing = test_data[sample_idx, :, feature_idx]
        true_original = test_original[sample_idx, :, feature_idx]
        missing_mask_sample = test_missing_mask[sample_idx, :, feature_idx]

        # 第一个子图：所有模型对比
        ax1 = axes[0]

        # 绘制真实完整数据
        ax1.plot(time_steps, true_original, 'k-', label='真实完整数据', linewidth=3, alpha=0.8)

        # 标记观测值（非缺失值）
        observed_mask = ~missing_mask_sample
        if observed_mask.sum() > 0:
            ax1.scatter(np.array(time_steps)[observed_mask],
                        data_with_missing[observed_mask],
                        color='blue', s=40, label='观测值', zorder=5, alpha=0.7)

        # 突出标记缺失值位置的真实值
        if missing_mask_sample.sum() > 0:
            ax1.scatter(np.array(time_steps)[missing_mask_sample],
                        true_original[missing_mask_sample],
                        color='red', s=60, marker='o', label='缺失位置真实值',
                        zorder=6, edgecolors='darkred', linewidth=2)

        # 绘制各模型的插补结果，特别突出缺失位置的插补
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
        for i, (model_name, result) in enumerate(self.results.items()):
            if 'imputed_data' in result:
                imputed_values = result['imputed_data'][sample_idx, :, feature_idx]

                # 绘制完整的插补序列（虚线）
                ax1.plot(time_steps, imputed_values, '--',
                         color=colors[i % len(colors)],
                         alpha=0.5, linewidth=1)

                # 突出显示缺失位置的插补值
                if missing_mask_sample.sum() > 0:
                    ax1.scatter(np.array(time_steps)[missing_mask_sample],
                                imputed_values[missing_mask_sample],
                                color=colors[i % len(colors)], s=50, marker='s',
                                label=f'{model_name} (RMSE: {result["RMSE"]:.3f})',
                                zorder=7, alpha=0.8)

        ax1.set_xlabel('时间步')
        ax1.set_ylabel('数值')
        ax1.set_title(f'时间序列插补结果对比 - 缺失值插补评估 (样本 {sample_idx}, 特征 {feature_idx})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 添加缺失值区域的背景色
        if missing_mask_sample.sum() > 0:
            missing_positions = np.where(missing_mask_sample)[0]
            for pos in missing_positions:
                ax1.axvspan(pos - 0.4, pos + 0.4, alpha=0.1, color='red', zorder=1)

        # 第二个子图：性能指标对比
        ax2 = axes[1]

        models = list(self.results.keys())
        x_pos = np.arange(len(models))

        # 绘制RMSE条形图
        rmse_values = [self.results[model]['RMSE'] for model in models]
        bars = ax2.bar(x_pos, rmse_values, alpha=0.7, color=colors[:len(models)])

        # 在条形图上添加数值标签
        for bar, rmse in zip(bars, rmse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{rmse:.3f}', ha='center', va='bottom')

        ax2.set_xlabel('模型')
        ax2.set_ylabel('RMSE')
        ax2.set_title('模型RMSE性能对比 (仅缺失值位置)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_metrics_comparison(self):
        """
        绘制详细的指标对比图
        """
        if not self.results:
            print("没有可用的结果进行绘图")
            return

        # 创建指标对比图
        metrics_to_plot = ['MAE', 'RMSE', 'R²', 'MAPE', 'Correlation']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        models = list(self.results.keys())
        x_pos = np.arange(len(models))
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for i, metric in enumerate(metrics_to_plot):
            if i >= len(axes):
                break

            values = [self.results[model].get(metric, np.nan) for model in models]

            bars = axes[i].bar(x_pos, values, alpha=0.7, color=colors)

            # 添加数值标签
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    height = bar.get_height()
                    if metric in ['MAPE', 'SMAPE', 'NRMSE']:
                        label = f'{value:.1f}%'
                    elif metric == 'R²':
                        label = f'{value:.3f}'
                    else:
                        label = f'{value:.3f}'
                    axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                                 label, ha='center', va='bottom')

            axes[i].set_title(f'{metric} 对比 (仅缺失值位置)')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(models, rotation=45)
            axes[i].grid(True, alpha=0.3)

        # 删除多余的子图
        for i in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def print_comprehensive_summary(self):
        """
        打印综合结果摘要
        """
        if not self.results:
            print("没有可用的结果")
            return

        print("\n" + "=" * 120)
        print("扩展模型性能综合摘要 - 缺失值插补评估")
        print("=" * 120)

        # 创建结果表格
        metrics = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE', 'SMAPE', 'NRMSE', 'MedAE', 'Correlation']

        print(f"{'模型':<15}", end="")
        for metric in metrics:
            print(f"{metric:>12}", end="")
        print()
        print("-" * 123)

        for model_name, result in self.results.items():
            print(f"{model_name:<15}", end="")
            for metric in metrics:
                value = result.get(metric, np.nan)
                if metric in ['MAPE', 'SMAPE', 'NRMSE']:
                    print(f"{value:>11.2f}%", end="")
                else:
                    print(f"{value:>12.4f}", end="")
            print()

        # 找出各指标的最佳模型
        print("\n🏆 各指标最佳模型:")

        best_models = {}

        # 最低误差指标
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'NRMSE', 'MedAE']:
            valid_results = {k: v for k, v in self.results.items()
                             if not np.isnan(v.get(metric, np.nan))}
            if valid_results:
                best_model = min(valid_results.keys(),
                                 key=lambda x: valid_results[x][metric])
                best_value = valid_results[best_model][metric]
                if metric in ['MAPE', 'SMAPE', 'NRMSE']:
                    print(f"  最低{metric}: {best_model} ({best_value:.2f}%)")
                else:
                    print(f"  最低{metric}: {best_model} ({best_value:.4f})")
                best_models[metric] = best_model

        # 最高性能指标
        for metric in ['R²', 'Correlation']:
            valid_results = {k: v for k, v in self.results.items()
                             if not np.isnan(v.get(metric, np.nan))}
            if valid_results:
                best_model = max(valid_results.keys(),
                                 key=lambda x: valid_results[x][metric])
                best_value = valid_results[best_model][metric]
                print(f"  最高{metric}: {best_model} ({best_value:.4f})")
                best_models[metric] = best_model

        # 统计最佳模型出现频次
        print(f"\n📊 模型排名统计:")
        model_counts = {}
        for model in best_models.values():
            model_counts[model] = model_counts.get(model, 0) + 1

        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (model, count) in enumerate(sorted_models, 1):
            print(f"  第{i}名: {model} (在{count}个指标中表现最佳)")

        # 打印详细统计信息
        print(f"\n📈 详细统计:")
        for model_name, result in self.results.items():
            missing_preds = result.get('missing_predictions', 0)
            valid_preds = result.get('valid_predictions', 0)
            print(f"  {model_name}: 缺失值预测 {missing_preds}, 有效预测 {valid_preds}")

        # 模型复杂度和训练时间分析
        print(f"\n🔧 模型特点分析:")
        model_features = {
            'SAITS': '自注意力机制，适用于长序列',
            'BRITS': 'RNN架构，处理时序依赖性强',
            'iTransformer': '逆变换器，关注变量间关系',
            'Transformer': '经典Transformer，平衡性能与复杂度',
            'ImputeFormer': '低秩诱导Transformer，专门设计的插补模型',
            'SegRNN': '分段RNN，适合分段模式数据',
            'Autoformer': '分解架构的Transformer，适合季节性数据',
            'Informer': '稀疏注意力机制，适合长序列'
        }

        for model_name in self.results.keys():
            if model_name in model_features:
                print(f"  {model_name}: {model_features[model_name]}")

        # 推荐使用场景
        print(f"\n💡 使用建议:")
        if 'RMSE' in best_models:
            rmse_winner = best_models['RMSE']
            print(f"  • 整体精度最佳: {rmse_winner}")
        if 'Correlation' in best_models:
            corr_winner = best_models['Correlation']
            print(f"  • 趋势保持最佳: {corr_winner}")
        if 'MAPE' in best_models:
            mape_winner = best_models['MAPE']
            print(f"  • 相对误差最小: {mape_winner}")

        print(f"\n✅ 注意: 本次评估仅在真实缺失值位置进行，避免了数据泄漏问题")


def main():
    """
    主函数 - 在这里指定数据文件路径
    """
    # ====== 在这里修改数据文件路径 ======
    MISSING_DATA_FILE_PATH = str(DATA_DIR / "train_continuous_missing_60.csv")
    ORIGINAL_DATA_FILE_PATH = str(DATA_DIR / "train_swat.csv")
    TIME_COLUMN = "Unnamed: 0"  # 如果有时间列，请指定列名，例如 "timestamp" 或 "date"

    print("修正版扩展时间序列插补模型测试")
    print("包含模型: SAITS, BRITS, iTransformer, Transformer, ImputeFormer, SegRNN + 扩展模型")
    print("✨ 新增功能: 智能数据划分，确保测试集包含缺失值，避免数据泄漏")
    print("=" * 90)
    print(f"缺失数据文件: {MISSING_DATA_FILE_PATH}")
    print(f"原始数据文件: {ORIGINAL_DATA_FILE_PATH}")
    print(f"时间列: {TIME_COLUMN}")

    # 初始化扩展插补器
    imputer = ExtendedTimeSeriesImputation(MISSING_DATA_FILE_PATH, ORIGINAL_DATA_FILE_PATH, TIME_COLUMN)

    # 加载数据
    if not imputer.load_data():
        print("数据加载失败，程序退出")
        return

    # 准备数据 - 使用智能划分确保测试集包含缺失值
    train_data, test_data, test_original, test_missing_mask = imputer.prepare_data(
        train_ratio=0.8,
        ensure_missing_in_test=True  # 🔧 关键参数
    )

    print(f"\n数据划分:")
    print(f"  训练集形状: {train_data.shape}")
    print(f"  测试集形状: {test_data.shape}")
    print(f"  训练集缺失率: {np.isnan(train_data).mean():.2%}")
    print(f"  测试集缺失率: {np.isnan(test_data).mean():.2%}")
    print(f"  测试集缺失值数量: {test_missing_mask.sum()}")

    # 确认测试集有缺失值再继续
    if test_missing_mask.sum() == 0:
        print("❌ 错误: 测试集中仍然没有缺失值，无法进行有效评估！")
        print("请检查数据或尝试不同的数据划分策略。")
        return

    print(f"✅ 确认: 测试集包含 {test_missing_mask.sum()} 个缺失值，可以进行有效评估")

    # 训练所有模型
    imputer.train_models(train_data)

    # 评估所有模型
    imputer.evaluate_models(test_data, test_original, test_missing_mask)

    # 显示综合结果摘要
    imputer.print_comprehensive_summary()

    # 绘制综合结果（可选）
    try:
        print("\n绘制综合对比图...")
        imputer.plot_comprehensive_results(test_data, test_original, test_missing_mask,
                                           sample_idx=0, feature_idx=0)

        print("\n绘制指标对比图...")
        imputer.plot_metrics_comparison()
    except Exception as e:
        print(f"绘图失败: {e}")

    print("\n🎉 修正版扩展时间序列插补程序执行完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
