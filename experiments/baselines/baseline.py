import numpy as np
import pandas as pd
from pypots.imputation import SAITS, BRITS, iTransformer

# 尝试导入额外的模型
try:
    from pypots.imputation import Transformer

    TRANSFORMER_AVAILABLE = True
except ImportError:
    print("Transformer模型不可用，将跳过")
    TRANSFORMER_AVAILABLE = False

try:
    from pypots.imputation import ImputeFormer

    IMPUTEFORMER_AVAILABLE = True
except ImportError:
    print("ImputeFormer模型不可用，将跳过")
    IMPUTEFORMER_AVAILABLE = False

try:
    from pypots.imputation import MRNN

    MRNN_AVAILABLE = True
except ImportError:
    try:
        from pypots.imputation import RITS  # 有时SeqRNN实现为RITS

        MRNN_AVAILABLE = True
        MRNN = RITS
    except ImportError:
        print("MRNN/SeqRNN模型不可用，将跳过")
        MRNN_AVAILABLE = False

try:
    from pypots.nn.functional import calc_mae, calc_mse, calc_rmse
except ImportError:
    from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import warnings
from spectraformer.config import DATA_DIR

warnings.filterwarnings('ignore')


class SparseColumnImputation:
    def __init__(self, missing_data_file_path, original_data_file_path, time_col=None):
        """
        针对少数列缺失值的时间序列插补器
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
        self.missing_columns = []
        self.complete_columns = []

    def load_and_analyze_data(self):
        """
        加载数据并分析缺失模式
        """
        try:
            print("=" * 60)
            print("加载和分析数据")
            print("=" * 60)

            # 加载数据
            if self.original_data_file_path.endswith('.csv'):
                df_original = pd.read_csv(self.original_data_file_path)
            else:
                df_original = pd.read_excel(self.original_data_file_path)

            if self.missing_data_file_path.endswith('.csv'):
                df_missing = pd.read_csv(self.missing_data_file_path)
            else:
                df_missing = pd.read_excel(self.missing_data_file_path)

            print(f"原始数据形状: {df_original.shape}")
            print(f"缺失数据形状: {df_missing.shape}")

            # 处理时间列
            if self.time_col and self.time_col in df_original.columns:
                df_original = df_original.drop(self.time_col, axis=1)
            if self.time_col and self.time_col in df_missing.columns:
                df_missing = df_missing.drop(self.time_col, axis=1)

            # 只保留数值列
            numeric_cols_original = df_original.select_dtypes(include=[np.number]).columns
            numeric_cols_missing = df_missing.select_dtypes(include=[np.number]).columns
            common_cols = list(set(numeric_cols_original) & set(numeric_cols_missing))

            df_original = df_original[common_cols]
            df_missing = df_missing[common_cols]

            # 确保数据形状一致
            min_rows = min(df_original.shape[0], df_missing.shape[0])
            df_original = df_original.iloc[:min_rows].reset_index(drop=True)
            df_missing = df_missing.iloc[:min_rows].reset_index(drop=True)

            print(f"处理后形状: {df_original.shape}")

            # 分析缺失模式
            print(f"\n缺失值分析:")
            missing_info = df_missing.isnull().sum()
            missing_rates = (missing_info / len(df_missing) * 100).round(2)

            self.missing_columns = []
            self.complete_columns = []

            for col in common_cols:
                missing_count = missing_info[col]
                missing_rate = missing_rates[col]

                if missing_count > 0:
                    self.missing_columns.append(col)
                    print(f"  {col}: {missing_count} 个缺失值 ({missing_rate}%)")
                else:
                    self.complete_columns.append(col)

            print(f"\n缺失值列数: {len(self.missing_columns)}")
            print(f"完整列数: {len(self.complete_columns)}")
            print(f"缺失值列: {self.missing_columns}")

            if len(self.missing_columns) == 0:
                print("错误: 数据中没有缺失值!")
                return False

            if len(self.missing_columns) > 10:
                print("警告: 缺失值列数较多，建议检查数据质量")

            # 数据一致性检查
            print(f"\n数据一致性检查:")
            consistency_ok = True
            for col in common_cols:
                if col not in self.missing_columns:  # 完整列应该完全一致
                    if not df_original[col].equals(df_missing[col]):
                        print(f"  ✗ 列 {col} 数据不一致!")
                        consistency_ok = False
                else:  # 缺失值列在非缺失位置应该一致
                    non_missing_mask = df_missing[col].notna()
                    if non_missing_mask.sum() > 0:
                        orig_vals = df_original.loc[non_missing_mask, col]
                        miss_vals = df_missing.loc[non_missing_mask, col]
                        if not np.allclose(orig_vals, miss_vals, equal_nan=True, rtol=1e-10):
                            print(f"  ✗ 列 {col} 在非缺失位置数据不一致!")
                            consistency_ok = False

            if consistency_ok:
                print("  ✓ 数据一致性检查通过")
            else:
                print("  ⚠️ 发现数据不一致，可能导致评估问题")

            # 存储数据
            self.original_data = df_original
            self.data_with_missing = df_missing

            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_data_for_models(self, train_ratio=0.8):
        """
        为深度学习模型准备3D数据格式
        """
        # 随机划分训练测试集
        n_samples = len(self.data_with_missing)
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        train_size = int(train_ratio * n_samples)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # 创建滑动窗口
        window_size = min(24, max(12, n_samples // 100))  # 自适应窗口大小

        def create_windows(data_df, indices):
            """创建滑动窗口数据"""
            windows = []
            valid_indices = []

            for idx in indices:
                # 确保有足够的历史数据
                start_idx = max(0, idx - window_size + 1)
                end_idx = idx + 1

                if end_idx - start_idx == window_size:
                    window_data = data_df.iloc[start_idx:end_idx].values
                    windows.append(window_data)
                    valid_indices.append(idx)

            if len(windows) == 0:
                # 如果窗口策略失败，使用简单的时间序列分割
                data_array = data_df.values
                n_steps = min(window_size, len(data_array))
                n_features = data_array.shape[1]

                # 单个序列
                windows = [data_array[:n_steps].reshape(1, n_steps, n_features)]
                valid_indices = [0]
            else:
                windows = np.array(windows)

            return windows, valid_indices

        # 创建训练和测试数据
        train_windows, train_valid_indices = create_windows(self.data_with_missing, train_indices)
        test_windows, test_valid_indices = create_windows(self.data_with_missing, test_indices)

        # 对应的原始数据
        train_original_windows, _ = create_windows(self.original_data, train_valid_indices)
        test_original_windows, _ = create_windows(self.original_data, test_valid_indices)

        print(f"\n3D数据准备:")
        print(f"  窗口大小: {window_size}")
        print(f"  训练窗口数: {len(train_windows)}")
        print(f"  测试窗口数: {len(test_windows)}")

        if len(train_windows) > 0:
            print(f"  每个窗口形状: {train_windows[0].shape}")

        return (np.array(train_windows), np.array(test_windows),
                np.array(train_original_windows), np.array(test_original_windows))

    def train_deep_models(self, train_data_3d):
        """
        训练深度学习模型 (SAITS, BRITS, iTransformer)
        """
        print(f"\n" + "=" * 60)
        print("训练深度学习模型")
        print("=" * 60)

        if len(train_data_3d) == 0:
            print("训练数据为空，跳过深度学习模型")
            return

        # 数据格式检查
        if len(train_data_3d.shape) == 3:
            n_samples, n_steps, n_features = train_data_3d.shape
        else:
            print("数据格式不正确，跳过深度学习模型")
            return

        print(f"训练数据形状: {train_data_3d.shape}")
        print(f"缺失率: {np.isnan(train_data_3d).mean():.2%}")

        # ===== 关键修复：数据预处理和稳定性检查 =====
        print(f"\n数据质量检查:")

        # 检查数据范围
        valid_data = train_data_3d[~np.isnan(train_data_3d)]
        if len(valid_data) > 0:
            data_min, data_max = valid_data.min(), valid_data.max()
            data_std = valid_data.std()
            print(f"  数据范围: [{data_min:.6f}, {data_max:.6f}]")
            print(f"  数据标准差: {data_std:.6f}")

            # 检查是否需要标准化
            if data_std > 10 or data_max > 100:
                print("  ⚠️ 数据范围较大，可能需要标准化")

            # 检查异常值
            if np.any(np.isinf(valid_data)) or np.any(np.abs(valid_data) > 1e6):
                print("  ⚠️ 发现异常值(inf或极大值)，可能导致训练不稳定")

        # 数据标准化处理（可选，但推荐）
        train_data_processed = train_data_3d.copy()

        # 对每个特征进行标准化
        for feat_idx in range(n_features):
            feat_data = train_data_processed[:, :, feat_idx]
            valid_mask = ~np.isnan(feat_data)

            if valid_mask.sum() > 1:
                feat_mean = feat_data[valid_mask].mean()
                feat_std = feat_data[valid_mask].std()

                if feat_std > 1e-8:  # 避免除零
                    train_data_processed[:, :, feat_idx] = (feat_data - feat_mean) / feat_std
                    print(f"  特征 {feat_idx}: 标准化 (均值={feat_mean:.4f}, 标准差={feat_std:.4f})")

        train_dict = {'X': train_data_processed.astype(np.float32)}

        # 扩展的模型配置，包含新增模型和稳定性改进
        models_config = {
            'SAITS': {
                'class': SAITS,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 1,  # 减少层数提高稳定性
                    'd_model': 32,
                    'd_ffn': 16,
                    'n_heads': 2,
                    'd_k': 16, 'd_v': 16,
                    'dropout': 0.1,  # 减少dropout
                    'epochs': 3,  # 减少训练轮数
                    'patience': 2,
                    'batch_size': min(4, max(1, n_samples // 2))  # 更小的批次大小
                }
            },
            'BRITS': {
                'class': BRITS,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'rnn_hidden_size': 8,  # 大幅减小隐藏层大小
                    'epochs': 3,
                    'patience': 2,
                    'batch_size': min(4, max(1, n_samples // 2))
                }
            },
            'iTransformer': {
                'class': iTransformer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 1,
                    'd_model': 8,  # 大幅减小模型大小
                    'n_heads': 1,  # 减少注意力头
                    'd_k': 8, 'd_v': 8,
                    'd_ffn': 16,
                    'dropout': 0.1,
                    'epochs': 3,
                    'patience': 2,
                    'batch_size': min(4, max(1, n_samples // 2))
                }
            }
        }

        # 添加额外的模型（如果可用）- 使用更保守的配置
        if TRANSFORMER_AVAILABLE:
            models_config['Transformer'] = {
                'class': Transformer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 1,
                    'd_model': 16,  # 减小模型大小
                    'd_ffn': 32,
                    'n_heads': 2,
                    'd_k': 8,
                    'd_v': 8,
                    'dropout': 0.1,
                    'epochs': 3,
                    'patience': 2,
                    'batch_size': min(4, max(1, n_samples // 2))
                }
            }

        if IMPUTEFORMER_AVAILABLE:
            models_config['ImputeFormer'] = {
                'class': ImputeFormer,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'n_layers': 1,
                    'd_model': 16,
                    'n_heads': 2,
                    'd_k': 8,
                    'd_v': 8,
                    'd_ffn': 32,
                    'dropout': 0.1,
                    'epochs': 3,
                    'patience': 2,
                    'batch_size': min(4, max(1, n_samples // 2))
                }
            }

        if MRNN_AVAILABLE:
            models_config['SeqRNN'] = {
                'class': MRNN,
                'params': {
                    'n_steps': n_steps,
                    'n_features': n_features,
                    'rnn_hidden_size': 8,  # 减小隐藏层
                    'epochs': 3,
                    'patience': 2,
                    'batch_size': min(4, max(1, n_samples // 2))
                }
            }

        print(f"可用模型: {list(models_config.keys())}")

        for model_name, config in models_config.items():
            print(f"\n训练 {model_name}...")
            try:
                # 尝试创建模型，如果参数不兼容则逐步简化
                model = self._create_model_safely(config['class'], config['params'])
                if model is not None:
                    print(f"  开始训练 {model_name}...")

                    # 添加训练监控
                    import time
                    start_time = time.time()

                    try:
                        model.fit(train_dict)
                        training_time = time.time() - start_time

                        self.models[model_name] = model
                        print(f"✓ {model_name} 训练完成 (耗时: {training_time:.1f}秒)")

                    except KeyboardInterrupt:
                        print(f"⚠️ {model_name} 训练被用户中断")
                        self.models[model_name] = None
                        continue

                    except Exception as train_error:
                        print(f"✗ {model_name} 训练过程中失败: {train_error}")

                        # 尝试更简单的配置
                        print(f"  尝试 {model_name} 的最简配置...")
                        minimal_config = self._get_minimal_config(config['class'], n_steps, n_features, n_samples)

                        if minimal_config:
                            try:
                                simple_model = config['class'](**minimal_config)
                                simple_model.fit(train_dict)
                                self.models[model_name] = simple_model
                                print(f"✓ {model_name} 简单配置训练完成")
                            except Exception as simple_error:
                                print(f"✗ {model_name} 简单配置也失败: {simple_error}")
                                self.models[model_name] = None
                        else:
                            self.models[model_name] = None
                else:
                    self.models[model_name] = None
                    print(f"✗ {model_name} 创建失败")
            except Exception as e:
                print(f"✗ {model_name} 初始化失败: {e}")
                self.models[model_name] = None

    def _get_minimal_config(self, model_class, n_steps, n_features, n_samples):
        """
        获取最简配置以提高稳定性
        """
        model_name = str(model_class.__name__).lower()

        base_config = {
            'n_steps': n_steps,
            'n_features': n_features,
            'epochs': 1,  # 最少训练轮数
            'batch_size': 1 if n_samples == 1 else 2
        }

        if 'saits' in model_name:
            base_config.update({
                'n_layers': 1,
                'd_model': 8,
                'n_heads': 1,
                'd_k': 8, 'd_v': 8,
                'd_ffn': 8
            })
        elif 'brits' in model_name:
            base_config.update({
                'rnn_hidden_size': 4
            })
        elif any(x in model_name for x in ['transformer', 'itransformer']):
            base_config.update({
                'n_layers': 1,
                'd_model': 4,
                'n_heads': 1,
                'd_k': 4, 'd_v': 4,
                'd_ffn': 8
            })
        elif 'imputeformer' in model_name:
            base_config.update({
                'n_layers': 1,
                'd_model': 4,
                'n_heads': 1,
                'd_k': 4, 'd_v': 4,
                'd_ffn': 8
            })
        elif any(x in model_name for x in ['mrnn', 'seqrnn', 'rits']):
            base_config.update({
                'rnn_hidden_size': 4
            })
        else:
            return None

        return base_config

    def _create_model_safely(self, model_class, params):
        """
        安全地创建模型，处理API版本差异
        """
        import inspect

        # 获取模型类的初始化参数
        try:
            sig = inspect.signature(model_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}

            # 只使用有效的参数
            filtered_params = {k: v for k, v in params.items() if k in valid_params}

            print(f"    使用参数: {list(filtered_params.keys())}")
            return model_class(**filtered_params)

        except Exception as e:
            print(f"    参数检查失败: {e}")

            # 回退到基本参数
            basic_params = {
                'n_steps': params['n_steps'],
                'n_features': params['n_features'],
                'epochs': 3,
                'batch_size': params.get('batch_size', 8)
            }

            # 根据模型类型添加特定参数
            model_name = str(model_class.__name__).lower()

            if 'saits' in model_name:
                basic_params.update({
                    'n_layers': 1,
                    'd_model': 32,
                    'n_heads': 2
                })
            elif 'brits' in model_name or 'rits' in model_name:
                basic_params.update({
                    'rnn_hidden_size': 16
                })
            elif 'transformer' in model_name:
                basic_params.update({
                    'n_layers': 1,
                    'd_model': 32,
                    'n_heads': 2,
                    'd_k': 16,
                    'd_v': 16
                })
            elif 'imputeformer' in model_name:
                basic_params.update({
                    'n_layers': 1,
                    'd_model': 32,
                    'n_heads': 2,
                    'd_k': 16,
                    'd_v': 16,
                    'd_ffn': 64
                })
            elif 'mrnn' in model_name or 'seqrnn' in model_name:
                basic_params.update({
                    'rnn_hidden_size': 16
                })
            elif 'itransformer' in model_name:
                basic_params.update({
                    'n_layers': 1,
                    'd_model': 16,
                    'n_heads': 2
                })

            try:
                print(f"    使用基本参数: {list(basic_params.keys())}")
                return model_class(**basic_params)
            except Exception as e2:
                print(f"    基本参数也失败: {e2}")

                # 最后的尝试：只使用最基本的参数
                minimal_params = {
                    'n_steps': params['n_steps'],
                    'n_features': params['n_features']
                }

                try:
                    print(f"    尝试最小参数: {list(minimal_params.keys())}")
                    return model_class(**minimal_params)
                except Exception as e3:
                    print(f"    最小参数也失败: {e3}")
                    return None

    def train_simple_imputers(self):
        """
        训练简单插补方法作为基线
        """
        print(f"\n训练简单插补方法...")

        # 只对缺失值列进行处理
        missing_data = self.data_with_missing[self.missing_columns].values

        # 均值插补
        mean_imputer = SimpleImputer(strategy='mean')
        self.models['Mean'] = mean_imputer.fit(missing_data)

        # KNN插补
        if len(self.missing_columns) <= 5:  # KNN对高维数据可能很慢
            knn_imputer = KNNImputer(n_neighbors=min(5, len(self.data_with_missing) // 10))
            self.models['KNN'] = knn_imputer.fit(missing_data)

        print("✓ 简单插补方法训练完成")

    def evaluate_all_models(self, test_data_3d, test_original_3d):
        """
        评估所有模型
        """
        print(f"\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)

        # 评估深度学习模型
        if len(test_data_3d) > 0:
            self.evaluate_deep_models(test_data_3d, test_original_3d)

        # 评估简单插补方法
        self.evaluate_simple_models()

    def evaluate_deep_models(self, test_data_3d, test_original_3d):
        """
        评估深度学习模型
        """
        print(f"\n评估深度学习模型:")

        model_names = ['SAITS', 'BRITS', 'iTransformer', 'Transformer', 'ImputeFormer', 'SeqRNN']

        for model_name in model_names:
            model = self.models.get(model_name)
            if model is None:
                continue

            print(f"\n评估 {model_name}...")
            try:
                # 插补
                test_dict = {'X': test_data_3d.astype(np.float32)}
                imputed_result = model.impute(test_dict)

                if isinstance(imputed_result, dict) and 'X' in imputed_result:
                    imputed_data_3d = imputed_result['X']
                else:
                    imputed_data_3d = imputed_result

                # 计算指标 - 只在缺失值列的缺失位置
                self.calculate_metrics_3d(model_name, test_data_3d, test_original_3d, imputed_data_3d)

            except Exception as e:
                print(f"✗ {model_name} 评估失败: {e}")
                import traceback
                traceback.print_exc()

    def evaluate_simple_models(self):
        """
        评估简单插补方法
        """
        print(f"\n评估简单插补方法:")

        # 准备测试数据
        test_missing_data = self.data_with_missing[self.missing_columns]
        test_original_data = self.original_data[self.missing_columns]

        for model_name in ['Mean', 'KNN']:
            model = self.models.get(model_name)
            if model is None:
                continue

            print(f"\n评估 {model_name}...")
            try:
                # 插补
                imputed_data = model.transform(test_missing_data.values)
                imputed_df = pd.DataFrame(imputed_data, columns=self.missing_columns)

                # 计算指标
                self.calculate_metrics_2d(model_name, test_missing_data, test_original_data, imputed_df)

            except Exception as e:
                print(f"✗ {model_name} 评估失败: {e}")

    def calculate_metrics_3d(self, model_name, test_data_3d, test_original_3d, imputed_data_3d):
        """
        计算3D数据的指标
        """
        # 找出缺失值列的索引
        missing_col_indices = [self.original_data.columns.get_loc(col) for col in self.missing_columns]

        all_true_values = []
        all_imputed_values = []

        # 遍历所有样本和时间步
        for sample_idx in range(test_data_3d.shape[0]):
            for time_idx in range(test_data_3d.shape[1]):
                for col_idx in missing_col_indices:
                    # 只计算真正缺失的位置
                    if np.isnan(test_data_3d[sample_idx, time_idx, col_idx]):
                        true_val = test_original_3d[sample_idx, time_idx, col_idx]
                        imputed_val = imputed_data_3d[sample_idx, time_idx, col_idx]

                        if not (np.isnan(true_val) or np.isnan(imputed_val)):
                            all_true_values.append(true_val)
                            all_imputed_values.append(imputed_val)

        if len(all_true_values) == 0:
            print(f"  ✗ {model_name}: 没有有效的预测值")
            return

        true_values = np.array(all_true_values)
        imputed_values = np.array(all_imputed_values)

        # 计算指标
        metrics = self.calculate_metrics(true_values, imputed_values)
        metrics['valid_predictions'] = len(true_values)

        self.results[model_name] = metrics
        self.print_model_metrics(model_name, metrics)

    def calculate_metrics_2d(self, model_name, test_missing_data, test_original_data, imputed_df):
        """
        计算2D数据的指标
        """
        all_true_values = []
        all_imputed_values = []

        for col in self.missing_columns:
            missing_mask = test_missing_data[col].isna()
            true_vals = test_original_data.loc[missing_mask, col]
            imputed_vals = imputed_df.loc[missing_mask, col]

            # 移除无效值
            valid_mask = ~(np.isnan(true_vals) | np.isnan(imputed_vals))

            all_true_values.extend(true_vals[valid_mask])
            all_imputed_values.extend(imputed_vals[valid_mask])

        if len(all_true_values) == 0:
            print(f"  ✗ {model_name}: 没有有效的预测值")
            return

        true_values = np.array(all_true_values)
        imputed_values = np.array(all_imputed_values)

        # 计算指标
        metrics = self.calculate_metrics(true_values, imputed_values)
        metrics['valid_predictions'] = len(true_values)

        self.results[model_name] = metrics
        self.print_model_metrics(model_name, metrics)

    def calculate_metrics(self, true_values, imputed_values):
        """
        计算评估指标
        """
        mae = mean_absolute_error(true_values, imputed_values)
        mse = mean_squared_error(true_values, imputed_values)
        rmse = np.sqrt(mse)

        try:
            r2 = r2_score(true_values, imputed_values)
        except:
            r2 = np.nan

        try:
            correlation = np.corrcoef(true_values, imputed_values)[0, 1]
        except:
            correlation = np.nan

        # MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((true_values - imputed_values) /
                                  (np.abs(true_values) + 1e-8))) * 100

        # 检查是否存在异常的完美预测
        perfect_predictions = np.sum(np.abs(true_values - imputed_values) < 1e-10)
        perfect_ratio = perfect_predictions / len(true_values)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Correlation': correlation,
            'perfect_ratio': perfect_ratio
        }

    def print_model_metrics(self, model_name, metrics):
        """
        打印单个模型的指标
        """
        print(f"  ✓ {model_name}:")
        print(f"    有效预测数: {metrics['valid_predictions']}")
        print(f"    MAE: {metrics['MAE']:.6f}")
        print(f"    RMSE: {metrics['RMSE']:.6f}")
        print(f"    R²: {metrics['R²']:.6f}")
        print(f"    相关系数: {metrics['Correlation']:.6f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")

        if metrics['perfect_ratio'] > 0.5:
            print(f"    ⚠️ 完美预测比例: {metrics['perfect_ratio']:.1%} (可能存在数据泄漏)")

    def print_summary(self):
        """
        打印结果摘要
        """
        if not self.results:
            print("没有可用的结果")
            return

        print(f"\n" + "=" * 80)
        print("模型性能摘要")
        print("=" * 80)

        print(f"{'模型':<15}{'MAE':<12}{'RMSE':<12}{'R²':<12}{'相关系数':<12}{'MAPE':<10}")
        print("-" * 75)

        # 按RMSE排序
        sorted_results = sorted(self.results.items(),
                                key=lambda x: x[1]['RMSE'] if not np.isnan(x[1]['RMSE']) else float('inf'))

        for model_name, metrics in sorted_results:
            mae = metrics['MAE']
            rmse = metrics['RMSE']
            r2 = metrics['R²']
            corr = metrics['Correlation']
            mape = metrics['MAPE']

            print(f"{model_name:<15}{mae:<12.6f}{rmse:<12.6f}{r2:<12.6f}{corr:<12.6f}{mape:<10.2f}%")

        # 最佳模型
        if sorted_results:
            best_model = sorted_results[0][0]
            best_rmse = sorted_results[0][1]['RMSE']
            print(f"\n最佳模型: {best_model} (RMSE: {best_rmse:.6f})")

    def plot_results_by_column(self):
        """
        按列绘制插补结果
        """
        if not self.results or not self.missing_columns:
            print("没有可用的结果进行绘图")
            return

        for col in self.missing_columns[:2]:  # 只绘制前两个缺失列
            plt.figure(figsize=(15, 8))

            # 获取列数据
            original_col = self.original_data[col]
            missing_col = self.data_with_missing[col]

            # 时间索引
            time_index = range(len(original_col))

            # 绘制原始数据
            plt.plot(time_index, original_col, 'k-', label='真实数据', linewidth=2, alpha=0.8)

            # 标记观测值
            observed_mask = missing_col.notna()
            plt.scatter(np.array(time_index)[observed_mask],
                        missing_col[observed_mask],
                        color='blue', s=20, label='观测值', alpha=0.6)

            # 标记缺失值位置
            missing_mask = missing_col.isna()
            if missing_mask.sum() > 0:
                plt.scatter(np.array(time_index)[missing_mask],
                            original_col[missing_mask],
                            color='red', s=20, marker='x', label='缺失位置真实值', alpha=0.8)

            # 绘制简单插补结果
            for model_name in ['Mean', 'KNN']:
                if model_name in self.models and self.models[model_name] is not None:
                    try:
                        imputed_data = self.models[model_name].transform(
                            self.data_with_missing[self.missing_columns].values)
                        col_idx = self.missing_columns.index(col)
                        imputed_col = imputed_data[:, col_idx]

                        plt.plot(time_index, imputed_col, '--',
                                 label=f'{model_name} 插补', linewidth=1, alpha=0.7)
                    except:
                        pass

            plt.xlabel('时间索引')
            plt.ylabel('数值')
            plt.title(f'列 {col} 的插补结果对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


def main():
    """
    主函数
    """
    # ====== 配置文件路径 ======
    MISSING_DATA_FILE_PATH = str(DATA_DIR / "train_continuous_missing_60.csv")
    ORIGINAL_DATA_FILE_PATH = str(DATA_DIR / "train_swat.csv")
    TIME_COLUMN = "Unnamed: 0"  # 时间列名，如果没有则设为None

    print("稀疏缺失值时间序列插补")
    print("=" * 60)
    print(f"缺失数据文件: {MISSING_DATA_FILE_PATH}")
    print(f"原始数据文件: {ORIGINAL_DATA_FILE_PATH}")

    # 初始化插补器
    imputer = SparseColumnImputation(MISSING_DATA_FILE_PATH, ORIGINAL_DATA_FILE_PATH, TIME_COLUMN)

    # 加载和分析数据
    if not imputer.load_and_analyze_data():
        print("数据加载失败，程序退出")
        return

    # 训练简单插补方法
    imputer.train_simple_imputers()

    # 为深度学习模型准备数据
    try:
        train_data_3d, test_data_3d, train_original_3d, test_original_3d = imputer.prepare_data_for_models()

        # 训练深度学习模型
        if len(train_data_3d) > 0:
            imputer.train_deep_models(train_data_3d)
        else:
            print("无法创建3D数据，跳过深度学习模型")
            test_data_3d = []
            test_original_3d = []
    except Exception as e:
        print(f"深度学习模型数据准备失败: {e}")
        test_data_3d = []
        test_original_3d = []

    # 评估所有模型
    imputer.evaluate_all_models(test_data_3d, test_original_3d)

    # 显示结果摘要
    imputer.print_summary()

    # 绘制结果
    try:
        imputer.plot_results_by_column()
    except Exception as e:
        print(f"绘图失败: {e}")

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
