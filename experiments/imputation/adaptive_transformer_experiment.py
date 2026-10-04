"""
改进的SpectraFormer缺失值填补 - 使用标准对角平均
本版本修改了MSSA和Transformer的结合方式，使用标准对角平均而非特征拼接
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# 导入基础模块
from spectraformer.mssa import MSSA, setup_chinese_fonts
from spectraformer.transformer.transformer_imputation import (
    PositionalEncoding,
    TimeSeriesDataset,
    TransformerImputer,
)
from sklearn.preprocessing import StandardScaler
import torch.utils.data
from spectraformer.config import DATA_DIR, imputation_output_path

# 设置中文字体
setup_chinese_fonts()

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


class ImprovedMSSATransformerImputer:
    """
    改进的SpectraFormer缺失值填补器
    使用标准对角平均方式结合MSSA分解和Transformer预测
    """

    def __init__(self, mssa_window_length: int = 30,
                 mssa_n_components: Optional[int] = None,
                 mssa_groups: Optional[Dict] = None,
                 transformer_config: Optional[Dict] = None):
        """
        参数:
            mssa_window_length: MSSA窗口长度
            mssa_n_components: MSSA保留的分量数
            mssa_groups: MSSA分组配置，如 {'trend': [0, 1], 'seasonal': [2, 3, 4]}
            transformer_config: Transformer配置参数
        """
        self.mssa_window_length = mssa_window_length
        self.mssa_n_components = mssa_n_components
        self.mssa_groups = mssa_groups

        # 默认MSSA分组（如果未指定）
        if self.mssa_groups is None:
            self.mssa_groups = {
                'trend': [0, 1],
                'seasonal': list(range(2, min(10, mssa_n_components or 20))),
                'noise': list(range(10, mssa_n_components or 20))
            }

        # 默认Transformer配置
        self.transformer_config = {
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        }
        if transformer_config:
            self.transformer_config.update(transformer_config)

        # 初始化组件
        self.mssa_model = None
        self.group_transformers = {}  # 为每个MSSA分组创建独立的Transformer
        self.scalers = {}  # 为每个分组创建独立的标准化器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")
        print(f"MSSA分组配置: {self.mssa_groups}")

    def _prepare_mssa_data_for_fitting(self, data_with_missing: np.ndarray) -> np.ndarray:
        """
        为MSSA准备数据（需要填充缺失值）
        """
        # 使用线性插值填充缺失值
        df_temp = pd.DataFrame(data_with_missing)
        data_interpolated = df_temp.interpolate(
            method='linear',
            limit_direction='both'
        ).fillna(method='bfill').fillna(method='ffill').values

        return data_interpolated

    def fit_mssa(self, data_with_missing: np.ndarray):
        """
        拟合MSSA模型
        """
        print("\n=== 拟合MSSA模型 ===")

        # 准备完整数据用于MSSA拟合
        data_for_mssa = self._prepare_mssa_data_for_fitting(data_with_missing)

        # 创建并拟合MSSA
        self.mssa_model = MSSA(
            window_length=self.mssa_window_length,
            n_components=self.mssa_n_components,
            groups=self.mssa_groups,
            verbose=True
        )

        self.mssa_model.fit(data_for_mssa)

        # 执行重构以获得分组的分量
        self.mssa_reconstructed = self.mssa_model.reconstruct(self.mssa_groups)

        print(f"MSSA重构形状: {self.mssa_reconstructed.shape}")
        print("MSSA拟合完成")

        return self

    def _reconstruct_with_diagonal_averaging(self, data_with_missing: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用MSSA模型和对角平均方法重构各个分组

        参数:
            data_with_missing: 包含缺失值的原始数据

        返回:
            各分组的重构结果字典
        """
        if self.mssa_model is None:
            raise ValueError("请先调用fit_mssa()方法")

        # 准备数据进行MSSA重构
        data_for_reconstruction = self._prepare_mssa_data_for_fitting(data_with_missing)

        # 如果数据长度与训练时不同，需要重新创建MSSA模型来处理
        N, P = data_for_reconstruction.shape
        M = self.mssa_model.window_length
        K = N - M + 1

        # 检查是否需要重新进行SVD（当数据长度不同时）
        original_K = self.mssa_model.K
        original_N = self.mssa_model.series_length

        if N != original_N:
            # 数据长度不同，需要重新进行MSSA分解
            print(f"数据长度不同 (原始: {original_N}, 当前: {N})，重新进行MSSA分解...")

            # 创建临时MSSA模型
            temp_mssa = MSSA(
                window_length=M,
                n_components=self.mssa_model.n_components,
                groups=self.mssa_groups,
                verbose=False
            )
            temp_mssa.fit(data_for_reconstruction)

            # 使用临时模型进行重构
            reconstructed_groups = {}
            for group_name, component_indices in self.mssa_groups.items():
                print(f"重构分组: {group_name}")

                # 构建该分组的重构矩阵
                R_group = np.zeros((M, P * K))

                for i in component_indices:
                    if i < temp_mssa.n_components:
                        # 使用临时模型的SVD分量
                        elementary_matrix = (temp_mssa.sigma[i] *
                                           np.outer(temp_mssa.U[:, i],
                                                   temp_mssa.Vt[i, :]))
                        R_group += elementary_matrix

                # 对每个时间序列进行对角平均
                group_reconstructed = np.zeros((N, P))
                for p in range(P):
                    R_p = R_group[:, p * K:(p + 1) * K]
                    group_reconstructed[:, p] = temp_mssa._diagonal_averaging(R_p)

                reconstructed_groups[group_name] = group_reconstructed

        else:
            # 数据长度相同，可以直接使用训练好的模型
            reconstructed_groups = {}

            for group_name, component_indices in self.mssa_groups.items():
                print(f"重构分组: {group_name}")

                # 构建该分组的重构矩阵
                R_group = np.zeros((M, P * K))

                for i in component_indices:
                    if i < self.mssa_model.n_components:
                        # 使用训练好的SVD分量
                        elementary_matrix = (self.mssa_model.sigma[i] *
                                           np.outer(self.mssa_model.U[:, i],
                                                   self.mssa_model.Vt[i, :]))
                        R_group += elementary_matrix

                # 对每个时间序列进行对角平均
                group_reconstructed = np.zeros((N, P))
                for p in range(P):
                    R_p = R_group[:, p * K:(p + 1) * K]
                    group_reconstructed[:, p] = self.mssa_model._diagonal_averaging(R_p)

                reconstructed_groups[group_name] = group_reconstructed

        return reconstructed_groups

    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
              sequence_length: int = 50, patience: int = 10, verbose: bool = True):
        """
        训练各个分组的Transformer模型
        """
        print("\n=== 训练Transformer模型 ===")

        # 首先拟合MSSA（如果还没有）
        if self.mssa_model is None:
            self.fit_mssa(train_data)

        # 获取各分组的重构数据
        print("重构训练数据...")
        train_groups = self._reconstruct_with_diagonal_averaging(train_data)

        val_groups = None
        if val_data is not None:
            print("重构验证数据...")
            val_groups = self._reconstruct_with_diagonal_averaging(val_data)

        # 为每个分组训练独立的Transformer
        all_train_losses = {}
        all_val_losses = {}

        for group_name, group_data in train_groups.items():
            print(f"\n--- 训练分组 '{group_name}' 的Transformer ---")

            # 创建该分组的Transformer
            input_dim = group_data.shape[1]  # 特征维度
            transformer = TransformerImputer(
                input_dim=input_dim,
                **self.transformer_config
            ).to(self.device)

            # 创建标准化器
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            # 标准化训练数据
            group_data_scaled = scaler.fit_transform(
                group_data.reshape(-1, input_dim)
            ).reshape(group_data.shape)

            # 创建数据集和数据加载器
            train_dataset = TimeSeriesDataset(
                group_data_scaled,
                sequence_length=sequence_length,
                prediction_length=self.transformer_config['prediction_length']
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            # 优化器和调度器
            optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )

            # 训练循环
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # 训练阶段
                transformer.train()
                epoch_loss = 0

                for batch in train_loader:
                    src, src_mask, target, target_mask = [x.to(self.device) for x in batch]

                    optimizer.zero_grad()
                    output = transformer(src, src_mask)

                    # 计算带掩码的损失
                    loss = self._masked_mse_loss(output, target, target_mask)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # 验证阶段
                val_loss = None
                if val_groups is not None:
                    try:
                        val_loss = self._evaluate_group(
                            transformer, scaler, val_groups[group_name], sequence_length
                        )
                        val_losses.append(val_loss)

                        scheduler.step(val_loss)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            # 保存最佳模型状态
                            best_model_state = transformer.state_dict()
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            if verbose:
                                print(f"早停在epoch {epoch}")
                            # 恢复最佳模型
                            if 'best_model_state' in locals():
                                transformer.load_state_dict(best_model_state)
                            break
                    except Exception as e:
                        if verbose:
                            print(f"验证过程出错，跳过验证: {e}")
                        val_groups = None  # 禁用后续验证

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}", end="")
                    if val_loss is not None:
                        print(f", Val Loss = {val_loss:.4f}")
                    else:
                        print()

            # 保存训练好的模型和标准化器
            self.group_transformers[group_name] = transformer
            self.scalers[group_name] = scaler
            all_train_losses[group_name] = train_losses
            all_val_losses[group_name] = val_losses

            print(f"分组 '{group_name}' 训练完成")

        return all_train_losses, all_val_losses

    def _masked_mse_loss(self, pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """计算带掩码的MSE损失"""
        masked_pred = pred * mask
        masked_target = target * mask

        loss = (masked_pred - masked_target) ** 2
        loss = loss.sum() / mask.sum()

        return loss

    def _evaluate_group(self, transformer, scaler, group_data: np.ndarray,
                       sequence_length: int) -> float:
        """评估单个分组的模型性能"""
        transformer.eval()

        # 标准化数据
        group_data_scaled = scaler.transform(
            group_data.reshape(-1, group_data.shape[-1])
        ).reshape(group_data.shape)

        # 创建数据集
        dataset = TimeSeriesDataset(
            group_data_scaled,
            sequence_length=sequence_length,
            prediction_length=self.transformer_config['prediction_length']
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                src, src_mask, target, target_mask = [x.to(self.device) for x in batch]
                output = transformer(src, src_mask)
                loss = self._masked_mse_loss(output, target, target_mask)
                total_loss += loss.item()

        return total_loss / len(loader)

    def impute(self, data_with_missing: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        使用训练好的模型进行缺失值填补

        通过对角平均将各分组的预测结果组合
        """
        print(f"\n=== 执行缺失值填补 (迭代{iterations}次) ===")

        if not self.group_transformers:
            raise ValueError("请先训练模型")

        imputed_data = data_with_missing.copy()

        for iteration in range(iterations):
            print(f"迭代 {iteration + 1}/{iterations}")

            # 获取当前数据的各分组重构
            current_groups = self._reconstruct_with_diagonal_averaging(imputed_data)

            # 为每个分组生成预测
            group_predictions = {}

            for group_name, group_data in current_groups.items():
                if group_name not in self.group_transformers:
                    continue

                transformer = self.group_transformers[group_name]
                scaler = self.scalers[group_name]

                # 预测该分组
                group_pred = self._predict_group(
                    transformer, scaler, group_data, imputed_data
                )
                group_predictions[group_name] = group_pred

            # 使用加权平均组合各分组的预测
            # 这里可以根据各分组的重要性设置不同权重
            weights = {
                'trend': 0.4,
                'seasonal': 0.4,
                'noise': 0.2
            }

            # 组合预测结果
            combined_prediction = np.zeros_like(imputed_data)
            total_weight = 0

            for group_name, group_pred in group_predictions.items():
                weight = weights.get(group_name, 1.0 / len(group_predictions))
                combined_prediction += weight * group_pred
                total_weight += weight

            if total_weight > 0:
                combined_prediction /= total_weight

            # 只更新缺失值位置
            missing_mask = np.isnan(data_with_missing)
            imputed_data[missing_mask] = combined_prediction[missing_mask]

        print("缺失值填补完成")
        return imputed_data

    def _predict_group(self, transformer, scaler, group_data: np.ndarray,
                      original_data: np.ndarray) -> np.ndarray:
        """为单个分组生成预测"""
        transformer.eval()
        predicted_data = group_data.copy()

        with torch.no_grad():
            # 标准化
            group_data_scaled = scaler.transform(
                group_data.reshape(-1, group_data.shape[-1])
            ).reshape(group_data.shape)

            # 逐步预测
            for t in range(len(original_data) - 1):
                if np.any(np.isnan(original_data[t + 1])):
                    # 准备输入序列
                    start_idx = max(0, t + 1 - 50)
                    seq = group_data_scaled[start_idx:t + 1]

                    if len(seq) < 50:
                        pad_length = 50 - len(seq)
                        seq = np.pad(seq, ((pad_length, 0), (0, 0)), mode='constant')

                    # 创建mask
                    mask = ~np.isnan(seq)
                    seq = np.nan_to_num(seq, nan=0.0)

                    # 转换为tensor
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

                    # 预测
                    pred = transformer(seq_tensor, mask_tensor)
                    pred = pred.squeeze(0).cpu().numpy()

                    # 反标准化
                    pred = scaler.inverse_transform(pred).squeeze()

                    # 更新预测数据
                    missing_mask = np.isnan(original_data[t + 1])
                    predicted_data[t + 1][missing_mask] = pred[missing_mask]

        return predicted_data

    def evaluate_imputation(self, true_data: np.ndarray,
                           imputed_data: np.ndarray,
                           original_missing_mask: np.ndarray) -> Dict[str, float]:
        """评估填补效果"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # 只评估原本缺失位置的填补效果
        true_missing = true_data[original_missing_mask]
        imputed_missing = imputed_data[original_missing_mask]

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(true_missing, imputed_missing)),
            'MAE': mean_absolute_error(true_missing, imputed_missing),
            'MAPE': np.mean(np.abs((true_missing - imputed_missing) / (true_missing + 1e-8))) * 100,
            'R2': r2_score(true_missing, imputed_missing)
        }

        # 计算各特征的评估指标
        feature_metrics = {}
        for i in range(true_data.shape[1]):
            feature_mask = original_missing_mask[:, i]
            if np.any(feature_mask):
                true_feat = true_data[feature_mask, i]
                imputed_feat = imputed_data[feature_mask, i]

                feature_metrics[f'Feature_{i}'] = {
                    'RMSE': np.sqrt(mean_squared_error(true_feat, imputed_feat)),
                    'MAE': mean_absolute_error(true_feat, imputed_feat),
                    'Missing_Count': np.sum(feature_mask)
                }

        return metrics, feature_metrics


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
                      c='red', s=20, label='填补值', zorder=5, alpha=0.8)

        ax.set_title(f'{feat} - 改进的SpectraFormer填补结果')
        ax.set_ylabel('值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('时间')
    plt.tight_layout()
    plt.show()


def main():
    """主函数：改进的SpectraFormer缺失值填补流程"""
    print("="*70)
    print("改进的SpectraFormer缺失值填补 - 使用标准对角平均")
    print("="*70)

    # 配置参数
    DATA_WITH_MISSING_PATH = str(DATA_DIR / "train_continuous_missing.csv")
    TRUE_DATA_PATH = str(DATA_DIR / "train_swat.csv")

    # MSSA参数
    MSSA_WINDOW_LENGTH = 48
    MSSA_N_COMPONENTS = 15

    # MSSA分组配置
    MSSA_GROUPS = {
        'trend': [0, 1, 2],           # 趋势分量
        'seasonal': [3, 4, 5, 6, 7, 8, 9, 10],  # 季节性分量
        'noise': [11, 12, 13, 14]     # 噪声分量
    }

    # Transformer配置
    TRANSFORMER_CONFIG = {
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'prediction_length': 1
    }

    # 1. 加载数据
    data_with_missing, true_data, missing_mask = load_data_with_missing(
        DATA_WITH_MISSING_PATH,
        TRUE_DATA_PATH,
        index_col=0
    )

    # 2. 数据集划分
    split_idx = int(len(data_with_missing) * 0.8)
    train_data = data_with_missing.iloc[:split_idx].values
    val_data = data_with_missing.iloc[split_idx:].values

    print(f"\n数据集划分:")
    print(f"训练集: {train_data.shape}")
    print(f"验证集: {val_data.shape}")

    # 3. 创建改进的填补器
    imputer = ImprovedMSSATransformerImputer(
        mssa_window_length=MSSA_WINDOW_LENGTH,
        mssa_n_components=MSSA_N_COMPONENTS,
        mssa_groups=MSSA_GROUPS,
        transformer_config=TRANSFORMER_CONFIG
    )

    # 4. 训练模型
    print("\n开始训练改进的SpectraFormer模型...")
    train_losses, val_losses = imputer.train(
        train_data,
        val_data,
        epochs=80,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=50,
        patience=15,
        verbose=True
    )

    # 5. 可视化训练过程
    fig, axes = plt.subplots(1, len(train_losses), figsize=(5*len(train_losses), 4))
    if len(train_losses) == 1:
        axes = [axes]

    for idx, (group_name, losses) in enumerate(train_losses.items()):
        ax = axes[idx]
        ax.plot(losses, label=f'{group_name} - Train')
        if group_name in val_losses:
            ax.plot(val_losses[group_name], label=f'{group_name} - Val')
        ax.set_title(f'分组 {group_name} 训练过程')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # 6. 执行填补
    print("\n执行缺失值填补...")
    imputed_data = imputer.impute(
        data_with_missing.values,
        iterations=3
    )

    # 7. 评估填补效果
    if true_data is not None:
        print("\n评估填补效果...")
        metrics, feature_metrics = imputer.evaluate_imputation(
            true_data.values,
            imputed_data,
            missing_mask
        )

        print("\n整体填补性能指标:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:>10}: {value:>10.4f}")

        print("\n各特征填补性能:")
        print("-" * 60)
        print(f"{'特征':>15} {'缺失数':>10} {'RMSE':>10} {'MAE':>10}")
        print("-" * 60)

        for i, col in enumerate(data_with_missing.columns):
            if f'Feature_{i}' in feature_metrics:
                feat_metrics = feature_metrics[f'Feature_{i}']
                print(f"{col:>15} {feat_metrics['Missing_Count']:>10} "
                      f"{feat_metrics['RMSE']:>10.4f} {feat_metrics['MAE']:>10.4f}")

    # 8. 保存结果
    print("\n保存填补后的数据...")
    imputed_df = pd.DataFrame(
        imputed_data,
        index=data_with_missing.index,
        columns=data_with_missing.columns
    )
    imputed_df.to_csv(imputation_output_path("improved_imputed_data.csv"))
    print("已保存到: improved_imputed_data.csv")

    # 9. 可视化结果
    visualize_results(
        data_with_missing,
        imputed_data,
        true_data,
        missing_mask,
        time_range=(200, 400),
        features_to_plot=None
    )

    print("\n改进的SpectraFormer填补流程完成！")

    return imputed_data, metrics if true_data is not None else None


if __name__ == "__main__":
    # 运行主流程
    imputed_data, metrics = main()
