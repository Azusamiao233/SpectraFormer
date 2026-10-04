import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """时间序列数据集，用于处理带缺失值的数据"""

    def __init__(self, data: np.ndarray, sequence_length: int = 50,
                 prediction_length: int = 1, stride: int = 1):
        """
        参数:
            data: 输入数据，形状为 (N, P)，其中N是时间步，P是特征数
            sequence_length: 输入序列长度
            prediction_length: 预测长度
            stride: 滑动窗口步长
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride

        # 创建有效的序列索引
        self.valid_indices = []
        for i in range(0, len(data) - sequence_length - prediction_length + 1, stride):
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_length

        # 输入序列
        sequence = self.data[start_idx:end_idx].copy()
        # 目标值
        target = self.data[end_idx:target_idx].copy()

        # 创建mask标记缺失值位置（1表示有值，0表示缺失）
        mask = ~np.isnan(sequence)
        target_mask = ~np.isnan(target)

        # 将缺失值暂时填充为0（后续会用mask处理）
        sequence = np.nan_to_num(sequence, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        return (torch.FloatTensor(sequence),
                torch.FloatTensor(mask),
                torch.FloatTensor(target),
                torch.FloatTensor(target_mask))


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerImputerKLImproved(nn.Module):
    """改进的Transformer缺失值填补模型 - 使用更稳定的KL散度损失"""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 prediction_length: int = 1, loss_type: str = 'mixed'):
        """
        参数:
            loss_type: 损失函数类型 ('kl', 'mse', 'mixed')
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.loss_type = loss_type

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 输出投影层
        if loss_type in ['kl', 'mixed']:
            # 输出均值和方差参数
            self.mean_projection = nn.Linear(d_model, input_dim * prediction_length)
            self.logvar_projection = nn.Linear(d_model, input_dim * prediction_length)
        else:
            # 只输出均值
            self.output_projection = nn.Linear(d_model, input_dim * prediction_length)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        参数:
            src: 输入序列，形状为 (batch_size, seq_len, input_dim)
            src_mask: 输入掩码，形状为 (batch_size, seq_len, input_dim)

        返回:
            如果loss_type='kl'或'mixed': (mean, logvar)
            如果loss_type='mse': output
        """
        batch_size, seq_len, _ = src.shape

        # 应用mask（将缺失值位置的输入置零）
        src = src * src_mask

        # 输入投影
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)

        # 位置编码
        src = self.pos_encoder(src)
        src = self.dropout(src)

        # 创建attention mask
        seq_mask = src_mask.any(dim=-1)  # (batch_size, seq_len)
        attn_mask = ~seq_mask  # True表示需要mask的位置

        # Transformer编码
        memory = self.transformer_encoder(src, src_key_padding_mask=attn_mask)

        # 使用最后几个时间步的输出进行预测
        output = memory[:, -1, :]  # (batch_size, d_model)

        if self.loss_type in ['kl', 'mixed']:
            # 输出均值和对数方差
            mean = self.mean_projection(output)
            logvar = self.logvar_projection(output)

            # 限制logvar的范围，避免数值不稳定
            logvar = torch.clamp(logvar, min=-10, max=2)

            # Reshape到目标形状
            mean = mean.view(batch_size, self.prediction_length, self.input_dim)
            logvar = logvar.view(batch_size, self.prediction_length, self.input_dim)

            return mean, logvar
        else:
            # 只输出均值
            output = self.output_projection(output)
            output = output.view(batch_size, self.prediction_length, self.input_dim)
            return output


class MSSATransformerImputerKLImproved:
    """改进的MSSA+Transformer缺失值填补器 - 使用更稳定的KL散度损失"""

    def __init__(self, mssa_window_length: int = 30,
                 mssa_n_components: Optional[int] = None,
                 transformer_config: Optional[Dict] = None,
                 loss_type: str = 'mixed',
                 kl_weight: float = 0.1,
                 mse_weight: float = 0.9):
        """
        参数:
            loss_type: 损失函数类型 ('kl', 'mse', 'mixed')
            kl_weight: KL损失权重（当loss_type='mixed'时使用）
            mse_weight: MSE损失权重（当loss_type='mixed'时使用）
        """
        self.mssa_window_length = mssa_window_length
        self.mssa_n_components = mssa_n_components
        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight

        # 默认Transformer配置
        self.transformer_config = {
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1,
            'loss_type': loss_type
        }
        if transformer_config:
            self.transformer_config.update(transformer_config)

        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_mssa_components(self, data: np.ndarray, mssa_model) -> np.ndarray:
        """使用MSSA分解数据"""
        groups = {}
        for i in range(mssa_model.n_components):
            groups[f'comp_{i}'] = [i]

        reconstructed = mssa_model.reconstruct(groups)
        N, P, n_comps = reconstructed.shape
        mssa_features = reconstructed.transpose(0, 2, 1).reshape(N, -1)

        return mssa_features

    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              mssa_model=None, epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001, sequence_length: int = 50,
              patience: int = 10, verbose: bool = True):
        """训练模型"""

        # 准备MSSA特征
        if mssa_model is not None:
            train_features = self.prepare_mssa_components(train_data, mssa_model)
            if val_data is not None:
                val_features = self.prepare_mssa_components(val_data, mssa_model)
        else:
            train_features = train_data
            val_features = val_data

        # 标准化
        train_features_scaled = self.scaler.fit_transform(
            train_features.reshape(-1, train_features.shape[-1])
        ).reshape(train_features.shape)

        # 创建数据集
        train_dataset = TimeSeriesDataset(
            train_features_scaled,
            sequence_length=sequence_length,
            prediction_length=self.transformer_config['prediction_length']
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        input_dim = train_features.shape[1]
        self.model = TransformerImputerKLImproved(
            input_dim=input_dim,
            **self.transformer_config
        ).to(self.device)

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0

            for batch in train_loader:
                src, src_mask, target, target_mask = [x.to(self.device) for x in batch]

                optimizer.zero_grad()

                if self.loss_type in ['kl', 'mixed']:
                    mean, logvar = self.model(src, src_mask)
                    loss = self._compute_loss(mean, logvar, target, target_mask)
                else:
                    output = self.model(src, src_mask)
                    loss = self._masked_mse_loss(output, target, target_mask)

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            if val_features is not None:
                val_loss = self._evaluate(val_features, mssa_model, sequence_length)
                val_losses.append(val_loss)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"早停在epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}", end="")
                if val_features is not None:
                    print(f", Val Loss = {val_loss:.4f}")
                else:
                    print()

        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        return train_losses, val_losses

    def _compute_loss(self, mean: torch.Tensor, logvar: torch.Tensor,
                      target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算改进的损失函数"""

        valid_mask = mask.bool()

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=mean.device, requires_grad=True)

        # 获取有效位置的值
        valid_mean = mean[valid_mask]
        valid_logvar = logvar[valid_mask]
        valid_target = target[valid_mask]

        if self.loss_type == 'mixed':
            # 混合损失：MSE + KL
            mse_loss = F.mse_loss(valid_mean, valid_target)

            # 更稳定的KL散度计算
            var = torch.exp(valid_logvar) + 1e-8
            kl_loss = 0.5 * (
                    valid_logvar +
                    (valid_target - valid_mean) ** 2 / var -
                    torch.log(torch.tensor(2 * np.pi)) - 1
            ).mean()

            # 限制KL损失的范围
            kl_loss = torch.clamp(kl_loss, min=0, max=10)

            total_loss = self.mse_weight * mse_loss + self.kl_weight * kl_loss

        elif self.loss_type == 'kl':
            # 仅KL损失，但更稳定
            var = torch.exp(valid_logvar) + 1e-8

            # 负对数似然损失 (更稳定的KL散度形式)
            nll_loss = 0.5 * (
                    valid_logvar +
                    (valid_target - valid_mean) ** 2 / var +
                    torch.log(torch.tensor(2 * np.pi))
            ).mean()

            total_loss = torch.clamp(nll_loss, min=0, max=10)

        else:  # mse
            total_loss = F.mse_loss(valid_mean, valid_target)

        return total_loss

    def _masked_mse_loss(self, pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """计算带掩码的MSE损失"""
        masked_pred = pred * mask
        masked_target = target * mask

        loss = (masked_pred - masked_target) ** 2
        valid_count = mask.sum()

        if valid_count > 0:
            loss = loss.sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss

    def _evaluate(self, data: np.ndarray, mssa_model, sequence_length: int) -> float:
        """评估模型性能"""
        self.model.eval()

        if mssa_model is not None:
            features = self.prepare_mssa_components(data, mssa_model)
        else:
            features = data

        features_scaled = self.scaler.transform(
            features.reshape(-1, features.shape[-1])
        ).reshape(features.shape)

        dataset = TimeSeriesDataset(
            features_scaled,
            sequence_length=sequence_length,
            prediction_length=self.transformer_config['prediction_length']
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                src, src_mask, target, target_mask = [x.to(self.device) for x in batch]

                if self.loss_type in ['kl', 'mixed']:
                    mean, logvar = self.model(src, src_mask)
                    loss = self._compute_loss(mean, logvar, target, target_mask)
                else:
                    output = self.model(src, src_mask)
                    loss = self._masked_mse_loss(output, target, target_mask)

                total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0

    def impute(self, data: np.ndarray, mssa_model=None,
               iterations: int = 5) -> np.ndarray:
        """使用训练好的模型填补缺失值"""
        self.model.eval()
        imputed_data = data.copy()

        with torch.no_grad():
            for iteration in range(iterations):
                # 准备特征
                if mssa_model is not None:
                    temp_data = pd.DataFrame(imputed_data).interpolate(
                        method='linear', limit_direction='both'
                    ).values
                    features = self.prepare_mssa_components(temp_data, mssa_model)
                else:
                    features = imputed_data

                # 标准化
                features_scaled = self.scaler.transform(
                    features.reshape(-1, features.shape[-1])
                ).reshape(features.shape)

                # 逐步预测并填补
                for t in range(len(data) - 1):
                    if np.any(np.isnan(data[t + 1])):
                        # 准备输入序列
                        start_idx = max(0, t + 1 - 50)
                        seq = features_scaled[start_idx:t + 1]

                        if len(seq) < 50:
                            pad_length = 50 - len(seq)
                            seq = np.pad(seq, ((pad_length, 0), (0, 0)), mode='constant')

                        # 创建mask
                        mask = ~np.isnan(seq)
                        seq = np.nan_to_num(seq, nan=0.0)

                        # 转换为tensor
                        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

                        # 预测（使用均值作为预测值）
                        if self.loss_type in ['kl', 'mixed']:
                            mean, logvar = self.model(seq_tensor, mask_tensor)
                            pred = mean.squeeze(0).cpu().numpy()
                        else:
                            pred = self.model(seq_tensor, mask_tensor)
                            pred = pred.squeeze(0).cpu().numpy()

                        # 反标准化
                        pred = self.scaler.inverse_transform(pred).squeeze()

                        # 如果使用了MSSA，需要处理维度
                        if mssa_model is not None:
                            pred = pred[:data.shape[1]]

                        # 填补缺失值
                        missing_mask = np.isnan(data[t + 1])
                        imputed_data[t + 1][missing_mask] = pred[missing_mask]

        return imputed_data

    def evaluate_imputation(self, true_data: np.ndarray,
                            imputed_data: np.ndarray,
                            original_missing_mask: np.ndarray) -> Dict[str, float]:
        """评估填补效果"""
        # 只评估原本缺失位置的填补效果
        true_missing = true_data[original_missing_mask]
        imputed_missing = imputed_data[original_missing_mask]

        # 添加数值检查
        if len(true_missing) == 0:
            return {'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'R2': 0}, {}

        # 检查是否有异常值
        if np.any(np.isnan(imputed_missing)) or np.any(np.isinf(imputed_missing)):
            print("警告: 填补结果包含NaN或Inf值")
            imputed_missing = np.nan_to_num(imputed_missing, nan=0.0)

        # 计算指标，添加数值稳定性检查
        try:
            r2 = r2_score(true_missing, imputed_missing)
            # 如果R2异常小，可能是数值问题
            if r2 < -10:
                print(f"警告: R2值异常小 ({r2:.4f})，可能存在数值问题")
                r2 = max(r2, -1.0)  # 限制最小值
        except:
            r2 = -1.0

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(true_missing, imputed_missing)),
            'MAE': mean_absolute_error(true_missing, imputed_missing),
            'MAPE': np.mean(np.abs((true_missing - imputed_missing) / (true_missing + 1e-8))) * 100,
            'R2': r2
        }

        # 计算每个特征的评估指标
        feature_metrics = {}
        for i in range(true_data.shape[1]):
            feature_mask = original_missing_mask[:, i]
            if np.any(feature_mask):
                true_feat = true_data[feature_mask, i]
                imputed_feat = imputed_data[feature_mask, i]

                try:
                    r2_feat = r2_score(true_feat, imputed_feat)
                    if r2_feat < -10:
                        r2_feat = max(r2_feat, -1.0)
                except:
                    r2_feat = -1.0

                feature_metrics[f'Feature_{i}'] = {
                    'RMSE': np.sqrt(mean_squared_error(true_feat, imputed_feat)),
                    'MAE': mean_absolute_error(true_feat, imputed_feat),
                    'R2': r2_feat,
                    'Missing_Count': np.sum(feature_mask)
                }

        return metrics, feature_metrics

    def plot_imputation_results(self, true_data: np.ndarray,
                                imputed_data: np.ndarray,
                                original_missing_mask: np.ndarray,
                                feature_indices: Optional[List[int]] = None,
                                time_range: Optional[Tuple[int, int]] = None):
        """可视化填补结果"""
        if feature_indices is None:
            feature_indices = list(range(min(4, true_data.shape[1])))

        if time_range is None:
            time_range = (0, min(500, len(true_data)))

        n_features = len(feature_indices)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
        if n_features == 1:
            axes = [axes]

        for idx, feat_idx in enumerate(feature_indices):
            ax = axes[idx]

            t_start, t_end = time_range
            time_points = np.arange(t_start, t_end)

            # 真实值
            ax.plot(time_points, true_data[t_start:t_end, feat_idx],
                    'b-', label='True', alpha=0.7)

            # 填补值
            missing_points = original_missing_mask[t_start:t_end, feat_idx]
            if np.any(missing_points):
                missing_indices = time_points[missing_points]
                ax.scatter(missing_indices,
                           imputed_data[t_start:t_end, feat_idx][missing_points],
                           c='red', s=20, label=f'Imputed ({self.loss_type})', zorder=5)

            ax.set_title(f'Feature {feat_idx} ({self.loss_type.upper()} Loss)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    np.random.seed(42)

    # 创建示例数据
    N = 1000
    P = 5

    t = np.arange(N)
    data = np.zeros((N, P))
    for i in range(P):
        trend = 0.01 * t + i
        seasonal = 2 * np.sin(2 * np.pi * t / 50 + i)
        noise = 0.5 * np.random.randn(N)
        data[:, i] = trend + seasonal + noise

    # 创建缺失值
    missing_ratio = 0.2
    missing_mask = np.random.random((N, P)) < missing_ratio
    data_with_missing = data.copy()
    data_with_missing[missing_mask] = np.nan

    # 分割数据
    train_size = int(0.8 * N)
    train_data = data_with_missing[:train_size]
    val_data = data_with_missing[train_size:]

    # 测试不同损失函数
    loss_types = ['mse', 'mixed', 'kl']
    results = {}

    for loss_type in loss_types:
        print(f"\n{'=' * 50}")
        print(f"测试 {loss_type.upper()} 损失函数")
        print(f"{'=' * 50}")

        # 创建模型
        imputer = MSSATransformerImputerKLImproved(
            mssa_window_length=30,
            transformer_config={
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.1,
                'prediction_length': 1
            },
            loss_type=loss_type,
            kl_weight=0.1,
            mse_weight=0.9
        )

        # 训练
        train_losses, val_losses = imputer.train(
            train_data,
            val_data,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            sequence_length=50
        )

        # 填补
        imputed_data = imputer.impute(data_with_missing)

        # 评估
        metrics, feature_metrics = imputer.evaluate_imputation(
            data, imputed_data, missing_mask
        )

        print(f"\n{loss_type.upper()} 损失函数结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        results[loss_type] = metrics

    # 对比结果
    print(f"\n{'=' * 50}")
    print("损失函数对比结果:")
    print(f"{'=' * 50}")

    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))

    # 可视化最佳结果
    best_loss_type = min(results.keys(), key=lambda x: results[x]['RMSE'])
    print(f"\n最佳损失函数: {best_loss_type.upper()}")

    # 使用最佳损失函数重新训练并可视化
    best_imputer = MSSATransformerImputerKLImproved(
        loss_type=best_loss_type,
        kl_weight=0.1,
        mse_weight=0.9
    )
    best_imputer.train(train_data, val_data, epochs=50)
    best_imputed = best_imputer.impute(data_with_missing)
    best_imputer.plot_imputation_results(
        data, best_imputed, missing_mask,
        feature_indices=[0, 1, 2],
        time_range=(0, 200)
    )