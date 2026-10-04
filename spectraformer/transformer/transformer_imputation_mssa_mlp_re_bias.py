import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import matplotlib.pyplot as plt
except ImportError:  # Plotting helpers are optional during model training.
    plt = None
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """时间序列数据集，用于处理带缺失值的数据"""

    def __init__(self, data: np.ndarray, mssa_features: Optional[np.ndarray] = None,
                 sequence_length: int = 50, prediction_length: int = 1, stride: int = 1):
        """
        参数:
            data: 原始输入数据，形状为 (N, P)，其中N是时间步，P是特征数
            mssa_features: MSSA处理后的特征，形状为 (N, P, K) 或 (N, P*K)
            sequence_length: 输入序列长度
            prediction_length: 预测长度
            stride: 滑动窗口步长
        """
        self.data = data
        self.mssa_features = mssa_features
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

        # 原始输入序列
        sequence = self.data[start_idx:end_idx].copy()
        # 目标值
        target = self.data[end_idx:target_idx].copy()

        # MSSA特征序列（如果有）
        mssa_sequence = None
        if self.mssa_features is not None:
            mssa_sequence = self.mssa_features[start_idx:end_idx].copy()

        # 创建mask标记缺失值位置（1表示有值，0表示缺失）
        mask = ~np.isnan(sequence)
        target_mask = ~np.isnan(target)

        # 将缺失值暂时填充为0（后续会用mask处理）
        sequence = np.nan_to_num(sequence, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        if mssa_sequence is not None:
            mssa_sequence = np.nan_to_num(mssa_sequence, nan=0.0)

        result = [
            torch.FloatTensor(sequence),
            torch.FloatTensor(mask),
            torch.FloatTensor(target),
            torch.FloatTensor(target_mask)
        ]

        if mssa_sequence is not None:
            result.append(torch.FloatTensor(mssa_sequence))

        return tuple(result)


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


class AdaptiveModalFusion(nn.Module):
    """自适应模态融合模块

    基于输入条件动态调整各模态的重要性权重
    """

    def __init__(self, n_modes: int, feature_dim: int, d_model: int = 128):
        """
        参数:
            n_modes: 模态数量 (K)
            feature_dim: 特征维度 (P)
            d_model: 内部表示维度
        """
        super().__init__()
        self.n_modes = n_modes
        self.feature_dim = feature_dim

        # 将每个模态映射到统一的表示空间
        self.mode_projection = nn.Linear(feature_dim, d_model)

        # 计算模态权重的网络
        self.weight_net = nn.Sequential(
            nn.Linear(d_model * n_modes, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_modes)
        )

        # 输出投影
        self.output_projection = nn.Linear(d_model, feature_dim)

    def forward(self, Z):
        """
        参数:
            Z: MSSA分解特征，形状为 (batch_size, seq_len, P, K)

        返回:
            fused_features: 融合后的特征，形状为 (batch_size, seq_len, P)
            attention_weights: 注意力权重，形状为 (batch_size, seq_len, K)
        """
        batch_size, seq_len, P, K = Z.shape

        # 1. 将每个模态投影到表示空间
        Z_reshaped = Z.permute(0, 1, 3, 2).contiguous()  # (batch, seq, K, P)
        Z_flat = Z_reshaped.view(-1, P)  # (batch*seq*K, P)
        h = F.relu(self.mode_projection(Z_flat))  # (batch*seq*K, d_model)
        h = h.view(batch_size, seq_len, K, -1)  # (batch, seq, K, d_model)

        # 2. 计算注意力权重
        h_concat = h.view(batch_size, seq_len, -1)  # (batch, seq, K*d_model)
        logits = self.weight_net(h_concat)  # (batch, seq, K)
        alpha = F.softmax(logits, dim=-1)  # (batch, seq, K)

        # 3. 加权融合
        alpha_expanded = alpha.unsqueeze(2)  # (batch, seq, 1, K)
        weighted_modes = Z * alpha_expanded  # (batch, seq, P, K)
        fused = weighted_modes.sum(dim=-1)  # (batch, seq, P)

        return fused, alpha


class SpectralAttentionBias(nn.Module):
    """谱感知注意力偏置模块"""

    def __init__(self, max_seq_len: int = 512, top_k_freq: int = 10,
                 learnable_weights: bool = True, temperature: float = 1.0):
        """
        参数:
            max_seq_len: 最大序列长度
            top_k_freq: 保留的主导频率数量
            learnable_weights: 是否使用可学习的频率权重
            temperature: 温度参数，控制偏置强度
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.top_k_freq = top_k_freq
        self.temperature = temperature

        # 可学习的频率权重
        if learnable_weights:
            self.freq_weights = nn.Parameter(torch.ones(top_k_freq))
        else:
            self.register_buffer('freq_weights', torch.ones(top_k_freq))

        # 预计算频率基础
        self.register_buffer('freq_base', torch.arange(1, top_k_freq + 1, dtype=torch.float))

    def compute_power_spectrum(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算输入序列的功率谱

        参数:
            x: 输入序列 (batch_size, seq_len, n_features)
            mask: 掩码 (batch_size, seq_len, n_features)

        返回:
            power_spectrum: 功率谱 (batch_size, n_features, n_freq_bins)
        """
        batch_size, seq_len, n_features = x.shape

        # 应用掩码，将缺失值位置置零
        x_masked = x * mask

        # 计算每个特征的DFT
        # 使用实数FFT以提高效率
        fft_result = torch.fft.rfft(x_masked, dim=1)  # (batch_size, n_freq_bins, n_features)

        # 计算功率谱（模的平方）
        power_spectrum = torch.abs(fft_result) ** 2  # (batch_size, n_freq_bins, n_features)

        # 转置以匹配预期形状
        power_spectrum = power_spectrum.transpose(1, 2)  # (batch_size, n_features, n_freq_bins)

        return power_spectrum

    def select_dominant_frequencies(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """
        选择主导频率

        参数:
            power_spectrum: 功率谱 (batch_size, n_features, n_freq_bins)

        返回:
            dominant_freqs: 主导频率索引 (batch_size, top_k_freq)
        """
        batch_size, n_features, n_freq_bins = power_spectrum.shape

        # 对所有特征的功率谱求平均
        avg_power = power_spectrum.mean(dim=1)  # (batch_size, n_freq_bins)

        # 选择top-k频率（排除DC分量，即索引0）
        if n_freq_bins > 1:
            # 排除DC分量
            avg_power_no_dc = avg_power[:, 1:]
            _, top_indices = torch.topk(avg_power_no_dc,
                                        min(self.top_k_freq, avg_power_no_dc.size(1)),
                                        dim=1)
            # 加1是因为我们排除了DC分量
            dominant_freqs = top_indices + 1
        else:
            # 如果只有DC分量，返回零频率
            dominant_freqs = torch.zeros(batch_size, 1, device=power_spectrum.device, dtype=torch.long)

        return dominant_freqs

    def compute_spectral_bias(self, seq_len: int, dominant_freqs: torch.Tensor) -> torch.Tensor:
        """
        计算谱注意力偏置矩阵

        参数:
            seq_len: 序列长度
            dominant_freqs: 主导频率 (batch_size, top_k_freq)

        返回:
            bias_matrix: 偏置矩阵 (batch_size, seq_len, seq_len)
        """
        batch_size = dominant_freqs.size(0)
        device = dominant_freqs.device

        # 创建时间差矩阵
        t1 = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        t2 = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        time_diff = (t1 - t2).float()  # (seq_len, seq_len)

        # 初始化偏置矩阵
        bias_matrix = torch.zeros(batch_size, seq_len, seq_len, device=device)

        # 对每个批次计算偏置
        for b in range(batch_size):
            batch_bias = torch.zeros(seq_len, seq_len, device=device)

            # 对每个主导频率计算贡献
            for i, freq in enumerate(dominant_freqs[b]):
                if i < len(self.freq_weights):
                    # 计算频率为freq时的余弦相似性
                    freq_contribution = torch.cos(2 * np.pi * freq.float() * time_diff / seq_len)

                    # 应用可学习权重
                    weight = torch.sigmoid(self.freq_weights[i])  # 确保权重为正
                    batch_bias += weight * freq_contribution

            bias_matrix[b] = batch_bias

        # 应用温度缩放
        bias_matrix = bias_matrix / self.temperature

        return bias_matrix

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入序列 (batch_size, seq_len, n_features)
            mask: 掩码 (batch_size, seq_len, n_features)

        返回:
            bias_matrix: 注意力偏置矩阵 (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, n_features = x.shape

        # 计算功率谱
        power_spectrum = self.compute_power_spectrum(x, mask)

        # 选择主导频率
        dominant_freqs = self.select_dominant_frequencies(power_spectrum)

        # 计算谱偏置矩阵
        bias_matrix = self.compute_spectral_bias(seq_len, dominant_freqs)

        return bias_matrix


class SpectralAwareMultiHeadAttention(nn.Module):
    """谱感知多头注意力机制"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 spectral_bias: bool = True, **spectral_kwargs):
        """
        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: dropout概率
            spectral_bias: 是否使用谱偏置
            **spectral_kwargs: 谱偏置模块的参数
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.use_spectral_bias = spectral_bias

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # 谱偏置模块
        if self.use_spectral_bias:
            self.spectral_bias = SpectralAttentionBias(**spectral_kwargs)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                original_input: torch.Tensor = None, input_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        """
        前向传播

        参数:
            query, key, value: 注意力的Q, K, V (batch_size, seq_len, d_model)
            original_input: 原始输入序列，用于计算谱偏置 (batch_size, seq_len, n_features)
            input_mask: 输入掩码 (batch_size, seq_len, n_features)
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
        """
        batch_size, seq_len, d_model = query.size()

        # 线性投影
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)  # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)

        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (batch_size, nhead, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (batch_size, nhead, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # (batch_size, nhead, seq_len, d_k)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # (batch_size, nhead, seq_len, seq_len)

        # 添加谱偏置
        if self.use_spectral_bias and original_input is not None and input_mask is not None:
            spectral_bias = self.spectral_bias(original_input, input_mask)  # (batch_size, seq_len, seq_len)
            # 扩展到多头
            spectral_bias = spectral_bias.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            scores = scores + spectral_bias

        # 应用其他掩码
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        output = torch.matmul(attn_weights, V)  # (batch_size, nhead, seq_len, d_k)

        # 重塑回原始形状
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 输出投影
        output = self.w_o(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + query)

        return output, attn_weights


class SpectralAwareTransformerEncoderLayer(nn.Module):
    """谱感知Transformer编码器层"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, spectral_bias: bool = True, **spectral_kwargs):
        super().__init__()

        # 谱感知多头注意力
        self.self_attn = SpectralAwareMultiHeadAttention(
            d_model, nhead, dropout, spectral_bias, **spectral_kwargs
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, original_input: torch.Tensor = None,
                input_mask: torch.Tensor = None, src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None):
        """
        参数:
            src: 输入 (batch_size, seq_len, d_model)
            original_input: 原始输入，用于计算谱偏置 (batch_size, seq_len, n_features)
            input_mask: 输入掩码 (batch_size, seq_len, n_features)
            src_mask: 源掩码
            src_key_padding_mask: 源键填充掩码
        """
        # 自注意力
        src2, attn_weights = self.self_attn(
            src, src, src, original_input, input_mask, src_mask, src_key_padding_mask
        )

        # 前馈网络
        src2 = self.ffn(src2)
        src = self.norm2(src + src2)

        return src, attn_weights


class MSSAFeatureProcessor(nn.Module):
    """MSSA特征处理器 - 集成自适应模态融合"""

    def __init__(self, mssa_feature_dim: int, output_dim: int, n_modes: int,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.1,
                 use_modal_fusion: bool = True, fusion_type: str = 'adaptive'):
        """
        参数:
            mssa_feature_dim: MSSA特征维度 (P * n_components)
            output_dim: 输出特征维度 (P)
            n_modes: MSSA模态数量 (K)
            hidden_dims: MLP隐藏层维度列表
            dropout: dropout比例
            use_modal_fusion: 是否使用模态融合
            fusion_type: 融合类型 ('adaptive', 'mlp')
        """
        super().__init__()
        self.mssa_feature_dim = mssa_feature_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.use_modal_fusion = use_modal_fusion
        self.fusion_type = fusion_type

        if use_modal_fusion and fusion_type == 'adaptive':
            # 使用自适应模态融合
            self.modal_fusion = AdaptiveModalFusion(n_modes, output_dim)

            # 融合后的微调MLP
            self.post_fusion_mlp = nn.Sequential(
                nn.Linear(output_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1], output_dim)
            )
        else:
            # 原始的MLP处理方式
            layers = []
            in_dim = mssa_feature_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                ])
                in_dim = hidden_dim

            layers.append(nn.Linear(in_dim, output_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, mssa_features):
        """
        参数:
            mssa_features: MSSA特征，形状为 (batch_size, seq_len, P*K) 或 (batch_size, seq_len, P, K)
        返回:
            processed_features: 处理后的特征，形状为 (batch_size, seq_len, P)
            attention_weights: 注意力权重（如果使用模态融合），形状为 (batch_size, seq_len, K)
        """
        batch_size, seq_len = mssa_features.shape[:2]

        if seq_len == 0:
            return torch.zeros(batch_size, 0, self.output_dim,
                               device=mssa_features.device, dtype=mssa_features.dtype), None

        if self.use_modal_fusion and self.fusion_type == 'adaptive':
            # 确保输入是4维的 (batch, seq, P, K)
            if len(mssa_features.shape) == 3:
                mssa_features = mssa_features.view(batch_size, seq_len, self.output_dim, self.n_modes)

            # 应用自适应模态融合
            fused_features, attention_weights = self.modal_fusion(mssa_features)

            # 通过后处理MLP
            fused_flat = fused_features.view(-1, self.output_dim)
            processed_flat = self.post_fusion_mlp(fused_flat)
            processed = processed_flat.view(batch_size, seq_len, self.output_dim)

            return processed, attention_weights
        else:
            # 原始MLP处理
            if len(mssa_features.shape) == 4:
                mssa_features = mssa_features.view(batch_size, seq_len, -1)

            mssa_flat = mssa_features.view(-1, self.mssa_feature_dim)

            if mssa_flat.size(0) == 0:
                return torch.zeros(batch_size, seq_len, self.output_dim,
                                   device=mssa_features.device, dtype=mssa_features.dtype), None

            processed_flat = self.mlp(mssa_flat)
            processed = processed_flat.view(batch_size, seq_len, -1)

            return processed, None


class SpectralAwareTransformerImputerMSSAMLP(nn.Module):
    """谱感知Transformer缺失值填补模型，结合MSSA+自适应模态融合"""

    def __init__(self, input_dim: int, mssa_feature_dim: Optional[int] = None,
                 n_modes: Optional[int] = None, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, prediction_length: int = 1,
                 mlp_hidden_dims: List[int] = [512, 256],
                 fusion_method: str = 'concat',
                 use_modal_fusion: bool = True,
                 modal_fusion_type: str = 'adaptive',
                 use_spectral_bias: bool = True,
                 use_transformer_encoder: bool = True,
                 spectral_config: Optional[Dict] = None):
        """
        参数:
            input_dim: 原始输入维度 (P)
            mssa_feature_dim: MSSA特征维度 (P * K)
            n_modes: MSSA模态数量 (K)
            d_model: transformer模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: dropout比例
            prediction_length: 预测长度
            mlp_hidden_dims: MLP隐藏层维度
            fusion_method: 特征融合方法
            use_modal_fusion: 是否使用模态融合
            modal_fusion_type: 模态融合类型
            use_spectral_bias: 是否使用谱偏置
            use_transformer_encoder: 是否使用Transformer编码器；False时改用轻量MLP重构头
            spectral_config: 谱偏置配置
        """
        super().__init__()

        self.input_dim = input_dim
        self.mssa_feature_dim = mssa_feature_dim
        self.n_modes = n_modes
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.fusion_method = fusion_method
        self.use_mssa = mssa_feature_dim is not None
        self.use_transformer_encoder = use_transformer_encoder
        # 谱偏置属于注意力编码器；移除编码器时该分支自然不可用。
        self.use_spectral_bias = use_spectral_bias and use_transformer_encoder
        self.use_modal_fusion = use_modal_fusion

        # 默认谱偏置配置
        default_spectral_config = {
            'max_seq_len': 512,
            'top_k_freq': 10,
            'learnable_weights': True,
            'temperature': 1.0
        }
        if spectral_config:
            default_spectral_config.update(spectral_config)
        self.spectral_config = default_spectral_config

        # MSSA特征处理器
        if self.use_mssa:
            self.mssa_processor = MSSAFeatureProcessor(
                mssa_feature_dim, input_dim, n_modes or (mssa_feature_dim // input_dim),
                mlp_hidden_dims, dropout, use_modal_fusion, modal_fusion_type
            )

            if fusion_method == 'concat':
                fusion_input_dim = input_dim * 2
            elif fusion_method == 'add':
                fusion_input_dim = input_dim
            elif fusion_method == 'gate':
                fusion_input_dim = input_dim
                self.gate_layer = nn.Sequential(
                    nn.Linear(input_dim * 2, input_dim),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"不支持的融合方法: {fusion_method}")
        else:
            fusion_input_dim = input_dim

        if self.use_transformer_encoder:
            # 输入投影层
            self.input_projection = nn.Linear(fusion_input_dim, d_model)

            # 位置编码
            self.pos_encoder = PositionalEncoding(d_model)

            # 谱感知Transformer编码器层
            self.encoder_layers = nn.ModuleList([
                SpectralAwareTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    spectral_bias=self.use_spectral_bias,
                    **default_spectral_config
                ) for _ in range(num_encoder_layers)
            ])

            # 输出投影层
            self.output_projection = nn.Linear(d_model, input_dim * prediction_length)
        else:
            # 消融变体：保留原始特征、MSSA模态融合和时间聚合，仅用一个小型
            # MLP重构下一时刻。输入同时包含末时刻状态与全窗口均值，避免把
            # “去掉Transformer”错误实现成完全没有时序上下文的逐点线性层。
            reconstruction_hidden_dim = max(16, min(d_model, 64))
            self.reconstruction_head = nn.Sequential(
                nn.Linear(fusion_input_dim * 2, reconstruction_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(reconstruction_hidden_dim, input_dim * prediction_length),
            )

        self.dropout = nn.Dropout(dropout)

        # 存储注意力权重用于分析
        self.last_modal_attention_weights = None
        self.last_spectral_attention_weights = None

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor,
                mssa_features: Optional[torch.Tensor] = None):
        """
        前向传播

        参数:
            src: 原始输入序列 (batch_size, seq_len, input_dim)
            src_mask: 输入掩码 (batch_size, seq_len, input_dim)
            mssa_features: MSSA特征 (batch_size, seq_len, mssa_feature_dim)

        返回:
            输出预测 (batch_size, prediction_length, input_dim)
        """
        batch_size, seq_len, _ = src.shape

        if seq_len == 0:
            return torch.zeros(batch_size, self.prediction_length, self.input_dim,
                               device=src.device, dtype=src.dtype)

        # 保存原始输入和掩码，用于谱偏置计算
        original_input = src.clone()
        input_mask = src_mask.clone()

        # 应用mask
        src_masked = src * src_mask

        # 处理MSSA特征并融合
        if self.use_mssa and mssa_features is not None:
            if mssa_features.size(1) != seq_len:
                if mssa_features.size(1) > seq_len:
                    mssa_features = mssa_features[:, :seq_len, :]
                else:
                    padding = torch.zeros(batch_size, seq_len - mssa_features.size(1),
                                          mssa_features.size(2), device=mssa_features.device)
                    mssa_features = torch.cat([mssa_features, padding], dim=1)

            processed_mssa, modal_attention_weights = self.mssa_processor(mssa_features)

            # 保存模态注意力权重
            if modal_attention_weights is not None:
                self.last_modal_attention_weights = modal_attention_weights.detach()

            if self.fusion_method == 'concat':
                fused_features = torch.cat([src_masked, processed_mssa], dim=-1)
            elif self.fusion_method == 'add':
                fused_features = src_masked + processed_mssa
            elif self.fusion_method == 'gate':
                gate_input = torch.cat([src_masked, processed_mssa], dim=-1)
                gate = self.gate_layer(gate_input)
                fused_features = gate * src_masked + (1 - gate) * processed_mssa
            else:
                fused_features = src_masked
        else:
            fused_features = src_masked

        if not self.use_transformer_encoder:
            last_features = fused_features[:, -1, :]
            pooled_features = fused_features.mean(dim=1)
            reconstruction_input = torch.cat(
                [last_features, pooled_features], dim=-1
            )
            output = self.reconstruction_head(reconstruction_input)
            return output.view(batch_size, self.prediction_length, self.input_dim)

        # 输入投影
        projected = self.input_projection(fused_features)

        # 位置编码
        projected = self.pos_encoder(projected)
        projected = self.dropout(projected)

        # 创建attention mask
        seq_mask = src_mask.any(dim=-1)
        attn_mask = ~seq_mask

        # 通过谱感知编码器层
        encoder_output = projected
        spectral_attention_weights = []

        for layer in self.encoder_layers:
            encoder_output, attn_weights = layer(
                encoder_output,
                original_input=original_input if self.use_spectral_bias else None,
                input_mask=input_mask if self.use_spectral_bias else None,
                src_key_padding_mask=attn_mask
            )
            spectral_attention_weights.append(attn_weights)

        # 保存谱注意力权重
        if spectral_attention_weights:
            self.last_spectral_attention_weights = [w.detach() for w in spectral_attention_weights]

        # 使用最后时间步的输出进行预测
        output = encoder_output[:, -1, :]

        # 输出投影
        output = self.output_projection(output)
        output = output.view(batch_size, self.prediction_length, self.input_dim)

        return output


class SpectralMSSATransformerImputer:
    """谱感知MSSA+Transformer缺失值填补器，结合自适应模态融合"""

    def __init__(self, mssa_window_length: int = 30,
                 mssa_n_components: Optional[int] = None,
                 transformer_config: Optional[Dict] = None,
                 mlp_config: Optional[Dict] = None,
                 spectral_config: Optional[Dict] = None,
                 fusion_method: str = 'concat',
                 use_modal_fusion: bool = True,
                 modal_fusion_type: str = 'adaptive',
                 use_spectral_bias: bool = True,
                 use_transformer_encoder: bool = True):
        """
        参数:
            mssa_window_length: MSSA窗口长度
            mssa_n_components: MSSA保留的分量数
            transformer_config: Transformer配置参数
            mlp_config: MLP配置参数
            spectral_config: 谱偏置配置参数
            fusion_method: 特征融合方法
            use_modal_fusion: 是否使用模态融合
            modal_fusion_type: 模态融合类型
            use_spectral_bias: 是否使用谱偏置
            use_transformer_encoder: 是否保留Transformer编码器
        """
        self.mssa_window_length = mssa_window_length
        self.mssa_n_components = mssa_n_components
        self.fusion_method = fusion_method
        self.use_modal_fusion = use_modal_fusion
        self.modal_fusion_type = modal_fusion_type
        self.use_transformer_encoder = use_transformer_encoder
        self.use_spectral_bias = use_spectral_bias and use_transformer_encoder

        # 默认Transformer配置
        self.transformer_config = {
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        }
        if transformer_config:
            self.transformer_config.update(transformer_config)

        # 默认MLP配置
        self.mlp_config = {
            'hidden_dims': [512, 256],
            'dropout': 0.1
        }
        if mlp_config:
            self.mlp_config.update(mlp_config)

        # 默认谱偏置配置
        self.spectral_config = {
            'max_seq_len': 512,
            'top_k_freq': 10,
            'learnable_weights': True,
            'temperature': 1.0
        }
        if spectral_config:
            self.spectral_config.update(spectral_config)

        self.scaler = StandardScaler()
        self.mssa_scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_mssa_components(self, data: np.ndarray, mssa_model) -> np.ndarray:
        """
        使用MSSA分解数据并准备用于处理的特征

        参数:
            data: 原始数据，形状为 (N, P)
            mssa_model: 已经fit的MSSA模型实例

        返回:
            分解后的数据，形状为 (N, P, K) 或 (N, P*K)
        """
        # 获取MSSA重构的分量
        groups = {}
        for i in range(mssa_model.n_components):
            groups[f'comp_{i}'] = [i]

        reconstructed = mssa_model.reconstruct(groups)  # (N, P, K)

        # 如果使用模态融合，保持3维形状；否则展平
        if self.use_modal_fusion and self.modal_fusion_type == 'adaptive':
            return reconstructed  # (N, P, K)
        else:
            N, P, n_comps = reconstructed.shape
            return reconstructed.transpose(0, 2, 1).reshape(N, -1)  # (N, P*K)

    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              mssa_model=None, epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001, sequence_length: int = 50,
              patience: int = 10, verbose: bool = True):
        """
        训练模型

        参数:
            train_data: 训练数据，形状为 (N, P)
            val_data: 验证数据，可选
            mssa_model: 已经fit的MSSA模型
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            sequence_length: 序列长度
            patience: 早停耐心值
            verbose: 是否打印训练信息
        """
        # 准备MSSA特征
        train_mssa_features = None
        val_mssa_features = None
        n_modes = None

        if mssa_model is not None:
            train_mssa_features = self.prepare_mssa_components(train_data, mssa_model)
            if val_data is not None:
                val_mssa_features = self.prepare_mssa_components(val_data, mssa_model)

            # 确定模态数量
            if len(train_mssa_features.shape) == 3:
                n_modes = train_mssa_features.shape[2]
            else:
                n_modes = train_mssa_features.shape[1] // train_data.shape[1]

        # 标准化原始数据
        train_data_scaled = self.scaler.fit_transform(
            train_data.reshape(-1, train_data.shape[-1])
        ).reshape(train_data.shape)

        # 标准化MSSA特征
        if train_mssa_features is not None:
            if len(train_mssa_features.shape) == 3:
                # 保持3维形状进行标准化
                shape_3d = train_mssa_features.shape
                train_mssa_flat = train_mssa_features.reshape(-1, shape_3d[1] * shape_3d[2])
                train_mssa_scaled = self.mssa_scaler.fit_transform(train_mssa_flat)
                train_mssa_scaled = train_mssa_scaled.reshape(shape_3d)
            else:
                train_mssa_scaled = self.mssa_scaler.fit_transform(
                    train_mssa_features.reshape(-1, train_mssa_features.shape[-1])
                ).reshape(train_mssa_features.shape)
        else:
            train_mssa_scaled = None

        # 创建数据集
        train_dataset = TimeSeriesDataset(
            train_data_scaled,
            train_mssa_scaled,
            sequence_length=sequence_length,
            prediction_length=self.transformer_config['prediction_length']
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        input_dim = train_data.shape[1]
        if train_mssa_features is not None:
            if len(train_mssa_features.shape) == 3:
                mssa_feature_dim = train_mssa_features.shape[1] * train_mssa_features.shape[2]
            else:
                mssa_feature_dim = train_mssa_features.shape[1]
        else:
            mssa_feature_dim = None

        self.model = SpectralAwareTransformerImputerMSSAMLP(
            input_dim=input_dim,
            mssa_feature_dim=mssa_feature_dim,
            n_modes=n_modes,
            mlp_hidden_dims=self.mlp_config['hidden_dims'],
            fusion_method=self.fusion_method,
            use_modal_fusion=self.use_modal_fusion,
            modal_fusion_type=self.modal_fusion_type,
            use_spectral_bias=self.use_spectral_bias,
            use_transformer_encoder=self.use_transformer_encoder,
            spectral_config=self.spectral_config,
            **self.transformer_config
        ).to(self.device)

        # 优化器和损失函数
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
                if len(batch) == 5:  # 有MSSA特征
                    src, src_mask, target, target_mask, mssa_features = [x.to(self.device) for x in batch]
                else:  # 没有MSSA特征
                    src, src_mask, target, target_mask = [x.to(self.device) for x in batch]
                    mssa_features = None

                optimizer.zero_grad()

                try:
                    output = self.model(src, src_mask, mssa_features)
                    loss = self._masked_mse_loss(output, target, target_mask)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()
                    epoch_loss += loss.item()
                except Exception as e:
                    if verbose:
                        print(f"训练批次出错: {e}")
                    continue

            if len(train_loader) > 0:
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # 验证阶段
                if val_data is not None:
                    val_loss = self._evaluate(val_data, val_mssa_features, mssa_model, sequence_length)
                    val_losses.append(val_loss)

                    scheduler.step(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        self.best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"早停在epoch {epoch}")
                        break

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}", end="")
                    if val_data is not None:
                        print(f", Val Loss = {val_loss:.4f}")
                    else:
                        print()

        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        return train_losses, val_losses

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

    def _evaluate(self, data: np.ndarray, mssa_features: Optional[np.ndarray],
                  mssa_model, sequence_length: int) -> float:
        """评估模型在验证集上的性能"""
        self.model.eval()

        # 标准化数据
        data_scaled = self.scaler.transform(
            data.reshape(-1, data.shape[-1])
        ).reshape(data.shape)

        mssa_scaled = None
        if mssa_features is not None:
            if len(mssa_features.shape) == 3:
                shape_3d = mssa_features.shape
                mssa_flat = mssa_features.reshape(-1, shape_3d[1] * shape_3d[2])
                mssa_scaled = self.mssa_scaler.transform(mssa_flat)
                mssa_scaled = mssa_scaled.reshape(shape_3d)
            else:
                mssa_scaled = self.mssa_scaler.transform(
                    mssa_features.reshape(-1, mssa_features.shape[-1])
                ).reshape(mssa_features.shape)

        dataset = TimeSeriesDataset(
            data_scaled,
            mssa_scaled,
            sequence_length=sequence_length,
            prediction_length=self.transformer_config['prediction_length']
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        total_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for batch in loader:
                try:
                    if len(batch) == 5:  # 有MSSA特征
                        src, src_mask, target, target_mask, mssa_batch = [x.to(self.device) for x in batch]
                    else:  # 没有MSSA特征
                        src, src_mask, target, target_mask = [x.to(self.device) for x in batch]
                        mssa_batch = None

                    output = self.model(src, src_mask, mssa_batch)
                    loss = self._masked_mse_loss(output, target, target_mask)
                    total_loss += loss.item()
                    valid_batches += 1
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue

        return total_loss / valid_batches if valid_batches > 0 else 0.0

    def impute_robust(self, data: np.ndarray, mssa_model=None,
                      iterations: int = 5, min_sequence_length: int = 10,
                      fallback_method: str = 'interpolate') -> np.ndarray:
        """
        稳健的缺失值填补方法

        参数:
            data: 带缺失值的数据，形状为 (N, P)
            mssa_model: MSSA模型实例
            iterations: 迭代填补次数
            min_sequence_length: 最小序列长度
            fallback_method: 后备填补方法 ('interpolate', 'forward_fill', 'mean')

        返回:
            填补后的数据
        """
        self.model.eval()
        imputed_data = data.copy()

        def fallback_imputation(data_temp, t, method=fallback_method):
            """后备填补方法"""
            if method == 'interpolate':
                df = pd.DataFrame(data_temp)
                interpolated = df.interpolate(method='linear', limit_direction='both')
                return interpolated.iloc[t].values
            elif method == 'forward_fill':
                for i in range(t, -1, -1):
                    if not np.any(np.isnan(data_temp[i])):
                        return data_temp[i]
                return np.nanmean(data_temp[:t + 1], axis=0)
            elif method == 'mean':
                return np.nanmean(data_temp[:t + 1], axis=0)
            else:
                return np.zeros(data_temp.shape[1])

        with torch.no_grad():
            for iteration in range(iterations):
                print(f"混合谱感知-模态融合填补迭代 {iteration + 1}/{iterations}")

                # 准备特征
                mssa_features = None
                if mssa_model is not None:
                    temp_data = pd.DataFrame(imputed_data).interpolate(
                        method='linear', limit_direction='both'
                    ).values
                    mssa_features = self.prepare_mssa_components(temp_data, mssa_model)

                    # 标准化MSSA特征
                    if len(mssa_features.shape) == 3:
                        shape_3d = mssa_features.shape
                        mssa_flat = mssa_features.reshape(-1, shape_3d[1] * shape_3d[2])
                        mssa_features_scaled = self.mssa_scaler.transform(mssa_flat)
                        mssa_features = mssa_features_scaled.reshape(shape_3d)
                    else:
                        mssa_features = self.mssa_scaler.transform(
                            mssa_features.reshape(-1, mssa_features.shape[-1])
                        ).reshape(mssa_features.shape)

                # 标准化原始数据
                data_scaled = self.scaler.transform(
                    imputed_data.reshape(-1, imputed_data.shape[-1])
                ).reshape(imputed_data.shape)

                # 统计填补情况
                model_imputed = 0
                fallback_imputed = 0

                # 逐步预测并填补
                for t in range(len(data) - 1):
                    if np.any(np.isnan(data[t + 1])):
                        missing_mask = np.isnan(data[t + 1])

                        # 检查是否有足够的历史数据
                        if t + 1 < min_sequence_length:
                            # 使用后备方法
                            fallback_pred = fallback_imputation(imputed_data, t + 1)
                            imputed_data[t + 1][missing_mask] = fallback_pred[missing_mask]
                            fallback_imputed += np.sum(missing_mask)
                            continue

                        # 准备序列数据
                        sequence_length = min(50, t + 1)
                        sequence_length = max(sequence_length, min_sequence_length)

                        start_idx = max(0, t + 1 - sequence_length)
                        seq = data_scaled[start_idx:t + 1]

                        # 处理MSSA特征序列
                        mssa_seq = None
                        if mssa_features is not None:
                            if start_idx < len(mssa_features):
                                mssa_end_idx = min(t + 1, len(mssa_features))
                                mssa_seq = mssa_features[start_idx:mssa_end_idx]

                                # 调整MSSA序列长度
                                if len(mssa_seq) < len(seq):
                                    pad_length = len(seq) - len(mssa_seq)
                                    if len(mssa_features.shape) == 3:
                                        mssa_seq = np.pad(mssa_seq, ((pad_length, 0), (0, 0), (0, 0)), mode='constant')
                                    else:
                                        mssa_seq = np.pad(mssa_seq, ((pad_length, 0), (0, 0)), mode='constant')
                                elif len(mssa_seq) > len(seq):
                                    mssa_seq = mssa_seq[:len(seq)]
                            else:
                                if len(mssa_features.shape) == 3:
                                    mssa_seq = np.zeros((len(seq), mssa_features.shape[1], mssa_features.shape[2]))
                                else:
                                    mssa_seq = np.zeros((len(seq), mssa_features.shape[1]))

                        # 填充到目标长度
                        target_length = 50
                        if len(seq) < target_length:
                            pad_length = target_length - len(seq)
                            seq = np.pad(seq, ((pad_length, 0), (0, 0)), mode='constant')
                            if mssa_seq is not None:
                                if len(mssa_seq.shape) == 3:
                                    mssa_seq = np.pad(mssa_seq, ((pad_length, 0), (0, 0), (0, 0)), mode='constant')
                                else:
                                    mssa_seq = np.pad(mssa_seq, ((pad_length, 0), (0, 0)), mode='constant')

                        # 创建mask
                        mask = ~np.isnan(seq)
                        seq = np.nan_to_num(seq, nan=0.0)
                        if mssa_seq is not None:
                            mssa_seq = np.nan_to_num(mssa_seq, nan=0.0)

                        # 转换为tensor
                        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                        mssa_tensor = torch.FloatTensor(mssa_seq).unsqueeze(0).to(
                            self.device) if mssa_seq is not None else None

                        try:
                            # 模型预测
                            pred = self.model(seq_tensor, mask_tensor, mssa_tensor)
                            pred = pred.squeeze(0).cpu().numpy()

                            # 反标准化
                            pred = self.scaler.inverse_transform(pred).squeeze()

                            # 填补缺失值
                            imputed_data[t + 1][missing_mask] = pred[missing_mask]
                            model_imputed += np.sum(missing_mask)

                        except Exception as e:
                            # 模型预测失败，使用后备方法
                            fallback_pred = fallback_imputation(imputed_data, t + 1)
                            imputed_data[t + 1][missing_mask] = fallback_pred[missing_mask]
                            fallback_imputed += np.sum(missing_mask)

                print(f"  混合模型填补: {model_imputed} 个值")
                print(f"  后备方法填补: {fallback_imputed} 个值")

        return imputed_data

    def evaluate_imputation(self, true_data: np.ndarray,
                            imputed_data: np.ndarray,
                            original_missing_mask: np.ndarray) -> Dict[str, float]:
        """
        评估填补效果

        参数:
            true_data: 真实完整数据
            imputed_data: 填补后的数据
            original_missing_mask: 原始缺失值位置的掩码（True表示缺失）

        返回:
            评估指标字典
        """
        # 只评估原本缺失位置的填补效果
        true_missing = true_data[original_missing_mask]
        imputed_missing = imputed_data[original_missing_mask]

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(true_missing, imputed_missing)),
            'MAE': mean_absolute_error(true_missing, imputed_missing),
            'MAPE': np.mean(np.abs((true_missing - imputed_missing) / (true_missing + 1e-8))) * 100,
            'R2': r2_score(true_missing, imputed_missing)
        }

        # 计算每个特征的评估指标
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

    def plot_imputation_results(self, true_data: np.ndarray,
                                imputed_data: np.ndarray,
                                original_missing_mask: np.ndarray,
                                feature_indices: Optional[List[int]] = None,
                                time_range: Optional[Tuple[int, int]] = None):
        """
        可视化填补结果

        参数:
            true_data: 真实数据
            imputed_data: 填补后的数据
            original_missing_mask: 缺失值掩码
            feature_indices: 要显示的特征索引
            time_range: 要显示的时间范围
        """
        if feature_indices is None:
            feature_indices = list(range(min(4, true_data.shape[1])))

        if time_range is None:
            time_range = (0, min(500, len(true_data)))

        n_features = len(feature_indices)
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 3 * n_features))
        if n_features == 1:
            axes = [axes]

        for idx, feat_idx in enumerate(feature_indices):
            ax = axes[idx]

            t_start, t_end = time_range
            time_points = np.arange(t_start, t_end)

            # 真实值
            ax.plot(time_points, true_data[t_start:t_end, feat_idx],
                    'b-', label='True', alpha=0.7, linewidth=1.5)

            # 填补值（只显示原本缺失的位置）
            missing_points = original_missing_mask[t_start:t_end, feat_idx]
            if np.any(missing_points):
                missing_indices = time_points[missing_points]
                ax.scatter(missing_indices,
                           imputed_data[t_start:t_end, feat_idx][missing_points],
                           c='red', s=25, label='Spectral+Modal Imputed', zorder=5, alpha=0.8)

            ax.set_title(f'Feature {feat_idx} (Spectral-Aware + Modal Fusion)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_spectral_analysis(self, data: np.ndarray, feature_idx: int = 0,
                               time_range: Optional[Tuple[int, int]] = None):
        """
        可视化谱分析结果

        参数:
            data: 数据
            feature_idx: 要分析的特征索引
            time_range: 时间范围
        """
        if time_range is None:
            time_range = (0, min(500, len(data)))

        t_start, t_end = time_range
        feature_data = data[t_start:t_end, feature_idx]

        # 计算功率谱
        fft_result = np.fft.rfft(feature_data)
        power_spectrum = np.abs(fft_result) ** 2
        freqs = np.fft.rfftfreq(len(feature_data))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 时域信号
        ax1.plot(np.arange(len(feature_data)), feature_data)
        ax1.set_title(f'Feature {feature_idx} - Time Domain')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)

        # 频域信号
        ax2.semilogy(freqs[1:], power_spectrum[1:])  # 排除DC分量
        ax2.set_title(f'Feature {feature_idx} - Power Spectrum')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Power')
        ax2.grid(True, alpha=0.3)

        # 标记主导频率
        top_k = 5
        dominant_indices = np.argsort(power_spectrum[1:])[::-1][:top_k] + 1
        ax2.scatter(freqs[dominant_indices], power_spectrum[dominant_indices],
                    c='red', s=50, zorder=5, label=f'Top {top_k} frequencies')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return freqs, power_spectrum

    def plot_modal_attention(self, n_samples: int = 5):
        """可视化模态注意力权重"""
        if self.model.last_modal_attention_weights is None:
            print("没有可用的模态注意力权重数据")
            return

        attention_weights = self.model.last_modal_attention_weights.cpu().numpy()

        # 选择样本进行可视化
        n_show = min(n_samples, attention_weights.shape[0])

        fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show))
        if n_show == 1:
            axes = [axes]

        for i in range(n_show):
            ax = axes[i]
            # 显示时间序列上的注意力权重
            im = ax.imshow(attention_weights[i].T, aspect='auto', cmap='hot')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('MSSA Mode')
            ax.set_title(f'Modal Attention Weights - Sample {i + 1}')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot_spectral_attention(self, layer_idx: int = 0, head_idx: int = 0, n_samples: int = 3):
        """可视化谱注意力权重"""
        if self.model.last_spectral_attention_weights is None:
            print("没有可用的谱注意力权重数据")
            return

        if layer_idx >= len(self.model.last_spectral_attention_weights):
            print(f"层索引 {layer_idx} 超出范围")
            return

        attention_weights = self.model.last_spectral_attention_weights[layer_idx].cpu().numpy()

        # 选择样本和头进行可视化
        n_show = min(n_samples, attention_weights.shape[0])

        fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show))
        if n_show == 1:
            axes = [axes]

        for i in range(n_show):
            ax = axes[i]
            # 显示特定头的注意力权重矩阵
            im = ax.imshow(attention_weights[i, head_idx], aspect='auto', cmap='Blues')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title(f'Spectral Attention - Layer {layer_idx}, Head {head_idx}, Sample {i + 1}')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {"error": "模型未训练"}

        info = {
            "model_type": (
                "Spectral-Aware + Modal Fusion Transformer"
                if self.use_transformer_encoder
                else "MSSA + Modal Fusion + Lightweight MLP Reconstruction Head"
            ),
            "use_spectral_bias": self.use_spectral_bias,
            "use_transformer_encoder": self.use_transformer_encoder,
            "use_modal_fusion": self.use_modal_fusion,
            "modal_fusion_type": self.modal_fusion_type,
            "fusion_method": self.fusion_method,
            "use_mssa": self.model.use_mssa,
            "spectral_config": self.spectral_config,
            "transformer_config": self.transformer_config,
            "mlp_config": self.mlp_config,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

        # 获取谱偏置权重信息
        if self.use_spectral_bias and hasattr(self.model, "encoder_layers"):
            spectral_weights = []
            for layer in self.model.encoder_layers:
                if hasattr(layer.self_attn.spectral_bias, 'freq_weights'):
                    weights = layer.self_attn.spectral_bias.freq_weights.data.cpu().numpy()
                    spectral_weights.append(weights)
            info["spectral_weights"] = spectral_weights

        # 获取模态融合权重信息
        if (self.use_modal_fusion and hasattr(self.model, "mssa_processor") and
                hasattr(self.model.mssa_processor, 'modal_fusion')):
            modal_fusion_params = {}
            modal_fusion = self.model.mssa_processor.modal_fusion
            if hasattr(modal_fusion, 'weight_net'):
                modal_fusion_params["weight_net_params"] = sum(p.numel() for p in modal_fusion.weight_net.parameters())
            if hasattr(modal_fusion, 'mode_projection'):
                modal_fusion_params["mode_projection_params"] = sum(
                    p.numel() for p in modal_fusion.mode_projection.parameters())
            info["modal_fusion_params"] = modal_fusion_params

        return info


# 使用示例
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. 创建示例数据
    N = 1000  # 时间步数
    P = 5  # 特征数
    K = 10  # MSSA模态数

    # 生成带有明显周期性和频率成分的示例数据
    t = np.arange(N)
    data = np.zeros((N, P))

    for i in range(P):
        # 添加多个频率成分
        trend = 0.01 * t + i
        seasonal1 = 2 * np.sin(2 * np.pi * t / 50 + i)  # 主要周期
        seasonal2 = 1 * np.sin(2 * np.pi * t / 20 + i * 0.5)  # 次要周期
        seasonal3 = 0.5 * np.sin(2 * np.pi * t / 100 + i * 0.3)  # 长周期
        noise = 0.3 * np.random.randn(N)
        data[:, i] = trend + seasonal1 + seasonal2 + seasonal3 + noise

    # 2. 创建缺失值
    missing_ratio = 0.2
    missing_mask = np.random.random((N, P)) < missing_ratio
    data_with_missing = data.copy()
    data_with_missing[missing_mask] = np.nan

    # 3. 创建混合谱感知+模态融合Transformer填补器
    imputer = SpectralMSSATransformerImputer(
        mssa_window_length=30,
        mssa_n_components=K,
        transformer_config={
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'prediction_length': 1
        },
        mlp_config={
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1
        },
        spectral_config={
            'max_seq_len': 512,
            'top_k_freq': 10,
            'learnable_weights': True,
            'temperature': 1.0
        },
        fusion_method='concat',  # 可选: 'concat', 'add', 'gate'
        use_modal_fusion=True,  # 启用模态融合
        modal_fusion_type='adaptive',  # 使用自适应模态融合
        use_spectral_bias=True  # 启用谱偏置
    )

    # 分割数据
    train_size = int(0.8 * N)
    train_data = data_with_missing[:train_size]
    val_data = data_with_missing[train_size:]

    # 训练模型
    print("开始训练混合谱感知+模态融合Transformer模型...")
    train_losses, val_losses = imputer.train(
        train_data,
        val_data,
        mssa_model=None,  # 如果有MSSA模型，在这里传入
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=50
    )

    # 4. 填补缺失值
    print("\n使用混合模型填补缺失值...")
    imputed_data = imputer.impute_robust(data_with_missing, mssa_model=None)

    # 5. 评估填补效果
    print("\n评估混合模型填补效果...")
    overall_metrics, feature_metrics = imputer.evaluate_imputation(
        data, imputed_data, missing_mask
    )

    print("\n整体填补指标 (Spectral-Aware + Modal Fusion):")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n各特征填补指标:")
    for feat, metrics in feature_metrics.items():
        print(f"\n{feat}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 6. 可视化结果
    print("\n可视化填补结果...")
    imputer.plot_imputation_results(
        data, imputed_data, missing_mask,
        feature_indices=[0, 1, 2],
        time_range=(0, 200)
    )

    # 7. 可视化谱分析
    print("\n可视化谱分析...")
    freqs, power_spectrum = imputer.plot_spectral_analysis(data, feature_idx=0, time_range=(0, 300))

    # 8. 可视化模态注意力权重
    print("\n可视化模态注意力权重...")
    imputer.plot_modal_attention(n_samples=3)

    # 9. 可视化谱注意力权重
    print("\n可视化谱注意力权重...")
    imputer.plot_spectral_attention(layer_idx=0, head_idx=0, n_samples=3)

    # 10. 打印模型架构信息
    print("\n混合模型架构信息:")
    model_info = imputer.get_model_info()
    for key, value in model_info.items():
        if key not in ["spectral_weights", "modal_fusion_params"]:
            print(f"{key}: {value}")

    if "spectral_weights" in model_info:
        print(f"\n谱偏置权重层数: {len(model_info['spectral_weights'])}")
        for i, weights in enumerate(model_info['spectral_weights']):
            print(f"  层 {i + 1} 权重前5个: {weights[:5]}")

    if "modal_fusion_params" in model_info:
        print(f"\n模态融合参数:")
        for key, value in model_info["modal_fusion_params"].items():
            print(f"  {key}: {value:,}")

    # 11. 训练损失可视化
    if train_losses and val_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
        plt.title('Spectral-Aware + Modal Fusion Model Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 12. 对比分析
    print("\n=== 模型特性总结 ===")
    print(f"✓ 谱感知注意力偏置: {imputer.use_spectral_bias}")
    print(f"✓ 自适应模态融合: {imputer.use_modal_fusion}")
    print(f"✓ 融合策略: {imputer.fusion_method}")
    print(f"✓ 模态融合类型: {imputer.modal_fusion_type}")
    print(f"✓ MSSA模态数: {K}")
    print(f"✓ 谱偏置主导频率数: {imputer.spectral_config['top_k_freq']}")
    print(f"✓ 总参数数量: {sum(p.numel() for p in imputer.model.parameters()):,}")
    print(f"✓ 可训练参数数量: {sum(p.numel() for p in imputer.model.parameters() if p.requires_grad):,}")

    print("\n混合谱感知+模态融合Transformer时间序列填补演示完成!")
