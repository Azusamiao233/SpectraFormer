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


class LearnableModalFusion(nn.Module):
    """可学习的模态融合模块
    
    实现基于注意力机制的模态加权融合，根据输入动态调整各模态的权重
    """
    
    def __init__(self, n_modes: int, feature_dim: int, hidden_dim: int = 128):
        """
        参数:
            n_modes: 模态数量 (MSSA分量数K)
            feature_dim: 特征维度 (P)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.n_modes = n_modes
        self.feature_dim = feature_dim
        
        # 注意力网络 - 生成模态权重
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim * n_modes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modes),
            nn.Softmax(dim=-1)  # 确保权重和为1
        )
        
        # 可选：学习一个全局的模态重要性偏置
        self.mode_bias = nn.Parameter(torch.zeros(n_modes))
        
    def forward(self, mssa_features, original_features=None):
        """
        参数:
            mssa_features: MSSA分解特征，形状为 (batch_size, seq_len, P, K) 或 (batch_size, seq_len, P*K)
            original_features: 原始特征，可选，形状为 (batch_size, seq_len, P)
            
        返回:
            fused_features: 融合后的特征，形状为 (batch_size, seq_len, P)
        """
        batch_size, seq_len = mssa_features.shape[:2]
        
        # 处理输入维度
        if len(mssa_features.shape) == 3:  # (batch_size, seq_len, P*K)
            # 重塑为 (batch_size, seq_len, P, K)
            mssa_features = mssa_features.view(batch_size, seq_len, self.feature_dim, self.n_modes)
        
        # 计算注意力权重
        # 将特征展平以输入注意力网络
        features_flat = mssa_features.view(batch_size * seq_len, -1)  # (batch*seq, P*K)
        
        # 通过注意力网络得到权重
        attention_weights = self.attention_net(features_flat)  # (batch*seq, K)
        
        # 添加模态偏置
        attention_weights = attention_weights + self.mode_bias
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 重塑权重
        attention_weights = attention_weights.view(batch_size, seq_len, self.n_modes, 1)  # (batch, seq, K, 1)
        
        # 应用权重进行融合
        # mssa_features: (batch, seq, P, K)
        # attention_weights: (batch, seq, K, 1)
        weighted_features = mssa_features * attention_weights.transpose(-2, -1)  # (batch, seq, P, K)
        
        # 对模态维度求和得到最终融合特征
        fused_features = weighted_features.sum(dim=-1)  # (batch, seq, P)
        
        return fused_features, attention_weights.squeeze(-1)


class AdaptiveModalFusion(nn.Module):
    """自适应模态融合模块 - 实现论文中的公式
    
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
        实现论文中的公式：
        h_i = φ(s_i) = ReLU(W s_i + b)
        α_i = exp(w^T h_i) / Σ_j exp(w^T h_j)
        Z = Σ_i α_i S_i
        
        参数:
            Z: MSSA分解特征，形状为 (batch_size, seq_len, P, K)
            
        返回:
            fused_features: 融合后的特征，形状为 (batch_size, seq_len, P)
            attention_weights: 注意力权重，形状为 (batch_size, seq_len, K)
        """
        batch_size, seq_len, P, K = Z.shape
        
        # 1. 将每个模态投影到表示空间 (相当于 φ(s_i))
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


class MSSAFeatureProcessor(nn.Module):
    """MSSA特征处理器 - 集成可学习的模态融合"""

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
            fusion_type: 融合类型 ('adaptive', 'learnable', 'mlp')
        """
        super().__init__()
        self.mssa_feature_dim = mssa_feature_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.use_modal_fusion = use_modal_fusion
        self.fusion_type = fusion_type
        
        if use_modal_fusion and fusion_type != 'mlp':
            # 使用模态融合时，先融合再通过较小的MLP
            if fusion_type == 'adaptive':
                self.modal_fusion = AdaptiveModalFusion(n_modes, output_dim)
            else:  # 'learnable'
                self.modal_fusion = LearnableModalFusion(n_modes, output_dim)
            
            # 融合后只需要一个较小的MLP进行微调
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
        
        if self.use_modal_fusion and self.fusion_type != 'mlp':
            # 确保输入是4维的 (batch, seq, P, K)
            if len(mssa_features.shape) == 3:
                mssa_features = mssa_features.view(batch_size, seq_len, self.output_dim, self.n_modes)
            
            # 应用模态融合
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


class TransformerImputerMSSAMLPFixed(nn.Module):
    """结合MSSA+可学习模态融合+Transformer的缺失值填补模型"""

    def __init__(self, input_dim: int, mssa_feature_dim: Optional[int] = None,
                 n_modes: Optional[int] = None, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 prediction_length: int = 1, mlp_hidden_dims: List[int] = [512, 256],
                 fusion_method: str = 'concat', use_modal_fusion: bool = True,
                 modal_fusion_type: str = 'adaptive'):
        """
        参数:
            input_dim: 原始输入维度 (P)
            mssa_feature_dim: MSSA特征维度 (P * K)
            n_modes: MSSA模态数量 (K)
            d_model: transformer模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: dropout比例
            prediction_length: 预测长度
            mlp_hidden_dims: MLP隐藏层维度
            fusion_method: 特征融合方法 ('concat', 'add', 'gate')
            use_modal_fusion: 是否使用可学习的模态融合
            modal_fusion_type: 模态融合类型 ('adaptive', 'learnable', 'mlp')
        """
        super().__init__()

        self.input_dim = input_dim
        self.mssa_feature_dim = mssa_feature_dim
        self.n_modes = n_modes
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.fusion_method = fusion_method
        self.use_mssa = mssa_feature_dim is not None
        self.use_modal_fusion = use_modal_fusion

        # MSSA特征处理器
        if self.use_mssa:
            self.mssa_processor = MSSAFeatureProcessor(
                mssa_feature_dim, input_dim, n_modes or (mssa_feature_dim // input_dim),
                mlp_hidden_dims, dropout, use_modal_fusion, modal_fusion_type
            )

            # 特征融合层
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

        # 输入投影层
        self.input_projection = nn.Linear(fusion_input_dim, d_model)

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
        self.output_projection = nn.Linear(d_model, input_dim * prediction_length)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 存储注意力权重用于分析
        self.last_attention_weights = None

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor,
                mssa_features: Optional[torch.Tensor] = None):
        """
        参数:
            src: 原始输入序列，形状为 (batch_size, seq_len, input_dim)
            src_mask: 输入掩码，形状为 (batch_size, seq_len, input_dim)
            mssa_features: MSSA特征，形状为 (batch_size, seq_len, mssa_feature_dim)

        返回:
            输出预测，形状为 (batch_size, prediction_length, input_dim)
        """
        batch_size, seq_len, _ = src.shape

        if seq_len == 0:
            return torch.zeros(batch_size, self.prediction_length, self.input_dim,
                               device=src.device, dtype=src.dtype)

        # 应用mask
        src_masked = src * src_mask

        # 处理MSSA特征并融合
        if self.use_mssa and mssa_features is not None:
            # 调整MSSA特征维度
            if mssa_features.size(1) != seq_len:
                if mssa_features.size(1) > seq_len:
                    mssa_features = mssa_features[:, :seq_len, :]
                else:
                    padding = torch.zeros(batch_size, seq_len - mssa_features.size(1),
                                          mssa_features.size(2), device=mssa_features.device)
                    mssa_features = torch.cat([mssa_features, padding], dim=1)

            # 通过处理器处理MSSA特征（包括模态融合）
            processed_mssa, attention_weights = self.mssa_processor(mssa_features)
            
            # 保存注意力权重用于分析
            if attention_weights is not None:
                self.last_attention_weights = attention_weights.detach()

            # 特征融合
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

        # 输入投影
        projected = self.input_projection(fused_features)

        # 位置编码
        projected = self.pos_encoder(projected)
        projected = self.dropout(projected)

        # 创建attention mask
        seq_mask = src_mask.any(dim=-1)
        attn_mask = ~seq_mask

        # Transformer编码
        memory = self.transformer_encoder(projected, src_key_padding_mask=attn_mask)

        # 使用最后的输出进行预测
        output = memory[:, -1, :]

        # 输出投影
        output = self.output_projection(output)
        output = output.view(batch_size, self.prediction_length, self.input_dim)

        return output


class MSSATransformerImputerMLPFixed:
    """结合MSSA、可学习模态融合和Transformer的缺失值填补器"""

    def __init__(self, mssa_window_length: int = 30,
                 mssa_n_components: Optional[int] = None,
                 transformer_config: Optional[Dict] = None,
                 mlp_config: Optional[Dict] = None,
                 fusion_method: str = 'concat',
                 use_modal_fusion: bool = True,
                 modal_fusion_type: str = 'adaptive'):
        """
        参数:
            mssa_window_length: MSSA窗口长度
            mssa_n_components: MSSA保留的分量数
            transformer_config: Transformer配置参数
            mlp_config: MLP配置参数
            fusion_method: 特征融合方法
            use_modal_fusion: 是否使用可学习的模态融合
            modal_fusion_type: 模态融合类型
        """
        self.mssa_window_length = mssa_window_length
        self.mssa_n_components = mssa_n_components
        self.fusion_method = fusion_method
        self.use_modal_fusion = use_modal_fusion
        self.modal_fusion_type = modal_fusion_type

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

        # 默认MLP配置
        self.mlp_config = {
            'hidden_dims': [512, 256],
            'dropout': 0.1
        }
        if mlp_config:
            self.mlp_config.update(mlp_config)

        self.scaler = StandardScaler()
        self.mssa_scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_mssa_components(self, data: np.ndarray, mssa_model) -> np.ndarray:
        """
        使用MSSA分解数据并准备特征
        
        返回:
            分解后的数据，形状为 (N, P*K) 或 (N, P, K)
        """
        # 获取MSSA重构的分量
        groups = {}
        for i in range(mssa_model.n_components):
            groups[f'comp_{i}'] = [i]

        reconstructed = mssa_model.reconstruct(groups)  # (N, P, K)
        
        # 如果使用模态融合，保持3维形状；否则展平
        if self.use_modal_fusion and self.modal_fusion_type != 'mlp':
            return reconstructed  # (N, P, K)
        else:
            N, P, n_comps = reconstructed.shape
            return reconstructed.transpose(0, 2, 1).reshape(N, -1)  # (N, P*K)

    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              mssa_model=None, epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001, sequence_length: int = 50,
              patience: int = 10, verbose: bool = True):
        """训练模型"""
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

        self.model = TransformerImputerMSSAMLPFixed(
            input_dim=input_dim,
            mssa_feature_dim=mssa_feature_dim,
            n_modes=n_modes,
            mlp_hidden_dims=self.mlp_config['hidden_dims'],
            fusion_method=self.fusion_method,
            use_modal_fusion=self.use_modal_fusion,
            modal_fusion_type=self.modal_fusion_type,
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
                    optimizer.step()
                    epoch_loss += loss.item()
                except Exception as e:
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

    def impute(self, data: np.ndarray, mssa_model=None,
               iterations: int = 5, min_sequence_length: int = 10) -> np.ndarray:
        """使用训练好的模型填补缺失值"""
        self.model.eval()
        imputed_data = data.copy()

        with torch.no_grad():
            for iteration in range(iterations):
                # 准备特征
                mssa_features = None
                if mssa_model is not None:
                    # 对于有缺失值的数据，先用简单插值填充以进行MSSA
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

                # 逐步预测并填补
                for t in range(min_sequence_length, len(data) - 1):
                    # 检查下一个时间步是否有缺失值
                    if np.any(np.isnan(data[t + 1])):
                        # 准备输入序列
                        sequence_length = min(50, t + 1)
                        sequence_length = max(sequence_length, min_sequence_length)

                        start_idx = max(0, t + 1 - sequence_length)
                        seq = data_scaled[start_idx:t + 1]

                        # 处理MSSA特征序列
                        mssa_seq = None
                        if mssa_features is not None:
                            mssa_seq = mssa_features[start_idx:t + 1]

                            # 调整序列长度
                            if len(mssa_seq) != len(seq):
                                if len(mssa_seq) == 0:
                                    if len(mssa_features.shape) == 3:
                                        mssa_seq = np.zeros((len(seq), mssa_features.shape[1], mssa_features.shape[2]))
                                    else:
                                        mssa_seq = np.zeros((len(seq), mssa_features.shape[1]))
                                elif len(mssa_seq) < len(seq):
                                    pad_length = len(seq) - len(mssa_seq)
                                    if len(mssa_features.shape) == 3:
                                        mssa_seq = np.pad(mssa_seq, ((pad_length, 0), (0, 0), (0, 0)), mode='constant')
                                    else:
                                        mssa_seq = np.pad(mssa_seq, ((pad_length, 0), (0, 0)), mode='constant')
                                else:
                                    mssa_seq = mssa_seq[:len(seq)]

                        # 确保序列长度符合要求
                        if len(seq) < min_sequence_length:
                            continue

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
                            # 预测
                            pred = self.model(seq_tensor, mask_tensor, mssa_tensor)
                            pred = pred.squeeze(0).cpu().numpy()

                            # 反标准化
                            pred = self.scaler.inverse_transform(pred).squeeze()

                            # 填补缺失值
                            missing_mask = np.isnan(data[t + 1])
                            imputed_data[t + 1][missing_mask] = pred[missing_mask]
                        except Exception as e:
                            print(f"预测时出错 (t={t}): {e}")
                            continue

        return imputed_data

    def evaluate_imputation(self, true_data: np.ndarray,
                            imputed_data: np.ndarray,
                            original_missing_mask: np.ndarray) -> Dict[str, float]:
        """评估填补效果"""
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

            # 填补值（只显示原本缺失的位置）
            missing_points = original_missing_mask[t_start:t_end, feat_idx]
            if np.any(missing_points):
                missing_indices = time_points[missing_points]
                ax.scatter(missing_indices,
                           imputed_data[t_start:t_end, feat_idx][missing_points],
                           c='red', s=20, label='Imputed (Modal Fusion)', zorder=5)

            ax.set_title(f'Feature {feat_idx} (MSSA+Modal Fusion+Transformer)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_modal_attention(self, n_samples: int = 5):
        """可视化模态注意力权重"""
        if self.model.last_attention_weights is None:
            print("没有可用的注意力权重数据")
            return
        
        attention_weights = self.model.last_attention_weights.cpu().numpy()
        
        # 选择样本进行可视化
        n_show = min(n_samples, attention_weights.shape[0])
        
        fig, axes = plt.subplots(n_show, 1, figsize=(10, 3 * n_show))
        if n_show == 1:
            axes = [axes]
        
        for i in range(n_show):
            ax = axes[i]
            # 显示时间序列上的注意力权重
            im = ax.imshow(attention_weights[i].T, aspect='auto', cmap='hot')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Mode')
            ax.set_title(f'Modal Attention Weights - Sample {i+1}')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    np.random.seed(42)

    # 1. 创建示例数据
    N = 1000  # 时间步数
    P = 5     # 特征数
    K = 10    # MSSA模态数

    # 生成示例数据
    t = np.arange(N)
    data = np.zeros((N, P))
    for i in range(P):
        trend = 0.01 * t + i
        seasonal = 2 * np.sin(2 * np.pi * t / 50 + i)
        noise = 0.5 * np.random.randn(N)
        data[:, i] = trend + seasonal + noise

    # 2. 创建缺失值
    missing_ratio = 0.2
    missing_mask = np.random.random((N, P)) < missing_ratio
    data_with_missing = data.copy()
    data_with_missing[missing_mask] = np.nan

    # 3. 创建带模态融合的Transformer填补器
    imputer = MSSATransformerImputerMLPFixed(
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
            'hidden_dims': [512, 256],
            'dropout': 0.1
        },
        fusion_method='concat',
        use_modal_fusion=True,      # 启用模态融合
        modal_fusion_type='adaptive' # 使用自适应融合
    )

    # 分割数据
    train_size = int(0.8 * N)
    train_data = data_with_missing[:train_size]
    val_data = data_with_missing[train_size:]

    # 训练模型
    print("开始训练带可学习模态融合的模型...")
    train_losses, val_losses = imputer.train(
        train_data,
        val_data,
        mssa_model=None,  # 这里应该传入真实的MSSA模型
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=50
    )

    # 4. 填补缺失值
    print("\n填补缺失值...")
    imputed_data = imputer.impute(data_with_missing, mssa_model=None)

    # 5. 评估填补效果
    print("\n评估填补效果...")
    overall_metrics, feature_metrics = imputer.evaluate_imputation(
        data, imputed_data, missing_mask
    )

    print("\n整体填补指标 (可学习模态融合):")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 6. 可视化结果
    imputer.plot_imputation_results(
        data, imputed_data, missing_mask,
        feature_indices=[0, 1, 2],
        time_range=(0, 200)
    )

    # 7. 可视化模态注意力权重（如果可用）
    imputer.plot_modal_attention(n_samples=3)

    # 8. 打印模型架构信息
    print("\n模型架构信息:")
    print(f"使用模态融合: {imputer.use_modal_fusion}")
    print(f"模态融合类型: {imputer.modal_fusion_type}")
    print(f"融合方法: {imputer.fusion_method}")
    print(f"MSSA模态数: {K}")
    print(f"总参数数量: {sum(p.numel() for p in imputer.model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in imputer.model.parameters() if p.requires_grad):,}")