import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional


class ContinuousTimeEmbedding(nn.Module):
    """连续时间编码 - 处理不规则时间间隔"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w = nn.Parameter(torch.randn(d_model))
        self.alpha = nn.Parameter(torch.randn(d_model))

    def forward(self, timestamps):
        time_emb = torch.zeros(*timestamps.shape, self.d_model, device=timestamps.device)
        time_emb[..., 0] = self.w[0] * timestamps + self.alpha[0]
        for d in range(1, self.d_model):
            time_emb[..., d] = torch.sin(self.w[d] * timestamps + self.alpha[d])
        return time_emb


class ObservationEncoder(nn.Module):
    """观测编码器"""

    def __init__(self, num_variables: int, d_model: int):
        super().__init__()
        self.time_encoder = ContinuousTimeEmbedding(d_model)
        self.variable_embedding = nn.Embedding(num_variables, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.activation = nn.ReLU()

    def forward(self, timestamps, values, variables):
        time_emb = self.time_encoder(timestamps)
        var_emb = self.variable_embedding(variables)
        val_emb = self.value_projection(values.unsqueeze(-1))
        node_emb = self.activation(time_emb + var_emb + val_emb)
        return node_emb


class GraphDeviationLayer(nn.Module):
    """
    GDN图偏差层 - 核心创新
    学习节点间的关系，并通过偏差检测异常
    """

    def __init__(self, d_model: int, num_edge_types: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_edge_types = num_edge_types

        # 嵌入层：为每种边类型学习不同的特征转换
        self.embedding_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_edge_types)
        ])

        # 图学习层：自适应学习边权重
        self.graph_learning = nn.ModuleDict({
            f'type_{i}': nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            ) for i in range(num_edge_types)
        })

        # 偏差计算层
        self.deviation_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, node_emb, edge_type_mask):
        """
        Args:
            node_emb: [B, N, d_model]
            edge_type_mask: [B, N, N, 3] - 三种边类型的mask
        Returns:
            updated_emb: [B, N, d_model]
            graph_deviation: [B, N] - 图偏差分数
        """
        B, N, d_model = node_emb.shape

        # 存储三种边类型的聚合特征
        aggregated_features = []
        deviation_scores = []

        for edge_type in range(self.num_edge_types):
            # 1. 特征嵌入
            embedded = self.embedding_layers[edge_type](node_emb)  # [B, N, d_model]

            # 2. 自适应图学习 - 学习邻接矩阵
            adjacency = self._learn_adjacency(
                node_emb, edge_type, edge_type_mask[..., edge_type]
            )  # [B, N, N]

            # 3. 图卷积 - 聚合邻居信息
            neighbor_features = torch.bmm(adjacency, embedded)  # [B, N, d_model]

            # 4. 计算偏差 - 当前节点与邻居的差异
            deviation = self._compute_deviation(
                node_emb, neighbor_features
            )  # [B, N]

            aggregated_features.append(neighbor_features)
            deviation_scores.append(deviation)

        # 5. 融合三种边类型的信息
        combined_features = torch.stack(aggregated_features, dim=0).sum(0)  # [B, N, d_model]
        combined_deviation = torch.stack(deviation_scores, dim=0).mean(0)  # [B, N]

        # 6. 使用注意力更新节点表示
        updated_emb, _ = self.attention(
            node_emb, combined_features, combined_features
        )

        # 7. 残差连接
        updated_emb = node_emb + updated_emb

        return updated_emb, combined_deviation

    def _learn_adjacency(self, node_emb, edge_type, edge_mask):
        """自适应学习邻接矩阵"""
        B, N, d_model = node_emb.shape

        # 计算节点对之间的相似度
        node_i = node_emb.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, d_model]
        node_j = node_emb.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, d_model]

        # 拼接特征
        pair_features = torch.cat([node_i, node_j], dim=-1)  # [B, N, N, 2*d_model]

        # 学习边权重
        edge_weights = self.graph_learning[f'type_{edge_type}'](
            pair_features
        ).squeeze(-1)  # [B, N, N]

        # 应用edge mask（只保留允许的边）
        edge_weights = edge_weights * edge_mask

        # 归一化（按行）
        row_sum = edge_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        adjacency = edge_weights / row_sum

        return adjacency

    def _compute_deviation(self, node_features, neighbor_features):
        """计算节点与邻居的偏差"""
        # 拼接节点特征和邻居聚合特征
        combined = torch.cat([node_features, neighbor_features], dim=-1)

        # 通过MLP计算偏差分数
        deviation = self.deviation_net(combined).squeeze(-1)  # [B, N]

        return torch.abs(deviation)


class IntraPatchGDNLayer(nn.Module):
    """Patch内部GDN层 - 捕获局部异常"""

    def __init__(self, d_model: int, num_layers: int = 2):
        super().__init__()
        self.gdn_layers = nn.ModuleList([
            GraphDeviationLayer(d_model) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, node_emb, patch_indices, timestamps, variables):
        """
        Args:
            node_emb: [B, N, d_model]
            patch_indices: [B, N]
            timestamps: [B, N]
            variables: [B, N]
        Returns:
            updated_emb: [B, N, d_model]
            deviation_scores: [B, N] - 累积偏差分数
        """
        B, N, d_model = node_emb.shape

        # 构建三种边类型的mask
        edge_type_mask = self._build_edge_masks(
            patch_indices, timestamps, variables, B, N
        )

        # 多层GDN处理
        total_deviation = torch.zeros(B, N, device=node_emb.device)

        for gdn_layer, layer_norm in zip(self.gdn_layers, self.layer_norms):
            updated, deviation = gdn_layer(node_emb, edge_type_mask)
            node_emb = layer_norm(updated)
            total_deviation += deviation

        # 平均偏差分数
        avg_deviation = total_deviation / len(self.gdn_layers)

        return node_emb, avg_deviation

    def _build_edge_masks(self, patch_indices, timestamps, variables, B, N):
        """构建三种边类型的mask"""
        edge_type_mask = torch.zeros(B, N, N, 3, device=patch_indices.device)

        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue

                    # 只在同一patch内连接
                    if patch_indices[b, i] != patch_indices[b, j]:
                        continue

                    same_var = (variables[b, i] == variables[b, j]).item()
                    same_time = (torch.abs(timestamps[b, i] - timestamps[b, j]) < 1e-5).item()

                    # SVDT: 同变量不同时间
                    if same_var and not same_time:
                        edge_type_mask[b, i, j, 0] = 1
                    # DVST: 不同变量同时间
                    elif not same_var and same_time:
                        edge_type_mask[b, i, j, 1] = 1
                    # DVDT: 不同变量不同时间
                    elif not same_var and not same_time:
                        edge_type_mask[b, i, j, 2] = 1

        return edge_type_mask


class InterPatchGDNLayer(nn.Module):
    """Patch间GDN层 - 捕获全局异常"""

    def __init__(self, d_model: int):
        super().__init__()
        self.gdn_layer = GraphDeviationLayer(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # 时间注意力聚合
        self.time_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, patch_nodes, patch_masks):
        """
        Args:
            patch_nodes: [B, num_patches, num_vars, d_model]
            patch_masks: [B, num_patches, num_vars]
        Returns:
            aggregated_nodes: [B, num_patches//2, num_vars, d_model]
            aggregated_masks: [B, num_patches//2, num_vars]
            deviation_scores: [B, num_patches, num_vars]
        """
        B, P, V, d_model = patch_nodes.shape

        # Reshape为 [B, P*V, d_model]
        nodes_flat = patch_nodes.view(B, P * V, d_model)

        # 构建patch间边的mask
        edge_mask = self._build_inter_patch_mask(patch_masks, P, V)

        # GDN处理
        updated, deviation = self.gdn_layer(nodes_flat, edge_mask)
        nodes_flat = self.layer_norm(updated)

        # Reshape回来
        nodes = nodes_flat.view(B, P, V, d_model)
        deviation = deviation.view(B, P, V)

        # 聚合相邻patch
        aggregated, agg_masks = self._aggregate_adjacent_patches(nodes, patch_masks)

        return aggregated, agg_masks, deviation

    def _build_inter_patch_mask(self, patch_masks, P, V):
        """构建patch间的边mask"""
        B = patch_masks.shape[0]
        N = P * V
        edge_mask = torch.zeros(B, N, N, 3, device=patch_masks.device)

        for b in range(B):
            for p in range(P - 1):
                for v1 in range(V):
                    for v2 in range(V):
                        i = p * V + v1
                        j = (p + 1) * V + v2

                        if patch_masks[b, p, v1] and patch_masks[b, p + 1, v2]:
                            if v1 == v2:
                                edge_mask[b, i, j, 0] = 1
                                edge_mask[b, j, i, 0] = 1
                            else:
                                edge_mask[b, i, j, 2] = 1
                                edge_mask[b, j, i, 2] = 1

        return edge_mask

    def _aggregate_adjacent_patches(self, nodes, masks):
        """聚合相邻的两个patch"""
        B, P, V, d_model = nodes.shape
        new_P = P // 2

        aggregated = torch.zeros(B, new_P, V, d_model, device=nodes.device)
        new_masks = torch.zeros(B, new_P, V, dtype=torch.bool, device=nodes.device)

        for p in range(new_P):
            p1, p2 = 2 * p, 2 * p + 1
            for v in range(V):
                for b in range(B):
                    has_p1 = masks[b, p1, v] if p1 < P else False
                    has_p2 = masks[b, p2, v] if p2 < P else False

                    if has_p1 and has_p2:
                        aggregated[b, p, v] = (nodes[b, p1, v] + nodes[b, p2, v]) / 2
                        new_masks[b, p, v] = True
                    elif has_p1:
                        aggregated[b, p, v] = nodes[b, p1, v]
                        new_masks[b, p, v] = True
                    elif has_p2:
                        aggregated[b, p, v] = nodes[b, p2, v]
                        new_masks[b, p, v] = True

        return aggregated, new_masks


class GDNAnomalyDetectionHead(nn.Module):
    """GDN异常检测头 - 结合图偏差和预测误差"""

    def __init__(self, d_model: int, num_variables: int):
        super().__init__()

        # 1. 预测分支（forecast-based detection）
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # 2. 重构分支（reconstruction-based detection）
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # 3. 图偏差注意力权重
        self.deviation_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings, graph_deviations, values):
        """
        Args:
            node_embeddings: [B, N, d_model]
            graph_deviations: [B, N] - 来自GDN层的偏差分数
            values: [B, N] - 真实观测值
        Returns:
            anomaly_scores: dict
        """
        # 1. 预测异常分数
        forecast_values = self.forecast_head(node_embeddings).squeeze(-1)
        forecast_error = torch.abs(forecast_values - values)

        # 2. 重构异常分数
        recon_values = self.reconstruction_head(node_embeddings).squeeze(-1)
        recon_error = torch.abs(recon_values - values)

        # 3. 图偏差权重（自适应）
        deviation_weights = self.deviation_attention(node_embeddings).squeeze(-1)

        # 4. 加权图偏差
        weighted_graph_deviation = graph_deviations * deviation_weights

        return {
            'forecast': forecast_error,
            'reconstruction': recon_error,
            'graph_deviation': weighted_graph_deviation,
            'forecast_values': forecast_values,
            'recon_values': recon_values
        }


class GADIMTS_GDN(nn.Module):
    """GAD-IMTS with GDN - 完整模型"""

    def __init__(
            self,
            num_variables: int,
            d_model: int = 64,
            num_intra_layers: int = 2,
            num_inter_layers: int = 2,
            patch_size: float = 10.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_variables = num_variables

        # 编码器
        self.obs_encoder = ObservationEncoder(num_variables, d_model)

        # GDN局部图层
        self.intra_patch_gdn = IntraPatchGDNLayer(d_model, num_intra_layers)

        # GDN全局图层
        self.inter_patch_gdns = nn.ModuleList([
            InterPatchGDNLayer(d_model)
            for _ in range(num_inter_layers)
        ])

        # GDN异常检测头
        self.anomaly_head = GDNAnomalyDetectionHead(d_model, num_variables)

    def forward(self, timestamps, values, variables, obs_mask=None):
        """
        Args:
            timestamps: [B, N]
            values: [B, N]
            variables: [B, N]
            obs_mask: [B, N]
        Returns:
            anomaly_scores: dict
            node_embeddings: [B, N, d_model]
        """
        if obs_mask is None:
            obs_mask = torch.ones_like(timestamps, dtype=torch.bool)

        B, N = timestamps.shape

        # 1. 观测编码
        node_embeddings = self.obs_encoder(timestamps, values, variables)

        # 2. 分配patch
        patch_indices = (timestamps / self.patch_size).long()

        # 3. Intra-patch GDN - 捕获局部偏差
        node_embeddings, local_deviation = self.intra_patch_gdn(
            node_embeddings, patch_indices, timestamps, variables
        )

        # 4. 转换为patch级表示
        patch_nodes, patch_masks = self._nodes_to_patches(
            node_embeddings, patch_indices, variables, obs_mask
        )

        # 5. Inter-patch GDN - 捕获全局偏差
        # 收集所有层的偏差并映射回节点级别
        global_deviation_nodes = torch.zeros(B, N, device=node_embeddings.device)

        if len(self.inter_patch_gdns) > 0:
            for layer_idx, inter_gdn in enumerate(self.inter_patch_gdns):
                # 通过GDN层
                patch_nodes_out, patch_masks_out, patch_deviation = inter_gdn(
                    patch_nodes, patch_masks
                )

                # 将当前层的偏差映射回节点级
                layer_deviation_nodes = self._patches_to_nodes_with_scale(
                    patch_deviation, patch_indices, variables, obs_mask,
                    scale=2 ** layer_idx  # 当前层的尺度
                )

                # 累积偏差
                global_deviation_nodes += layer_deviation_nodes

                # 更新patch表示用于下一层
                patch_nodes = patch_nodes_out
                patch_masks = patch_masks_out

            # 平均多层偏差
            global_deviation_nodes = global_deviation_nodes / len(self.inter_patch_gdns)

        # 6. 综合局部和全局偏差
        combined_deviation = (local_deviation + global_deviation_nodes) / 2

        # 7. 异常检测
        anomaly_scores = self.anomaly_head(
            node_embeddings, combined_deviation, values
        )

        # 8. 只保留真实观测的分数
        for key in anomaly_scores:
            if anomaly_scores[key].dim() == 2:
                anomaly_scores[key] = anomaly_scores[key] * obs_mask.float()

        return anomaly_scores, node_embeddings

    def _nodes_to_patches(self, node_embeddings, patch_indices, variables, obs_mask):
        """将节点表示转换为patch级表示"""
        B, N, d_model = node_embeddings.shape
        max_patch = patch_indices.max().item() + 1

        patch_nodes = torch.zeros(
            B, max_patch, self.num_variables, d_model,
            device=node_embeddings.device
        )
        patch_masks = torch.zeros(
            B, max_patch, self.num_variables,
            dtype=torch.bool, device=node_embeddings.device
        )
        patch_counts = torch.zeros_like(patch_masks, dtype=torch.float32)

        for b in range(B):
            for n in range(N):
                if not obs_mask[b, n]:
                    continue
                p = patch_indices[b, n].item()
                v = variables[b, n].item()
                patch_nodes[b, p, v] += node_embeddings[b, n]
                patch_counts[b, p, v] += 1
                patch_masks[b, p, v] = True

        patch_counts = patch_counts.unsqueeze(-1).clamp(min=1)
        patch_nodes = patch_nodes / patch_counts

        return patch_nodes, patch_masks

    def _patches_to_nodes(self, patch_values, patch_indices, variables, obs_mask):
        """将patch级值映射回节点级"""
        B, N = patch_indices.shape
        node_values = torch.zeros(B, N, device=patch_values.device)

        for b in range(B):
            for n in range(N):
                if not obs_mask[b, n]:
                    continue
                p = patch_indices[b, n].item()
                v = variables[b, n].item()
                if p < patch_values.shape[1] and v < patch_values.shape[2]:
                    node_values[b, n] = patch_values[b, p, v]

        return node_values

    def _patches_to_nodes_with_scale(self, patch_values, patch_indices, variables, obs_mask, scale=1):
        """
        将patch级值映射回节点级，考虑尺度因子

        Args:
            patch_values: [B, P_scaled, V] - 当前尺度的patch值
            patch_indices: [B, N] - 原始patch索引
            variables: [B, N] - 变量索引
            obs_mask: [B, N] - 观测mask
            scale: int - 当前层相对于第一层的尺度倍数 (1, 2, 4, 8, ...)
        """
        B, N = patch_indices.shape
        node_values = torch.zeros(B, N, device=patch_values.device)

        for b in range(B):
            for n in range(N):
                if not obs_mask[b, n]:
                    continue

                # 原始patch索引
                p_orig = patch_indices[b, n].item()
                v = variables[b, n].item()

                # 映射到当前尺度的patch索引
                p_scaled = p_orig // scale

                # 检查边界
                if p_scaled < patch_values.shape[1] and v < patch_values.shape[2]:
                    node_values[b, n] = patch_values[b, p_scaled, v]

        return node_values

    def detect_anomalies(
            self,
            timestamps,
            values,
            variables,
            obs_mask=None,
            weights=(0.3, 0.3, 0.4),  # (forecast, recon, graph_deviation)
            threshold=None
    ):
        """
        检测异常

        Args:
            weights: (w_forecast, w_recon, w_graph_dev)
        """
        anomaly_scores, _ = self.forward(timestamps, values, variables, obs_mask)

        # 归一化
        forecast_scores = torch.sigmoid(anomaly_scores['forecast'])
        recon_scores = torch.sigmoid(anomaly_scores['reconstruction'])
        graph_dev_scores = torch.sigmoid(anomaly_scores['graph_deviation'])

        # 加权组合
        w_f, w_r, w_g = weights
        final_scores = (
                w_f * forecast_scores +
                w_r * recon_scores +
                w_g * graph_dev_scores
        )

        # 自动阈值
        if threshold is None:
            if obs_mask is not None:
                valid_scores = final_scores[obs_mask]
            else:
                valid_scores = final_scores.flatten()
            threshold = torch.quantile(valid_scores, 0.95)

        is_anomaly = (final_scores > threshold).float()

        if obs_mask is not None:
            is_anomaly = is_anomaly * obs_mask.float()

        return final_scores, is_anomaly, threshold


class GADIMTS_GDN_Trainer:
    """GDN训练器"""

    def __init__(
            self,
            model: GADIMTS_GDN,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_step(self, timestamps, values, variables, obs_mask=None):
        """训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()

        anomaly_scores, _ = self.model(timestamps, values, variables, obs_mask)

        # 1. 预测损失
        forecast_loss = F.mse_loss(
            anomaly_scores['forecast_values'],
            values,
            reduction='none'
        )

        # 2. 重构损失
        recon_loss = F.mse_loss(
            anomaly_scores['recon_values'],
            values,
            reduction='none'
        )

        # 3. 图偏差正则化（鼓励正常样本偏差小）
        graph_dev_reg = anomaly_scores['graph_deviation'].mean()

        # 应用mask
        if obs_mask is not None:
            forecast_loss = (forecast_loss * obs_mask.float()).sum() / obs_mask.sum()
            recon_loss = (recon_loss * obs_mask.float()).sum() / obs_mask.sum()
        else:
            forecast_loss = forecast_loss.mean()
            recon_loss = recon_loss.mean()

        # 总损失
        total_loss = forecast_loss + recon_loss + 0.1 * graph_dev_reg

        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'forecast_loss': forecast_loss.item(),
            'recon_loss': recon_loss.item(),
            'graph_dev_reg': graph_dev_reg.item()
        }


# ============= 使用示例 =============

def create_synthetic_data(batch_size=32, max_len=100, num_variables=5, missing_rate=0.7):
    """创建合成数据"""
    num_obs = int(max_len * (1 - missing_rate))

    timestamps = torch.sort(torch.rand(batch_size, num_obs) * 100)[0]
    values = torch.randn(batch_size, num_obs)
    variables = torch.randint(0, num_variables, (batch_size, num_obs))

    # 添加异常
    anomaly_mask = torch.rand(batch_size, num_obs) < 0.05
    values[anomaly_mask] += torch.randn(anomaly_mask.sum()) * 5

    return timestamps, values, variables, anomaly_mask


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("GAD-IMTS with GDN 异常检测模型")
    print("=" * 60)

    # 1. 创建模型
    model = GADIMTS_GDN(
        num_variables=5,
        d_model=64,
        num_intra_layers=2,
        num_inter_layers=2,
        patch_size=10.0
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 创建训练器
    trainer = GADIMTS_GDN_Trainer(model, learning_rate=1e-3)

    # 3. 生成数据
    print("\n生成合成数据...")
    timestamps, values, variables, true_anomalies = create_synthetic_data(
        batch_size=32, num_variables=5, missing_rate=0.7
    )

    print(f"数据形状: {timestamps.shape}")
    print(f"缺失率: {1 - timestamps.shape[1] / 100:.1%}")
    print(f"真实异常率: {true_anomalies.float().mean():.2%}")

    # 4. 训练
    print("\n开始训练...")
    for epoch in range(10):
        losses = trainer.train_step(timestamps, values, variables)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Total={losses['total_loss']:.4f}, "
                  f"Forecast={losses['forecast_loss']:.4f}, "
                  f"Recon={losses['recon_loss']:.4f}, "
                  f"GraphDev={losses['graph_dev_reg']:.4f}")

    # 5. 检测异常
    print("\n执行异常检测...")
    model.eval()
    with torch.no_grad():
        scores, is_anomaly, threshold = model.detect_anomalies(
            timestamps, values, variables,
            weights=(0.3, 0.3, 0.4)  # GDN权重更高
        )

if __name__ == "__main__":
    example_usage()