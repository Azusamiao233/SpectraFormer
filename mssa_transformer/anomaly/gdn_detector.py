import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


# ==================== 数据处理 ====================

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                 window_size: int = 10, stride: int = 1):
        self.data = torch.FloatTensor(data)
        self.mask = torch.FloatTensor(mask)
        self.labels = torch.FloatTensor(labels)
        self.window_size = window_size
        self.stride = stride

        # 预先创建窗口索引
        self.indices = list(range(0, len(data) - window_size, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size

        return {
            'data': self.data[start:end],
            'mask': self.mask[start:end],
            'label': self.labels[end - 1]
        }


# ==================== 图注意力层 ====================

class GraphAttentionLayer(nn.Module):
    """单头图注意力层"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, num_nodes, in_features]
        adj: [batch, num_nodes, num_nodes] 或 None
        """
        batch_size, num_nodes = x.shape[0], x.shape[1]

        # 线性变换
        Wx = self.W(x)  # [batch, num_nodes, out_features]

        # 计算注意力系数
        # 复制节点特征用于拼接
        a_input = torch.cat([
            Wx.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, -1),
            Wx.repeat(1, num_nodes, 1)
        ], dim=2).view(batch_size, num_nodes, num_nodes, 2 * self.out_features)

        # 计算注意力得分
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [batch, num_nodes, num_nodes]

        # 如果提供了邻接矩阵，应用mask
        if adj is not None:
            e = e.masked_fill(adj == 0, -1e9)

        # Softmax归一化
        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)

        # 聚合邻居特征
        h_prime = torch.matmul(attention, Wx)

        return h_prime, attention


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力"""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout)
            for _ in range(num_heads)
        ])

        if concat:
            self.out_proj = nn.Linear(out_features * num_heads, out_features)
        else:
            self.out_proj = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None):
        outputs = []
        attentions = []

        for attn in self.attentions:
            out, attn_weights = attn(x, adj)
            outputs.append(out)
            attentions.append(attn_weights)

        if self.concat:
            h = torch.cat(outputs, dim=-1)
        else:
            h = torch.mean(torch.stack(outputs), dim=0)

        h = self.out_proj(h)

        return h, torch.stack(attentions).mean(0)


# ==================== 核心GDN模型 ====================

class GDNAnomalyDetector(nn.Module):
    """
    简化但高效的GDN异常检测器
    核心思想：通过图注意力学习节点间依赖，检测偏离正常模式的异常
    """

    def __init__(self, num_nodes: int, window_size: int,
                 feature_dim: int = 32, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 节点特征提取（时序卷积）
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 特征投影（将feature_dim投影到hidden_dim）
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # 图注意力层
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            self.gat_layers.append(
                MultiHeadGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 图结构学习
        self.graph_learner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 节点级重建器
        self.node_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, window_size)
        )

        # 图级异常分类器
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: [batch, window_size, num_nodes]
        mask: [batch, window_size, num_nodes]
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. 提取每个节点的时序特征
        node_features = []
        for i in range(self.num_nodes):
            node_seq = x[:, :, i].unsqueeze(1)  # [batch, 1, window]
            node_mask = mask[:, :, i].unsqueeze(1)

            # 对缺失值进行mask（乘以mask）
            node_seq = node_seq * node_mask

            # 时序卷积
            feat = self.temporal_conv(node_seq)  # [batch, feature_dim, window]
            feat = self.pool(feat).squeeze(-1)  # [batch, feature_dim]
            node_features.append(feat)

        # [batch, num_nodes, feature_dim]
        node_features = torch.stack(node_features, dim=1)

        # 投影到hidden_dim
        h = self.feature_proj(node_features)  # [batch, num_nodes, hidden_dim]

        # 2. 学习图结构（构建邻接矩阵）
        graph_emb = self.graph_learner(h)  # [batch, num_nodes, hidden_dim]

        # 计算节点相似度作为邻接矩阵
        adj = torch.matmul(graph_emb, graph_emb.transpose(1, 2))  # [batch, num_nodes, num_nodes]
        adj = F.relu(adj)

        # 归一化邻接矩阵
        degree = adj.sum(dim=2, keepdim=True) + 1e-6
        adj = adj / degree

        # 3. 多层图注意力
        all_attentions = []

        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_new, attn = gat(h, adj)
            h = norm(h_new + h)
            all_attentions.append(attn)

        # h: [batch, num_nodes, hidden_dim]

        # 4. 节点重建（用于重建损失）
        node_reconstructions = []
        for i in range(self.num_nodes):
            recon = self.node_reconstructor(h[:, i, :])  # [batch, window]
            node_reconstructions.append(recon)

        # [batch, num_nodes, window]
        reconstructions = torch.stack(node_reconstructions, dim=1)

        # 5. 图级异常分类
        graph_feat = h.reshape(batch_size, -1)  # [batch, num_nodes * hidden_dim]
        anomaly_score = self.graph_classifier(graph_feat).squeeze(-1)  # [batch]

        return {
            'reconstructions': reconstructions,  # [batch, num_nodes, window]
            'anomaly_scores': anomaly_score,  # [batch]
            'node_features': h,  # [batch, num_nodes, hidden_dim]
            'attentions': all_attentions,
            'adj_matrix': adj
        }


# ==================== 损失函数 ====================

class AnomalyDetectionLoss(nn.Module):
    """组合损失：重建损失 + 分类损失 + 对比损失"""

    def __init__(self, alpha: float = 0.5, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha  # 重建损失权重
        self.beta = beta  # 对比损失权重

    def forward(self, predictions: Dict, targets: torch.Tensor,
                data: torch.Tensor, mask: torch.Tensor):
        """
        predictions: 模型输出
        targets: [batch] 标签 (0/1)
        data: [batch, window, num_nodes]
        mask: [batch, window, num_nodes]
        """
        batch_size = data.shape[0]

        # 1. 重建损失（只在有效位置计算）
        recons = predictions['reconstructions']  # [batch, num_nodes, window]
        data_transposed = data.transpose(1, 2)  # [batch, num_nodes, window]
        mask_transposed = mask.transpose(1, 2)

        # 计算MSE，只在mask=1的位置
        recon_errors = (recons - data_transposed) ** 2
        recon_errors = recon_errors * mask_transposed
        recon_loss = recon_errors.sum() / (mask_transposed.sum() + 1e-6)

        # 2. 分类损失（BCE）
        anomaly_scores = predictions['anomaly_scores']
        cls_loss = F.binary_cross_entropy_with_logits(anomaly_scores, targets)

        # 3. 对比损失（可选，让正常和异常样本在特征空间分离）
        node_feat = predictions['node_features'].mean(dim=1)  # [batch, hidden_dim]

        normal_mask = (targets == 0).float()
        anomaly_mask = (targets == 1).float()

        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            normal_center = (node_feat * normal_mask.unsqueeze(1)).sum(0) / (normal_mask.sum() + 1e-6)
            anomaly_center = (node_feat * anomaly_mask.unsqueeze(1)).sum(0) / (anomaly_mask.sum() + 1e-6)

            # 希望两个中心距离更远
            contrastive_loss = -F.cosine_similarity(
                normal_center.unsqueeze(0),
                anomaly_center.unsqueeze(0)
            ).mean()
        else:
            contrastive_loss = torch.tensor(0.0).to(data.device)

        # 总损失
        total_loss = self.alpha * recon_loss + (1 - self.alpha - self.beta) * cls_loss + self.beta * contrastive_loss

        return {
            'total': total_loss,
            'recon': recon_loss,
            'cls': cls_loss,
            'contrast': contrastive_loss
        }


# ==================== 训练 ====================

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str) -> Dict[str, float]:
    model.train()
    total_losses = {'total': 0.0, 'recon': 0.0, 'cls': 0.0, 'contrast': 0.0}

    for batch in dataloader:
        data = batch['data'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(data, mask)
        losses = criterion(outputs, labels, data, mask)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for key in total_losses:
            total_losses[key] += losses[key].item()

    return {k: v / len(dataloader) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores = []
    all_labels = []

    for batch in dataloader:
        data = batch['data'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(data, mask)

        # 组合异常分数：分类得分 + 重建误差
        cls_scores = torch.sigmoid(outputs['anomaly_scores'])

        recons = outputs['reconstructions']
        data_t = data.transpose(1, 2)
        mask_t = mask.transpose(1, 2)

        recon_errors = ((recons - data_t) ** 2 * mask_t).sum(dim=[1, 2]) / (mask_t.sum(dim=[1, 2]) + 1e-6)
        recon_errors_norm = (recon_errors - recon_errors.mean()) / (recon_errors.std() + 1e-6)

        # 组合分数
        combined_scores = 0.5 * cls_scores + 0.5 * torch.sigmoid(recon_errors_norm)

        all_scores.append(combined_scores.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ==================== 评估指标 ====================

def calculate_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = None):
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix, precision_recall_curve
    )

    if threshold is None:
        # 使用Youden's J statistic找最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx] if best_idx < len(thresholds) else np.percentile(y_scores, 90)

    y_pred = (y_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        auc_pr = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    except:
        auc_roc = auc_pr = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'auc_roc': auc_roc, 'auc_pr': auc_pr, 'threshold': threshold,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


def print_metrics(metrics: Dict, title: str = "评估结果"):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}  |  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:    {metrics['auc_pr']:.4f}  |  Threshold: {metrics['threshold']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP: {metrics['tp']:3d}  |  FP: {metrics['fp']:3d}")
    print(f"  FN: {metrics['fn']:3d}  |  TN: {metrics['tn']:3d}")
    print(f"{'=' * 60}\n")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 配置
    num_nodes = 20
    window_size = 10
    num_samples = 10000
    batch_size = 128
    epochs = 150
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"配置: Device={device}, Nodes={num_nodes}, Window={window_size}")

    # 生成数据（更真实的异常模式）
    data = np.random.randn(num_samples, num_nodes) * 0.5

    # 添加正常的周期性模式
    for i in range(num_nodes):
        freq = np.random.uniform(0.1, 0.3)
        phase = np.random.uniform(0, 2 * np.pi)
        data[:, i] += np.sin(np.arange(num_samples) * freq + phase)

    # 注入点异常（突发型）
    labels = np.zeros(num_samples)
    anomaly_indices = np.random.choice(
        range(window_size, num_samples - window_size),
        size=int(num_samples * 0.1),
        replace=False
    )
    labels[anomaly_indices] = 1

    for idx in anomaly_indices:
        num_anomaly_nodes = np.random.randint(3, 8)
        anomaly_nodes = np.random.choice(num_nodes, num_anomaly_nodes, replace=False)
        # 突发异常：大幅偏离
        data[idx, anomaly_nodes] += np.random.choice([-1, 1], num_anomaly_nodes) * np.random.uniform(3, 6,
                                                                                                     num_anomaly_nodes)

    # 生成缺失模式
    mask = np.ones((num_samples, num_nodes))
    for t in range(num_samples):
        num_missing = np.random.randint(2, 6)
        missing_nodes = np.random.choice(num_nodes, num_missing, replace=False)
        mask[t, missing_nodes] = 0

    # 数据归一化
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / (data_std + 1e-6)

    # 划分数据
    split = int(0.7 * num_samples)
    train_data, test_data = data[:split], data[split:]
    train_mask, test_mask = mask[:split], mask[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # 数据集
    train_dataset = TimeSeriesDataset(train_data, train_mask, train_labels, window_size)
    test_dataset = TimeSeriesDataset(test_data, test_mask, test_labels, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    model = GDNAnomalyDetector(
        num_nodes=num_nodes,
        window_size=window_size,
        feature_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ).to(device)

    criterion = AnomalyDetectionLoss(alpha=0.4, beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练
    print(f"\n{'=' * 60}")
    print("开始训练...")
    print(f"{'=' * 60}\n")

    best_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {train_losses['total']:.4f} "
                  f"(Recon: {train_losses['recon']:.4f}, Cls: {train_losses['cls']:.4f})")

            # 验证
            val_scores, val_labels = evaluate(model, test_loader, device)
            val_metrics = calculate_metrics(val_labels, val_scores)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_model_state = model.state_dict().copy()
                print(f"  ✓ 最佳F1更新: {best_f1:.4f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 最终测试
    print(f"\n{'=' * 60}")
    print("最终测试集评估")
    print(f"{'=' * 60}")

    test_scores, test_labels = evaluate(model, test_loader, device)
    test_metrics = calculate_metrics(test_labels, test_scores)
    print_metrics(test_metrics, "最终结果")

    print(f"实际异常: {test_labels.sum():.0f}/{len(test_labels)} ({test_labels.mean() * 100:.1f}%)")
    print(f"检测异常: {test_metrics['tp'] + test_metrics['fp']}/{len(test_labels)} "
          f"({(test_metrics['tp'] + test_metrics['fp']) / len(test_labels) * 100:.1f}%)")