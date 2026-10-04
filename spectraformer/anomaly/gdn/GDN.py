import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Parameter, Linear
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd  # 添加pandas用于读取CSV
import os
import math
from spectraformer.config import DATA_DIR, OUTPUTS_DIR


GDN_OUTPUT_DIR = OUTPUTS_DIR / "anomaly_detection" / "gdn"


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """创建批量边索引"""
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num: (i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


def softmax(src, index, num_nodes=None):
    """自定义softmax函数，类似torch_geometric.utils.softmax"""
    if num_nodes is None:
        num_nodes = int(index.max()) + 1

    out = src - src.max()
    out = out.exp()

    out_sum = torch.zeros(num_nodes, dtype=out.dtype, device=out.device)
    out_sum.scatter_add_(0, index, out)
    out_sum = out_sum[index]

    return out / (out_sum + 1e-16)


class GraphLayer(nn.Module):
    """自定义图卷积层，基于注意力机制"""

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1):
        super(GraphLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_i)
        nn.init.xavier_uniform_(self.att_j)
        nn.init.zeros_(self.att_em_i)
        nn.init.zeros_(self.att_em_j)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        # 线性变换
        x = self.lin(x)

        # 添加自环
        num_nodes = x.size(0)
        self_loop_index = torch.arange(num_nodes, dtype=torch.long, device=x.device)
        self_loop_index = self_loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)

        # 消息传递
        out = self.propagate(x, edge_index, embedding, return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def propagate(self, x, edge_index, embedding, return_attention_weights):
        # 获取边的源节点和目标节点
        row, col = edge_index

        # 重塑x为多头注意力格式
        x = x.view(-1, self.heads, self.out_channels)

        # 获取源节点和目标节点的特征
        x_i = x[row]  # 源节点特征
        x_j = x[col]  # 目标节点特征

        if embedding is not None:
            embedding_i = embedding[row]
            embedding_j = embedding[col]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)
        else:
            key_i = x_i
            key_j = x_j

        # 计算注意力权重
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # 应用softmax
        alpha = softmax(alpha.view(-1), row, num_nodes=x.size(0))
        alpha = alpha.view(-1, self.heads, 1)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 消息聚合
        out = x_j * alpha

        # 聚合消息
        out_sum = torch.zeros_like(x)
        out_sum.scatter_add_(0, row.view(-1, 1, 1).expand(-1, self.heads, self.out_channels), out)

        return out_sum


class OutLayer(nn.Module):
    """输出层"""

    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    """GNN层"""

    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(
            in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False
        )

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(
            x, edge_index, embedding, return_attention_weights=True
        )
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)

        return self.relu(out)


class GDN(nn.Module):
    """Graph Deviation Network for Anomaly Detection"""

    def __init__(
            self,
            edge_index_sets,
            node_num,
            dim=64,
            out_layer_inter_dim=256,
            input_dim=10,
            out_layer_num=1,
            topk=20,
    ):
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        device = get_device()

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1)
                for i in range(edge_set_num)
            ]
        )

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(
            dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim
        )

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index=None):
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if (
                    cache_edge_index is None
                    or cache_edge_index.shape[1] != edge_num * batch_num
            ):
                self.cache_edge_index_sets[i] = get_batch_edge_index(
                    edge_index, batch_num, node_num
                ).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(
                weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
            )
            cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)

            # 添加数值稳定性检查
            if torch.isnan(cos_ji_mat).any() or torch.isinf(cos_ji_mat).any():
                print("Warning: NaN or Inf detected in cosine similarity matrix")
                cos_ji_mat = torch.eye(node_num, device=device)  # 使用单位矩阵作为fallback

            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = (
                torch.arange(0, node_num)
                .T.unsqueeze(1)
                .repeat(1, topk_num)
                .flatten()
                .to(device)
                .unsqueeze(0)
            )
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)

            gcn_out = self.gnn_layers[i](
                x,
                batch_gated_edge_index,
                node_num=node_num * batch_num,
                embedding=all_embeddings,
            )

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        return out


def load_data(train_path, test_path):
    """加载训练和测试数据，支持CSV和NPY格式"""
    print(f"Loading training data from: {train_path}")
    print(f"Loading testing data from: {test_path}")

    def load_single_file(file_path):
        """加载单个文件，自动检测格式"""
        if file_path.endswith('.csv'):
            # 加载CSV文件
            import pandas as pd
            print(f"Reading CSV file: {file_path}")

            # 读取CSV，处理可能的缺失值
            data = pd.read_csv(file_path)

            # 打印原始数据信息
            print(f"Original CSV shape: {data.shape}")
            print(f"Original data info:")
            print(f"  - Columns: {list(data.columns)}")
            print(f"  - Missing values per column: {data.isnull().sum().sum()}")
            print(f"  - Data types: {data.dtypes.value_counts().to_dict()}")

            # 处理非数值列
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < len(data.columns):
                print(f"Dropping non-numeric columns: {list(set(data.columns) - set(numeric_columns))}")
                data = data[numeric_columns]

            # 处理缺失值
            if data.isnull().any().any():
                print(f"Found {data.isnull().sum().sum()} missing values, filling with column means")
                data = data.fillna(data.mean())

                # 如果还有NaN（可能整列都是NaN），用0填充
                if data.isnull().any().any():
                    print("Still have NaN values, filling with 0")
                    data = data.fillna(0)

            # 处理无穷值
            data = data.replace([np.inf, -np.inf], np.nan)
            if data.isnull().any().any():
                print(f"Found infinite values, replacing with column means")
                data = data.fillna(data.mean())
                data = data.fillna(0)  # 最后的安全网

            # 检查极值并裁剪
            data_array = data.values

            # 统计极值
            print(f"Data range before clipping: min={data_array.min():.4f}, max={data_array.max():.4f}")

            # 裁剪极值（使用99.9%分位数）
            lower_bound = np.percentile(data_array, 0.1)
            upper_bound = np.percentile(data_array, 99.9)
            data_array = np.clip(data_array, lower_bound, upper_bound)

            print(f"Data range after clipping: min={data_array.min():.4f}, max={data_array.max():.4f}")
            print(f"Final processed CSV shape: {data_array.shape}")

            return data_array

        elif file_path.endswith('.npy'):
            # 加载NPY文件
            data = np.load(file_path, allow_pickle=True)
            print(f"Loaded NPY with shape: {data.shape}")
            return data
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    # 加载训练数据
    train_data = load_single_file(train_path)

    # 加载测试数据
    test_data = load_single_file(test_path)

    # 处理数据格式
    def process_data(data_array, data_name):
        """处理数据数组，转换为所需格式"""
        print(f"Processing {data_name} data with shape: {data_array.shape}")

        if len(data_array.shape) == 2:
            # 2D数据: (samples, features)
            # 需要转换为: (samples, nodes, features)
            samples, total_features = data_array.shape

            # 假设最后一列是标签
            if total_features > 1:
                features = data_array[:, :-1]  # 除最后一列外的所有列
                labels = data_array[:, -1]  # 最后一列作为标签

                # 将特征重塑为图数据格式
                # 假设每个样本的特征可以平均分配给多个节点
                feature_dim = min(10, features.shape[1])  # 每个节点的特征维度
                node_num = max(1, features.shape[1] // feature_dim)  # 节点数量

                if features.shape[1] % feature_dim != 0:
                    # 如果不能整除，截断或填充
                    features = features[:, :node_num * feature_dim]

                features_3d = features.reshape(samples, node_num, feature_dim)
                labels_2d = np.tile(labels.reshape(-1, 1), (1, node_num))  # 复制标签到所有节点

            else:
                # 只有一列，全部作为特征
                features_3d = data_array.reshape(samples, 1, 1)
                labels_2d = np.zeros((samples, 1))

        elif len(data_array.shape) == 3:
            # 3D数据: 已经是 (samples, nodes, features) 格式
            features_3d = data_array[:, :, :-1] if data_array.shape[2] > 1 else data_array
            labels_2d = data_array[:, :, -1] if data_array.shape[2] > 1 else np.zeros(
                (data_array.shape[0], data_array.shape[1]))

        else:
            raise ValueError(f"Unsupported data shape: {data_array.shape}")

        print(f"Processed {data_name} - Features: {features_3d.shape}, Labels: {labels_2d.shape}")
        return features_3d, labels_2d

    train_features, train_labels = process_data(train_data, "training")
    test_features, test_labels = process_data(test_data, "testing")

    return train_features, train_labels, test_features, test_labels


def create_edge_index(node_num, connection_type='fully_connected'):
    """创建图的边索引"""
    if connection_type == 'fully_connected':
        edge_list = []
        for i in range(node_num):
            for j in range(node_num):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    elif connection_type == 'knn':
        k = min(5, node_num - 1)
        edge_list = []
        for i in range(node_num):
            for j in range(max(0, i - k // 2), min(node_num, i + k // 2 + 1)):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_list = []
        for i in range(node_num - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index


def prepare_data(features, labels, scaler=None, fit_scaler=True):
    """准备数据，包括归一化和清理"""
    print(f"Preparing data with shape: {features.shape}")

    # 检查原始数据
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    print(f"Before cleaning - NaN count: {nan_count}, Inf count: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("Cleaning data...")
        # 处理NaN值
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # 再次检查
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        print(f"After cleaning - NaN count: {nan_count}, Inf count: {inf_count}")

    # 归一化
    original_shape = features.shape
    features_reshaped = features.reshape(-1, features.shape[-1])

    if fit_scaler and scaler is None:
        scaler = StandardScaler()
        try:
            features_normalized = scaler.fit_transform(features_reshaped)
        except Exception as e:
            print(f"Scaler error: {e}, using min-max normalization instead")
            # 使用简单的min-max归一化作为后备
            min_vals = np.min(features_reshaped, axis=0)
            max_vals = np.max(features_reshaped, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # 避免除零
            features_normalized = (features_reshaped - min_vals) / range_vals
            features_normalized = features_normalized * 2 - 1  # 缩放到[-1, 1]
            scaler = None
    elif scaler is not None:
        try:
            features_normalized = scaler.transform(features_reshaped)
        except Exception as e:
            print(f"Transform error: {e}, using original data")
            features_normalized = features_reshaped
    else:
        features_normalized = features_reshaped

    features_normalized = features_normalized.reshape(original_shape)

    # 最终检查和清理
    nan_count = np.isnan(features_normalized).sum()
    inf_count = np.isinf(features_normalized).sum()

    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in normalized features, replacing with zeros")
        features_normalized = np.nan_to_num(features_normalized, nan=0.0)

    if inf_count > 0:
        print(f"Warning: {inf_count} Inf values in normalized features, clipping")
        features_normalized = np.nan_to_num(features_normalized, posinf=3.0, neginf=-3.0)

    # 最终裁剪确保数值在合理范围内
    features_normalized = np.clip(features_normalized, -5, 5)

    print(f"Final normalized data range: [{features_normalized.min():.4f}, {features_normalized.max():.4f}]")

    return features_normalized, labels, scaler


class AnomalyDetectionTrainer:
    def __init__(self, model, device, learning_rate=0.0001):  # 降低学习率
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加权重衰减
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, edge_index):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # 检查数据是否包含NaN或inf
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Warning: NaN or Inf detected in input data at batch {batch_idx}")
                continue

            self.optimizer.zero_grad()

            try:
                # 使用GDN进行前向传播
                output = self.model(data)

                # 检查模型输出
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"Warning: NaN or Inf detected in model output at batch {batch_idx}")
                    continue

                # 计算重构损失
                batch_size, node_num = output.shape
                feature_dim = data.shape[-1]

                # 创建重构层（如果不存在）
                if not hasattr(self, 'reconstruction_layer'):
                    self.reconstruction_layer = nn.Linear(1, feature_dim).to(self.device)
                    # 初始化重构层权重
                    nn.init.xavier_uniform_(self.reconstruction_layer.weight)
                    nn.init.zeros_(self.reconstruction_layer.bias)

                output_reshaped = output.unsqueeze(-1)  # (batch_size, node_num, 1)
                reconstructed = self.reconstruction_layer(output_reshaped)

                # 检查重构结果
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    print(f"Warning: NaN or Inf detected in reconstructed output at batch {batch_idx}")
                    continue

                loss = self.criterion(reconstructed, data)

                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected at batch {batch_idx}")
                    continue

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if hasattr(self, 'reconstruction_layer'):
                    torch.nn.utils.clip_grad_norm_(self.reconstruction_layer.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        if num_batches == 0:
            return float('inf')
        return total_loss / num_batches

    def validate(self, dataloader, edge_index):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                # 检查数据
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue

                try:
                    output = self.model(data)

                    if torch.isnan(output).any() or torch.isinf(output).any():
                        continue

                    # 使用相同的重构逻辑
                    if hasattr(self, 'reconstruction_layer'):
                        output_reshaped = output.unsqueeze(-1)
                        reconstructed = self.reconstruction_layer(output_reshaped)

                        if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                            continue

                        loss = self.criterion(reconstructed, data)

                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            total_loss += loss.item()
                            num_batches += 1

                except Exception as e:
                    print(f"Validation error in batch {batch_idx}: {e}")
                    continue

        if num_batches == 0:
            return float('inf')
        return total_loss / num_batches

    def train(self, train_loader, val_loader, edge_index, epochs=100):
        print("Starting training...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, edge_index)
            val_loss = self.validate(val_loader, edge_index)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        print("Training completed!")


def detect_anomalies(model, test_loader, trainer, device, threshold_percentile=95):
    """使用训练好的模型检测异常"""
    model.eval()
    reconstruction_errors = []
    true_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 获取GDN输出
            output = model(data)

            # 重构特征
            output_reshaped = output.unsqueeze(-1)
            reconstructed = trainer.reconstruction_layer(output_reshaped)

            # 计算重构误差
            error = torch.mean((reconstructed - data) ** 2, dim=[1, 2])  # 每个样本的平均重构误差

            reconstruction_errors.extend(error.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)

    # 将节点级标签转换为样本级标签
    sample_labels = []
    for i in range(len(true_labels)):
        # 如果任何节点是异常，则整个样本被认为是异常
        sample_label = int(np.any(true_labels[i] > 0.5))
        sample_labels.append(sample_label)

    sample_labels = np.array(sample_labels)

    # 使用阈值确定异常
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    predicted_labels = (reconstruction_errors > threshold).astype(int)

    return predicted_labels, sample_labels, reconstruction_errors, threshold


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_results(trainer, reconstruction_errors, threshold, save_dir=GDN_OUTPUT_DIR):
    """绘制训练曲线和异常检测结果"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(reconstruction_errors, bins=50, alpha=0.7, density=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'))
    plt.show()


def main():
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 数据路径 (请根据实际情况修改)
    train_data_path = str(DATA_DIR / "gdn" / "train.csv")
    test_data_path = str(DATA_DIR / "gdn" / "test.csv")

    try:
        # 加载数据
        train_features, train_labels, test_features, test_labels = load_data(
            train_data_path, test_data_path
        )
        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")

    except FileNotFoundError:
        print("数据文件未找到，生成模拟数据...")
        # 生成模拟数据
        batch_size = 100
        node_num = 25
        feature_dim = 10

        # 训练数据 (正常数据)
        train_features = np.random.randn(batch_size, node_num, feature_dim) * 0.5
        train_labels = np.zeros((batch_size, node_num))

        # 测试数据 (包含异常)
        test_normal = np.random.randn(50, node_num, feature_dim) * 0.5
        test_anomaly = np.random.randn(20, node_num, feature_dim) * 2.0

        test_features = np.concatenate([test_normal, test_anomaly], axis=0)
        test_labels = np.concatenate([
            np.zeros((50, node_num)),
            np.ones((20, node_num))
        ], axis=0)

        print(f"Generated train features shape: {train_features.shape}")
        print(f"Generated test features shape: {test_features.shape}")

    # 数据预处理
    train_features_norm, train_labels_norm, scaler = prepare_data(
        train_features, train_labels, fit_scaler=True
    )
    test_features_norm, test_labels_norm, _ = prepare_data(
        test_features, test_labels, scaler=scaler, fit_scaler=False
    )

    # 获取数据维度
    batch_size, node_num, feature_dim = train_features_norm.shape

    # 创建图结构
    edge_index = create_edge_index(node_num, connection_type='knn')
    edge_index_sets = [edge_index]

    print(f"Created graph with {edge_index.shape[1]} edges")

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features_norm),
        torch.FloatTensor(train_labels_norm)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_features_norm),
        torch.FloatTensor(test_labels_norm)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 创建模型
    model = GDN(
        edge_index_sets=edge_index_sets,
        node_num=node_num,
        dim=64,
        out_layer_inter_dim=256,
        input_dim=feature_dim,
        out_layer_num=2,
        topk=min(10, node_num - 1)
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 创建训练器
    trainer = AnomalyDetectionTrainer(model, device, learning_rate=0.0001)

    # 打印数据统计信息
    print(f"\nData statistics:")
    print(f"Train features - Min: {train_features_norm.min():.4f}, Max: {train_features_norm.max():.4f}")
    print(f"Train features - Mean: {train_features_norm.mean():.4f}, Std: {train_features_norm.std():.4f}")
    print(f"Test features - Min: {test_features_norm.min():.4f}, Max: {test_features_norm.max():.4f}")
    print(f"Test features - Mean: {test_features_norm.mean():.4f}, Std: {test_features_norm.std():.4f}")

    # 训练模型
    trainer.train(train_loader, train_loader, edge_index, epochs=50)

    # 异常检测
    print("\nPerforming anomaly detection...")
    predicted_labels, true_labels, reconstruction_errors, threshold = detect_anomalies(
        model, test_loader, trainer, device, threshold_percentile=90
    )

    # 计算评估指标
    metrics = calculate_metrics(true_labels, predicted_labels)

    # 打印结果
    print("\n" + "=" * 50)
    print("异常检测结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score): {metrics['f1_score']:.4f}")
    print(f"检测阈值: {threshold:.6f}")
    print(f"异常样本数量: {sum(predicted_labels)}/{len(predicted_labels)}")
    print(f"真实异常数量: {sum(true_labels)}/{len(true_labels)}")

    # 混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"\n混淆矩阵:")
    print(f"True Negative: {cm[0, 0]}, False Positive: {cm[0, 1]}")
    print(f"False Negative: {cm[1, 0]}, True Positive: {cm[1, 1]}")

    # 绘制结果
    plot_results(trainer, reconstruction_errors, threshold)

    # 保存模型
    model_save_path = GDN_OUTPUT_DIR / "gdn_model.pth"
    os.makedirs(GDN_OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n模型已保存到: {model_save_path}")


if __name__ == "__main__":
    main()
