import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
from mssa_transformer.anomaly.gdn import GDN
from mssa_transformer.config import DATA_DIR, OUTPUTS_DIR


GDN_OUTPUT_DIR = OUTPUTS_DIR / "anomaly_detection" / "gdn"


# 工具函数
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(train_path, test_path):
    """
    加载训练和测试数据
    假设数据格式为numpy数组，包含特征和标签
    """
    print(f"Loading training data from: {train_path}")
    print(f"Loading testing data from: {test_path}")

    # 加载训练数据
    train_data = np.load(train_path, allow_pickle=True)
    if isinstance(train_data, dict):
        train_features = train_data['features']  # 形状: (samples, nodes, features)
        train_labels = train_data['labels']  # 形状: (samples, nodes)
    else:
        # 如果是单个数组，假设最后一列是标签
        train_features = train_data[:, :, :-1]
        train_labels = train_data[:, :, -1]

    # 加载测试数据
    test_data = np.load(test_path, allow_pickle=True)
    if isinstance(test_data, dict):
        test_features = test_data['features']
        test_labels = test_data['labels']
    else:
        test_features = test_data[:, :, :-1]
        test_labels = test_data[:, :, -1]

    return train_features, train_labels, test_features, test_labels


def create_edge_index(node_num, connection_type='fully_connected'):
    """
    创建图的边索引
    """
    if connection_type == 'fully_connected':
        # 创建全连接图
        edge_list = []
        for i in range(node_num):
            for j in range(node_num):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    elif connection_type == 'knn':
        # 创建KNN图(这里简化为每个节点连接到相邻的k个节点)
        k = min(5, node_num - 1)
        edge_list = []
        for i in range(node_num):
            for j in range(max(0, i - k // 2), min(node_num, i + k // 2 + 1)):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # 默认创建链式连接
        edge_list = []
        for i in range(node_num - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index


def prepare_data(features, labels, scaler=None, fit_scaler=True):
    """
    准备数据，包括归一化
    """
    # 重塑数据以便归一化
    original_shape = features.shape
    features_reshaped = features.reshape(-1, features.shape[-1])

    if fit_scaler and scaler is None:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_reshaped)
    elif scaler is not None:
        features_normalized = scaler.transform(features_reshaped)
    else:
        features_normalized = features_reshaped

    # 恢复原始形状
    features_normalized = features_normalized.reshape(original_shape)

    return features_normalized, labels, scaler


class AnomalyDetectionTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, edge_index):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 使用GDN进行前向传播
            output = self.model(data, edge_index)

            # 计算重构损失
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader, edge_index):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, edge_index)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(dataloader)

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


def detect_anomalies(model, test_loader, edge_index, device, threshold_percentile=95):
    """
    使用训练好的模型检测异常
    """
    model.eval()
    reconstruction_errors = []
    true_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 获取重构输出
            output = model(data, edge_index)

            # 计算重构误差
            error = torch.mean((output - target) ** 2, dim=1)  # 每个样本的平均重构误差

            reconstruction_errors.extend(error.cpu().numpy())
            true_labels.extend(target.cpu().numpy().flatten())

    # 转换为numpy数组
    reconstruction_errors = np.array(reconstruction_errors)
    true_labels = np.array(true_labels)

    # 使用阈值确定异常
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    predicted_labels = (reconstruction_errors > threshold).astype(int)

    return predicted_labels, true_labels, reconstruction_errors, threshold


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    """
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
    """
    绘制训练曲线和异常检测结果
    """
    os.makedirs(save_dir, exist_ok=True)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 绘制重构误差分布
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
        train_labels = np.zeros((batch_size, node_num))  # 正常数据标签为0

        # 测试数据 (包含异常)
        test_normal = np.random.randn(50, node_num, feature_dim) * 0.5
        test_anomaly = np.random.randn(20, node_num, feature_dim) * 2.0  # 异常数据方差更大

        test_features = np.concatenate([test_normal, test_anomaly], axis=0)
        test_labels = np.concatenate([
            np.zeros((50, node_num)),  # 正常
            np.ones((20, node_num))  # 异常
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
    edge_index_sets = [edge_index]  # GDN支持多个图结构

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
    trainer = AnomalyDetectionTrainer(model, device, learning_rate=0.001)

    # 训练模型
    trainer.train(train_loader, train_loader, edge_index, epochs=50)  # 使用train_loader作为验证集

    # 异常检测
    print("\nPerforming anomaly detection...")
    predicted_labels, true_labels, reconstruction_errors, threshold = detect_anomalies(
        model, test_loader, edge_index, device, threshold_percentile=90
    )

    # 计算评估指标
    # 将节点级别的标签转换为样本级别
    true_labels_sample = []
    predicted_labels_sample = []

    samples_processed = 0
    for i, (_, target_batch) in enumerate(test_loader):
        batch_size_current = target_batch.shape[0]
        for j in range(batch_size_current):
            # 如果任何节点是异常，则整个样本被认为是异常
            true_sample_label = int(np.any(target_batch[j].numpy() > 0.5))
            pred_sample_label = predicted_labels[samples_processed + j]

            true_labels_sample.append(true_sample_label)
            predicted_labels_sample.append(pred_sample_label)

        samples_processed += batch_size_current

    # 计算指标
    metrics = calculate_metrics(true_labels_sample, predicted_labels_sample)

    # 打印结果
    print("\n" + "=" * 50)
    print("异常检测结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score): {metrics['f1_score']:.4f}")
    print(f"检测阈值: {threshold:.6f}")
    print(f"异常样本数量: {sum(predicted_labels_sample)}/{len(predicted_labels_sample)}")
    print(f"真实异常数量: {sum(true_labels_sample)}/{len(true_labels_sample)}")

    # 混淆矩阵
    cm = confusion_matrix(true_labels_sample, predicted_labels_sample)
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
