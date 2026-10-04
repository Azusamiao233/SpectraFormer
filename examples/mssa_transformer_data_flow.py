"""
MSSA输出与Transformer输入连接详解
展示MSSA分解的数据如何转换为Transformer的输入特征
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mssa_transformer.mssa import MSSA, setup_chinese_fonts

# 设置中文字体
setup_chinese_fonts()
np.random.seed(42)


def demonstrate_mssa_transformer_connection():
    """
    演示MSSA输出如何连接到Transformer输入
    """
    print("=" * 60)
    print("MSSA与Transformer数据连接演示")
    print("=" * 60)

    # 步骤1: 创建示例数据
    print("\n步骤1: 创建原始多变量时间序列数据")
    N = 200  # 时间步数
    P = 3  # 变量数（特征数）

    # 生成3个相关的时间序列
    t = np.arange(N)
    data = np.zeros((N, P))

    # 变量1: 带趋势和季节性
    data[:, 0] = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 20) + 0.5 * np.random.randn(N)

    # 变量2: 与变量1相关
    data[:, 1] = 15 + 0.03 * t + 2 * np.sin(2 * np.pi * t / 20 + np.pi / 4) + 0.3 * data[:, 0] + 0.5 * np.random.randn(
        N)

    # 变量3: 独立的模式
    data[:, 2] = 20 + 4 * np.sin(2 * np.pi * t / 30) + 0.5 * np.random.randn(N)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['Var1', 'Var2', 'Var3'])

    print(f"原始数据形状: {df.shape}")
    print(f"原始数据前5行:\n{df.head()}")

    # 可视化原始数据
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i, col in enumerate(df.columns):
        axes[i].plot(df[col])
        axes[i].set_title(f'{col} - 原始数据')
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

    # 步骤2: MSSA分解
    print("\n步骤2: 执行MSSA分解")
    window_length = 30
    n_components = 10

    mssa = MSSA(
        window_length=window_length,
        n_components=n_components,
        verbose=True
    )

    # 拟合MSSA
    mssa.fit(df.values)

    # 步骤3: 获取MSSA分解结果
    print("\n步骤3: 获取MSSA分解的各个分量")

    # 为每个分量创建单独的组
    groups = {}
    for i in range(n_components):
        groups[f'Component_{i}'] = [i]

    # 重构各个分量
    reconstructed = mssa.reconstruct(groups)
    print(f"MSSA重构数据形状: {reconstructed.shape}")
    print(f"形状含义: (时间步数={N}, 原始变量数={P}, 分量数={n_components})")

    # 步骤4: 转换为Transformer输入格式
    print("\n步骤4: 转换MSSA输出为Transformer输入")

    # 方法1: 将所有分量展平为特征
    print("\n方法1: 展平所有分量作为特征")
    # 原始形状: (N, P, n_components) = (200, 3, 10)
    # 目标形状: (N, P*n_components) = (200, 30)

    # 重新排列维度并展平
    mssa_features_v1 = reconstructed.transpose(0, 2, 1).reshape(N, -1)
    print(f"Transformer输入形状 (方法1): {mssa_features_v1.shape}")
    print(f"特征数: {mssa_features_v1.shape[1]} = {P}变量 × {n_components}分量")

    # 创建特征名称
    feature_names_v1 = []
    for comp in range(n_components):
        for var in range(P):
            feature_names_v1.append(f'Var{var + 1}_Comp{comp}')

    print(f"\n前10个特征名: {feature_names_v1[:10]}")

    # 方法2: 选择主要分量
    print("\n\n方法2: 只选择主要分量")
    n_main_components = 5  # 只使用前5个主要分量

    # 计算各分量的贡献率
    component_contributions = mssa.component_weights[:n_main_components]
    print(f"前{n_main_components}个分量的贡献率: {component_contributions}")
    print(f"累积贡献率: {np.cumsum(component_contributions)}")

    # 只使用主要分量
    mssa_features_v2 = reconstructed[:, :, :n_main_components].transpose(0, 2, 1).reshape(N, -1)
    print(f"Transformer输入形状 (方法2): {mssa_features_v2.shape}")

    # 方法3: 分组聚合
    print("\n\n方法3: 按物理意义分组聚合")

    # 定义有意义的分组
    physical_groups = {
        'Trend': [0, 1],  # 趋势分量
        'Seasonal': [2, 3, 4],  # 季节性分量
        'Noise': [5, 6, 7, 8, 9]  # 噪声分量
    }

    # 重构分组
    grouped_reconstructed = mssa.reconstruct(physical_groups)
    print(f"分组重构形状: {grouped_reconstructed.shape}")

    # 转换为特征 (N, P*n_groups)
    n_groups = len(physical_groups)
    mssa_features_v3 = grouped_reconstructed.transpose(0, 2, 1).reshape(N, -1)
    print(f"Transformer输入形状 (方法3): {mssa_features_v3.shape}")

    # 步骤5: 展示具体数据示例
    print("\n步骤5: 具体数据示例")

    # 原始数据示例
    print("\n原始数据 (前5行):")
    print(df.head())

    # MSSA特征示例（方法1）
    print("\n\nMSSA特征 - 方法1 (前5行，前10列):")
    mssa_df_v1 = pd.DataFrame(mssa_features_v1[:5, :10],
                              columns=feature_names_v1[:10])
    print(mssa_df_v1)

    # 步骤6: 处理缺失值场景
    print("\n\n步骤6: 处理缺失值场景")

    # 添加一些缺失值
    data_with_missing = df.copy()
    missing_mask = np.random.random((N, P)) < 0.1  # 10%缺失
    data_with_missing[missing_mask] = np.nan

    print(f"缺失值数量: {missing_mask.sum()}")

    # 对缺失数据进行插值（用于MSSA）
    data_interpolated = data_with_missing.interpolate(method='linear', limit_direction='both')

    # 在插值数据上执行MSSA
    mssa_missing = MSSA(window_length=window_length, n_components=n_components)
    mssa_missing.fit(data_interpolated.values)

    # 获取MSSA特征
    reconstructed_missing = mssa_missing.reconstruct(groups)
    mssa_features_missing = reconstructed_missing.transpose(0, 2, 1).reshape(N, -1)

    print(f"处理缺失值后的MSSA特征形状: {mssa_features_missing.shape}")

    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 原始数据vs带缺失数据
    axes[0, 0].plot(df['Var1'], label='原始')
    axes[0, 0].plot(data_with_missing['Var1'], '.', label='带缺失')
    axes[0, 0].set_title('变量1: 原始 vs 缺失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MSSA分量示例
    axes[0, 1].plot(reconstructed[:, 0, 0], label='分量0')
    axes[0, 1].plot(reconstructed[:, 0, 1], label='分量1')
    axes[0, 1].plot(reconstructed[:, 0, 2], label='分量2')
    axes[0, 1].set_title('变量1的前3个MSSA分量')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # MSSA特征示例
    axes[1, 0].plot(mssa_features_v1[:, 0], label='特征0')
    axes[1, 0].plot(mssa_features_v1[:, 1], label='特征1')
    axes[1, 0].set_title('Transformer输入特征示例')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 特征分布
    axes[1, 1].hist(mssa_features_v1.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('所有MSSA特征的分布')
    axes[1, 1].set_xlabel('特征值')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return df, mssa, mssa_features_v1, mssa_features_missing


def explain_data_flow():
    """
    详细解释数据流程
    """
    print("\n" + "=" * 60)
    print("MSSA到Transformer的数据流程总结")
    print("=" * 60)

    print("""
    1. 原始数据: shape = (N, P)
       - N: 时间步数
       - P: 变量/特征数

    2. MSSA分解: 
       - 输入: (N, P)
       - 输出: (N, P, K) 其中K是分量数
       - 每个变量被分解为K个分量

    3. 特征转换（三种方法）:

       方法1 - 全部展平:
       - 转换: (N, P, K) → (N, P×K)
       - 优点: 保留所有信息
       - 缺点: 特征维度高

       方法2 - 选择主要分量:
       - 转换: (N, P, K) → (N, P, k) → (N, P×k) 其中k<K
       - 优点: 降维，去噪
       - 缺点: 可能丢失信息

       方法3 - 物理分组:
       - 转换: (N, P, K) → (N, P, G) → (N, P×G) 其中G是组数
       - 优点: 有物理意义，可解释性强
       - 缺点: 需要领域知识

    4. Transformer输入:
       - 形状: (N, D) 其中D是转换后的特征维度
       - 序列处理: 滑动窗口创建 (batch, seq_len, D)
    """)

    # 创建流程图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.9, "数据流程图", ha='center', fontsize=16, weight='bold')

    # 绘制流程
    boxes = [
        (0.1, 0.7, "原始数据\n(200, 3)"),
        (0.3, 0.7, "MSSA分解\n窗口=30"),
        (0.5, 0.7, "重构分量\n(200, 3, 10)"),
        (0.7, 0.8, "方法1: 展平\n(200, 30)"),
        (0.7, 0.7, "方法2: 主分量\n(200, 15)"),
        (0.7, 0.6, "方法3: 分组\n(200, 9)"),
        (0.9, 0.7, "Transformer\n输入特征")
    ]

    for x, y, text in boxes:
        ax.text(x, y, text, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 添加箭头
    arrows = [
        (0.18, 0.7, 0.08, 0),
        (0.38, 0.7, 0.08, 0),
        (0.58, 0.75, 0.08, 0.03),
        (0.58, 0.7, 0.08, 0),
        (0.58, 0.65, 0.08, -0.03),
        (0.78, 0.7, 0.08, 0)
    ]

    for x, y, dx, dy in arrows:
        ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, fc='black', ec='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1)
    ax.axis('off')
    plt.show()


def create_transformer_ready_data(data, mssa_model, method='full'):
    """
    将数据转换为Transformer可用的格式

    参数:
        data: 原始数据 (N, P)
        mssa_model: 训练好的MSSA模型
        method: 'full', 'main', 'grouped'

    返回:
        transformer_features: Transformer输入特征
    """
    # 获取所有分量
    groups = {}
    for i in range(mssa_model.n_components):
        groups[f'comp_{i}'] = [i]

    reconstructed = mssa_model.reconstruct(groups)
    N, P, K = reconstructed.shape

    if method == 'full':
        # 使用所有分量
        features = reconstructed.transpose(0, 2, 1).reshape(N, -1)

    elif method == 'main':
        # 只使用主要分量（累积贡献率>0.9）
        cum_weights = np.cumsum(mssa_model.component_weights)
        n_main = np.argmax(cum_weights > 0.9) + 1
        features = reconstructed[:, :, :n_main].transpose(0, 2, 1).reshape(N, -1)

    elif method == 'grouped':
        # 按物理意义分组
        physical_groups = {
            'trend': [0, 1],
            'seasonal': [2, 3, 4, 5],
            'residual': list(range(6, K))
        }
        grouped = mssa_model.reconstruct(physical_groups)
        features = grouped.transpose(0, 2, 1).reshape(N, -1)

    return features


if __name__ == "__main__":
    # 运行演示
    print("开始MSSA-Transformer连接演示...\n")

    # 主要演示
    df, mssa, features_full, features_missing = demonstrate_mssa_transformer_connection()

    # 数据流程说明
    explain_data_flow()

    # 实际应用示例
    print("\n" + "=" * 60)
    print("实际应用示例")
    print("=" * 60)

    # 测试不同转换方法
    for method in ['full', 'main', 'grouped']:
        features = create_transformer_ready_data(df.values, mssa, method=method)
        print(f"\n方法 '{method}' - 特征形状: {features.shape}")
        print(f"特征维度: {features.shape[1]}")
        print(f"内存占用: {features.nbytes / 1024:.2f} KB")

    print("\n演示完成！")
