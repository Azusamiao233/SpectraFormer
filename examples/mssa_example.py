import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectraformer.mssa import MSSA, setup_chinese_fonts
from matplotlib.colors import Normalize


def visualize_3d_matrix(matrix: np.ndarray, title: str = "3D Matrix Visualization") -> None:
    """
    可视化三维矩阵的切片和投影

    参数:
        matrix (np.ndarray): 形状为 (depth, height, width) 的三维矩阵
        title (str): 图表标题
    """
    # 验证输入维度
    if matrix.ndim != 3:
        raise ValueError("输入必须是三维矩阵")

    depth, height, width = matrix.shape
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    # 正视图（YZ平面切片）
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_title("Front View (YZ Plane Slices)")
    for z in range(depth):
        slice_yz = matrix[z, :, :]
        x = np.full((height, width), z)
        y, w = np.mgrid[0:height, 0:width]
        ax1.plot_surface(x, y, w, facecolors=plt.cm.viridis(slice_yz / np.max(matrix)), alpha=0.7)

    # 侧视图（XZ平面切片）
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_title("Side View (XZ Plane Slices)")
    for y in range(height):
        slice_xz = matrix[:, y, :]
        x, z = np.mgrid[0:depth, 0:width]
        y_plane = np.full((depth, width), y)
        ax2.plot_surface(x, y_plane, z, facecolors=plt.cm.plasma(slice_xz / np.max(matrix)), alpha=0.7)

    # 顶视图（XY平面投影）
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.set_title("Top View (XY Plane Projection)")
    x, y = np.mgrid[0:depth, 0:height]
    z_proj = np.mean(matrix, axis=2)  # Z轴均值投影
    ax3.plot_surface(x, y, z_proj, cmap='inferno', alpha=0.8)

    # 体积投影（正交投影）
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title("Orthographic Volume Projection")
    vol_proj = np.max(matrix, axis=0)  # 最大强度投影
    im = ax4.imshow(vol_proj, cmap='viridis', norm=Normalize(vmin=matrix.min(), vmax=matrix.max()))
    plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_xlabel('Width')
    ax4.set_ylabel('Height')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def print_ndarray_info(arr: np.ndarray) -> None:
    """
    打印ndarray的详细信息

    参数:
        arr (np.ndarray): 要分析的numpy数组

    异常:
        TypeError: 如果输入不是numpy数组
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("输入必须是numpy ndarray类型")

    print("ndarray信息概览：")
    print(f"形状 (shape): {arr.shape}")  # 数组维度元组
    print(f"维度 (ndim): {arr.ndim}")  # 数组维度数
    print(f"数据类型 (dtype): {arr.dtype}")  # 元素数据类型
    print(f"元素总数 (size): {arr.size}")  # 总元素个数
    print(f"元素字节大小 (itemsize): {arr.itemsize} bytes")  # 单个元素字节数

    # 统计信息（仅数值类型数组）
    if np.issubdtype(arr.dtype, np.number):
        print(f"最小值: {arr.min()}")
        print(f"最大值: {arr.max()}")
        print(f"平均值: {arr.mean():.4f}")  # 保留4位小数
        print(f"标准差: {arr.std():.4f}")
    else:
        print("非数值类型数组，跳过统计计算")
# 1. 生成模拟数据
def generate_multivariate_test_data(N=500, P=3):
    """
    生成多变量时间序列测试数据

    参数:
        N: 序列长度
        P: 变量数量

    返回:
        X: 形状为(N, P)的原始数据
        components: 形状为(N, P, 3)的成分数据 (趋势, 周期, 噪声)
    """
    t = np.arange(N)

    # 初始化存储数组
    X = np.zeros((N, P))
    components = np.zeros((N, P, 3))  # 趋势, 周期, 噪声

    for p in range(P):
        # 1. 趋势成分: 不同的多项式趋势
        trend = 0.01 * (p + 1) * t + 0.0002 * (p + 1) * t ** 2
        components[:, p, 0] = trend

        # 2. 周期成分: 不同的频率和相位
        period = 3.0 * np.sin(2 * np.pi * t / (30.0 + 5 * p)) + 1.5 * np.sin(2 * np.pi * t / (80.0 - 10 * p))
        components[:, p, 1] = period

        # 3. 噪声成分: 不同的方差
        noise = (0.5 + 0.2 * p) * np.random.randn(N)
        components[:, p, 2] = noise

        # 合并所有成分
        X[:, p] = trend + period + noise

    return X, components


# 2. 实际使用示例
def run_mssa_example():
    # 生成测试数据
    N = 500  # 数据长度
    P = 3  # 变量数量
    X, true_components = generate_multivariate_test_data(N, P)

    # 定义时间序列名称
    series_names = [f'Series {i + 1}' for i in range(P)]

    # 可视化原始多变量时间序列
    plt.figure(figsize=(12, 8))
    for p in range(P):
        plt.subplot(P, 1, p + 1)
        plt.plot(X[:, p])
        plt.title(series_names[p])
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle('原始多变量时间序列数据', fontsize=16, y=1.02)
    plt.show()

    # 配置MSSA
    window_length = 100  # 窗口长度(通常L需要根据数据特点设置，一般建议 N/5 < L < N/2)

    # 定义分组 (通常我们无法预先知道哪些分量对应趋势、周期等，这里只是演示)
    # 通常需要通过检查奇异谱、奇异向量的周期性以及w-相关矩阵来确定分组
    groups = {
        'trend': [0, 1],  # 低频分量通常对应趋势
        'periodic': [2, 3, 4, 5],  # 成对的分量通常对应周期信号
        'noise': list(range(6, 20))  # 高频分量通常对应噪声
    }

    # 创建MSSA对象并拟合数据
    mssa = MSSA(window_length=window_length, groups=groups, verbose=True)
    mssa.fit(X)

    # 绘制奇异谱
    #mssa.plot_singular_spectrum(figsize=(12, 5))

    # 重构时间序列
    reconstructed = mssa.reconstruct()
    print(print_ndarray_info(reconstructed))
    visualize_3d_matrix(reconstructed, title="3D Gaussian-like Matrix Visualization")

    # 可视化重构结果与原始数据对比
    """mssa.plot_reconstructed(
        original_data=X,
        series_names=series_names,
        figsize=(15, 10)
    )"""

    # 计算并可视化w-相关矩阵
    #mssa.plot_w_correlation(figsize=(8, 6))

    # 可视化真实成分与重构成分的比较
    plt.figure(figsize=(15, 12))
    component_names = ['趋势', '周期', '噪声']

    for p in range(P):
        for c in range(3):
            plt.subplot(P, 3, p * 3 + c + 1)
            if c < 2:  # 趋势和周期分量
                plt.plot(true_components[:, p, c], label='真实', color='black', alpha=0.7)
                plt.plot(reconstructed[:, p, c], label='重构', color='red')
                plt.title(f'{series_names[p]} - {component_names[c]}')
            else:  # 噪声分量
                plt.plot(true_components[:, p, c], label='真实噪声', alpha=0.5)
                plt.plot(reconstructed[:, p, 2], label='重构噪声', alpha=0.5)
                plt.title(f'{series_names[p]} - 噪声')
            plt.grid(True)
            plt.legend()

    plt.tight_layout()
    plt.suptitle('真实成分与重构成分对比', fontsize=16, y=1.02)
    plt.show()

    # 返回MSSA对象供进一步分析
    return mssa, X, true_components


# 3. 使用真实数据 (例如CSV文件)
def run_with_real_data(file_path, window_length=50, n_components=None):
    """
    使用实际数据运行MSSA

    参数:
        file_path (str): CSV文件路径
        window_length (int): 窗口长度
        n_components (int, optional): 要保留的分量数量
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, index_col=0)  # 假设第一列是序号
        print(f"成功加载数据: {df.shape[0]}行, {df.shape[1]}列")

        # 确保数据为浮点数类型
        df = df.astype(float)

        # 创建并拟合MSSA模型
        mssa = MSSA(window_length=window_length, n_components=n_components, verbose=True)
        mssa.fit(df)

        # 绘制奇异谱
        mssa.plot_singular_spectrum()

        # 根据奇异谱，建议分组
        # 检查奇异值的下降趋势以判断主要分量
        sigma = mssa.sigma
        diffs = np.diff(sigma)

        # 查找奇异值陡降的位置
        significant_drops = np.where(diffs / sigma[:-1] < -0.3)[0]

        # 如果没有明显陡降，则尝试找到贡献率达到80%和95%的分量数量
        cum_contrib = np.cumsum(mssa.component_weights)
        comp_80 = np.argmax(cum_contrib >= 0.8) + 1
        comp_95 = np.argmax(cum_contrib >= 0.95) + 1

        print(f"建议的主要成分数量:")
        print(f"  - 基于奇异值陡降: {significant_drops + 1 if len(significant_drops) > 0 else '未发现明显陡降'}")
        print(f"  - 累积贡献率80%: 前{comp_80}个分量")
        print(f"  - 累积贡献率95%: 前{comp_95}个分量")

        # 根据分析建议分组
        suggested_groups = {
            'trend': list(range(min(3, len(sigma)))),  # 趋势通常由前几个分量捕获
            'signal': list(range(3, min(comp_80, len(sigma)))),  # 主要信号
            'noise': list(range(comp_80, min(comp_95, len(sigma))))  # 噪声部分
        }

        print("\n建议的分组:")
        for group_name, indices in suggested_groups.items():
            print(f"  - {group_name}: {indices}")

        # 使用建议的分组进行重构
        reconstructed = mssa.reconstruct(suggested_groups)

        # 可视化重构结果与原始数据对比
        mssa.plot_reconstructed(
            original_data=df.values,
            group_names=list(suggested_groups.keys()),
            series_names=df.columns.tolist()
        )

        # 计算并可视化w-相关矩阵
        mssa.plot_w_correlation()

        return mssa, df

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except Exception as e:
        print(f"处理数据时发生错误: {e}")


# 运行示例
if __name__ == "__main__":
    setup_chinese_fonts()#设置字体
    mssa, X, _ = run_mssa_example()

    #如果有真实数据，取消下面的注释
    #数据文件应该是CSV格式，第一列是序号，其他列是多变量时间序列
    #mssa_real, df = run_with_real_data('swat.csv', window_length=50)
