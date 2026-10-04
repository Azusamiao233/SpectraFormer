import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.linalg import hankel
from typing import List, Tuple, Union, Dict, Optional
import platform


# 设置中文字体
def setup_chinese_fonts():
    """设置matplotlib的中文字体"""
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    system = platform.system()

    if system == 'Windows':
        try:
            # Windows下使用微软雅黑或者黑体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
            return True
        except:
            pass
    elif system == 'Darwin':  # macOS
        try:
            # Mac下使用苹方字体
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Apple LiGothic Medium', 'STHeiti']
            return True
        except:
            pass
    else:  # Linux
        try:
            # Linux下使用文泉驿或Noto字体
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC']
            return True
        except:
            pass

    # 如果上述字体都无法使用，尝试matplotlib自带的字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    return False

class MSSA:
    """
    多元奇异谱分析 (Multivariate Singular Spectrum Analysis)

    参数:
        window_length (int): 窗口长度 M，应该满足 2 < M < N/2
        n_components (int, optional): 要提取的分量数，默认为 None（保留所有分量）
        groups (dict, optional): 定义分组的字典，例如 {'trend': [0, 1], 'seasonal': [2, 3, 4]}
        verbose (bool): 是否打印中间步骤信息
    """

    def __init__(self, window_length: int,
                 n_components: Optional[int] = None,
                 groups: Optional[Dict[str, List[int]]] = None,
                 verbose: bool = False):
        self.window_length = window_length
        self.n_components = n_components
        self.groups = groups
        self.verbose = verbose

        # 在 fit 过程中设置的属性
        self.n_series = None  # 时间序列数量 P
        self.series_length = None  # 时间序列长度 N
        self.K = None  # K = N - M + 1
        self.trajectory_matrix = None  # 轨迹矩阵 Y
        self.U = None  # 左奇异向量矩阵 U
        self.sigma = None  # 奇异值
        self.Vt = None  # 右奇异向量矩阵 V^T
        self.reconstructed = None  # 重构后的时间序列
        self.component_weights = None  # 各分量的权重

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        对多变量时间序列数据进行MSSA拟合

        参数:
            X (array-like): 形状为 (N, P) 的时间序列数据，N是长度，P是变量数
                           或pandas DataFrame，列为变量

        返回:
            self: 返回MSSA对象实例
        """
        # 将输入转换为numpy数组，确保形状为 (N, P)
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.series_length, self.n_series = X.shape  # N, P
        self.K = self.series_length - self.window_length + 1  # K = N - M + 1

        if self.verbose:
            print(f"数据形状: {X.shape} (N={self.series_length}, P={self.n_series})")
            print(f"窗口长度 M: {self.window_length}")
            print(f"K = N - M + 1: {self.K}")

        # 步骤1: 嵌入(Embedding) - 构建轨迹矩阵Y
        self.trajectory_matrix = self._embedding(X)

        if self.verbose:
            print(
                f"轨迹矩阵形状: {self.trajectory_matrix.shape} (M={self.window_length}, P*K={self.n_series * self.K})")

        # 步骤2: 奇异值分解(SVD)
        self._decompose()

        return self

    def _embedding(self, X: np.ndarray) -> np.ndarray:
        """
        嵌入步骤: 将多变量时间序列转换为轨迹矩阵

        原理: 构建向量 Y_j^(p) = (x_j^(p), ..., x_(j+M-1)^(p))^T ∈ R^M
              然后将所有 P 个轨迹矩阵组合成一个块 Hankel 轨迹矩阵 Y
              其定义为 Y = [Y^(1)^T, Y^(2)^T, ..., Y^(P)^T]^T

        参数:
            X (np.ndarray): 形状为 (N, P) 的时间序列数据

        返回:
            np.ndarray: 轨迹矩阵 Y，形状为 (M, P*K)
        """
        M = self.window_length
        N, P = X.shape
        K = N - M + 1

        # 初始化轨迹矩阵（大小为 M x PK）
        Y = np.zeros((M, P * K))

        # 对每个时间序列构建轨迹矩阵
        for p in range(P):
            # 构建当前时间序列的 Hankel 矩阵
            series = X[:, p]
            Y_p = np.zeros((M, K))

            for i in range(K):
                Y_p[:, i] = series[i:i + M]

            # 将该序列的轨迹矩阵添加到总轨迹矩阵中
            Y[:, p * K:(p + 1) * K] = Y_p

        return Y

    def _decompose(self):
        """
        对轨迹矩阵进行奇异值分解(SVD)

        原理: Y = α^(1/2) * U * Λ * V^T，其中 α = PK
        """
        M = self.window_length
        PK = self.n_series * self.K
        alpha = PK  # 归一化因子

        # 使用经济型SVD (full_matrices=False)，避免大矩阵计算问题
        try:
            U, sigma, Vt = np.linalg.svd(self.trajectory_matrix, full_matrices=False)

            # 确定要保留的分量数
            d = min(M, PK)
            if self.n_components is None or self.n_components > d:
                self.n_components = d

            # 保留前n_components个分量
            self.U = U[:, :self.n_components]
            self.sigma = sigma[:self.n_components]
            self.Vt = Vt[:self.n_components, :]

            # 计算各分量的权重（贡献率）
            total_weight = np.sum(np.square(sigma))
            self.component_weights = np.square(self.sigma) / total_weight

            if self.verbose:
                print(f"SVD完成: 提取了{self.n_components}个分量")
                print(f"U形状: {self.U.shape}, sigma形状: {self.sigma.shape}, Vt形状: {self.Vt.shape}")
                print(f"前10个奇异值: {self.sigma[:min(10, len(self.sigma))]}")
                print(f"前10个分量权重: {self.component_weights[:min(10, len(self.component_weights))]}")

        except np.linalg.LinAlgError as e:
            print(f"SVD计算失败: {e}")
            print("尝试使用full_matrices=False和减小窗口长度来解决大矩阵SVD问题")
            raise
        except MemoryError:
            print("内存不足无法进行SVD计算，尝试减小窗口长度或使用更有效的SVD方法")
            raise

    def reconstruct(self, groups=None):
        """
        根据选定的分量或分组重构时间序列

        参数:
            groups (dict, optional): 分量分组信息，如 {'trend': [0, 1], 'seasonal': [2, 3, 4]}
                                    如果为None，将使用初始化时设定的分组

        返回:
            np.ndarray: 重构的时间序列，形状为(N, P, n_groups)
                       如果未指定分组，n_groups=1
        """
        if groups is None:
            groups = self.groups

        if groups is None:
            # 没有指定分组时，将所有分量视为一组
            groups = {'all': list(range(self.n_components))}

        N, P = self.series_length, self.n_series
        M = self.window_length
        K = N - M + 1

        # 初始化存储重构结果的数组，形状为 (N, P, n_groups)
        reconstructed = np.zeros((N, P, len(groups)))

        for idx, (group_name, component_indices) in enumerate(groups.items()):
            if self.verbose:
                print(f"重构分组 '{group_name}' 使用分量: {component_indices}")

            # 步骤3: 分组 - 仅使用指定分量构建重构矩阵
            R_group = np.zeros((M, P * K))

            for i in component_indices:
                if i < self.n_components:
                    # 构建部分重构矩阵
                    elementary_matrix = self.sigma[i] * np.outer(self.U[:, i], self.Vt[i, :])
                    R_group += elementary_matrix

            # 步骤4: 对角平均 - 将重构矩阵转换回原始时间序列
            for p in range(P):
                # 提取当前序列的重构矩阵
                R_p = R_group[:, p * K:(p + 1) * K]

                # 执行对角平均
                reconstructed[:, p, idx] = self._diagonal_averaging(R_p)

        self.reconstructed = reconstructed
        return reconstructed

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """
        对角平均化: 将矩阵转换为时间序列

        原理:
        x̃_j = {
            1/j ∑ r_t,j-t+1              1 ≤ j < m*
            1/m* ∑ r_t,j-t+1            m* ≤ j < k*
            1/(N-j+1) ∑ r_t,j-t+1       k* ≤ j < N
        }

        参数:
            matrix (np.ndarray): 重构矩阵 R_p，形状为 (M, K)

        返回:
            np.ndarray: 重构的时间序列，长度为 N
        """
        M, K = matrix.shape
        N = M + K - 1  # 原始时间序列长度 N = M + K - 1

        m_star = min(M, K)
        k_star = max(M, K)

        # 初始化重构的时间序列
        reconstructed_ts = np.zeros(N)

        # 第一部分: 1 ≤ j < m*
        for j in range(1, m_star):
            s = 0
            for t in range(1, j + 1):
                # 矩阵索引从0开始，故需要调整
                s += matrix[t - 1, j - t]
            reconstructed_ts[j - 1] = s / j

        # 第二部分: m* ≤ j < k*
        for j in range(m_star, k_star):
            s = 0
            for t in range(1, m_star + 1):
                s += matrix[t - 1, j - t]
            reconstructed_ts[j - 1] = s / m_star

        # 第三部分: k* ≤ j < N
        for j in range(k_star, N + 1):
            s = 0
            for t in range(j - k_star + 1, j - k_star + m_star + 1):
                # 注意索引边界
                if t <= M and (j - t) < K:
                    s += matrix[t - 1, j - t - 1]
            reconstructed_ts[j - 1] = s / (N - j + 1)

        return reconstructed_ts

    def w_correlation(self, components: List[int] = None) -> np.ndarray:
        """
        计算不同重构分量之间的w-相关系数

        参数:
            components (List[int], optional): 要计算w-相关的分量索引列表，
                                             如果为None，则计算所有分量

        返回:
            np.ndarray: w-相关系数矩阵
        """
        if not hasattr(self, 'reconstructed') or self.reconstructed is None:
            raise ValueError("请先调用reconstruct()方法生成重构分量")

        N = self.series_length
        P = self.n_series
        M = self.window_length

        if components is None:
            # 使用所有分组的所有分量
            n_comps = self.reconstructed.shape[2]
            component_indices = range(n_comps)
        else:
            component_indices = components
            n_comps = len(component_indices)

        # 初始化w-相关矩阵
        w_corr = np.zeros((n_comps, n_comps))

        # 构建权重向量w
        w = np.array([min(i + 1, M, N - i) for i in range(N)])

        # 对每对分量计算w-相关系数
        for i, i_idx in enumerate(component_indices):
            for j, j_idx in enumerate(component_indices):
                if i <= j:  # 利用相关矩阵的对称性
                    # 计算P个时间序列的平均w-相关
                    corr_sum = 0
                    for p in range(P):
                        X = self.reconstructed[:, p, i_idx]
                        Y = self.reconstructed[:, p, j_idx]

                        # 计算加权内积(X, Y)_w
                        weighted_product = np.sum(w * X * Y)

                        # 计算加权范数||X||_w和||Y||_w
                        norm_X = np.sqrt(np.sum(w * X * X))
                        norm_Y = np.sqrt(np.sum(w * Y * Y))

                        # 计算w-相关
                        if norm_X > 0 and norm_Y > 0:
                            corr_sum += weighted_product / (norm_X * norm_Y)

                    # 计算平均w-相关
                    w_corr[i, j] = corr_sum / P
                    w_corr[j, i] = w_corr[i, j]  # 对称矩阵

        return w_corr

    def plot_reconstructed(self, original_data=None, group_names=None, series_names=None,
                           figsize=(12, 10), ncols=1):
        """
        绘制原始时间序列和重构的时间序列

        参数:
            original_data (np.ndarray, optional): 原始时间序列数据，形状为(N, P)
            group_names (list, optional): 分组名称列表
            series_names (list, optional): 序列名称列表
            figsize (tuple): 图形大小
            ncols (int): 每行的子图数量
        """
        if not hasattr(self, 'reconstructed') or self.reconstructed is None:
            raise ValueError("请先调用reconstruct()方法生成重构分量")

        N, P, G = self.reconstructed.shape

        if group_names is None:
            if self.groups is not None:
                group_names = list(self.groups.keys())
            else:
                group_names = [f'Component Group {i + 1}' for i in range(G)]

        if series_names is None:
            series_names = [f'Series {i + 1}' for i in range(P)]

        nrows = ((P + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for p in range(P):
            ax = axes[p]

            # 绘制原始数据
            if original_data is not None:
                ax.plot(original_data[:, p], 'k-', label='Original', alpha=0.7)

            # 绘制每个分组的重构数据
            for g in range(G):
                ax.plot(self.reconstructed[:, p, g], label=group_names[g])

            ax.set_title(f'{series_names[p]}')
            ax.grid(True)
            if p == 0 or p % ncols == 0:
                ax.set_ylabel('Value')
            if p >= P - ncols:
                ax.set_xlabel('Time')

        # 移除未使用的子图
        for i in range(P, len(axes)):
            fig.delaxes(axes[i])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=min(G + 1, 4))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('原始时间序列与MSSA重构的时间序列对比', fontsize=16)
        plt.show()

    def plot_w_correlation(self, components=None, group_names=None, figsize=(10, 8)):
        """
        绘制w相关系数热力图

        参数:
            components (List[int], optional): 要计算w-相关的分量索引列表
            group_names (list, optional): 分组名称列表
            figsize (tuple): 图形大小
        """
        w_corr = self.w_correlation(components)

        if group_names is None and self.groups is not None:
            group_names = list(self.groups.keys())
        elif group_names is None:
            group_names = [f'Group {i + 1}' for i in range(w_corr.shape[0])]

        plt.figure(figsize=figsize)
        plt.imshow(np.abs(w_corr), cmap='viridis', origin='upper', vmin=0, vmax=1)
        plt.colorbar(label='|W-correlation|')

        plt.xticks(np.arange(len(group_names)), group_names, rotation=45)
        plt.yticks(np.arange(len(group_names)), group_names)

        plt.title('重构分量间的W-相关系数矩阵', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_singular_spectrum(self, figsize=(10, 6), style='bar', log_scale=False):
        """
        绘制奇异谱（奇异值和累积贡献率）

        参数:
            figsize (tuple): 图形大小
            style (str): 'bar' 或 'line'，指定绘图样式
            log_scale (bool): 是否使用对数刻度
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 计算累积贡献率
        cum_weights = np.cumsum(self.component_weights)

        # 绘制奇异值
        x = np.arange(len(self.sigma)) + 1
        if style == 'bar':
            ax1.bar(x, self.sigma, alpha=0.7)
        else:
            ax1.plot(x, self.sigma, 'o-')

        if log_scale:
            ax1.set_yscale('log')

        ax1.set_title('奇异值谱', fontsize=12)
        ax1.set_xlabel('分量索引')
        ax1.set_ylabel('奇异值')
        ax1.grid(True)

        # 绘制累积贡献率
        ax2.plot(x, cum_weights, 'o-', color='r')
        ax2.axhline(y=0.8, linestyle='--', color='gray', alpha=0.7)
        ax2.axhline(y=0.9, linestyle='--', color='gray', alpha=0.7)
        ax2.axhline(y=0.95, linestyle='--', color='gray', alpha=0.7)

        ax2.set_title('累积贡献率', fontsize=12)
        ax2.set_xlabel('分量索引')
        ax2.set_ylabel('累积贡献率')
        ax2.set_ylim(0, 1.02)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()