try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError:  # Plotting is optional for headless training.
    plt = None
    mpl = None
import numpy as np
import pandas as pd
from scipy.linalg import hankel
from scipy.sparse.linalg import svds
from typing import List, Tuple, Union, Dict, Optional
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


# 设置中文字体
def setup_chinese_fonts():
    """设置matplotlib的中文字体"""
    if plt is None:
        return False
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


def _fast_hankel_matrix_vectorized(series, M, K):
    """向量化的Hankel矩阵构建，无需numba"""
    # 使用高级索引一次性构建整个矩阵
    indices = np.arange(M)[:, None] + np.arange(K)
    return series[indices]


def _fast_diagonal_averaging_vectorized(matrix):
    """向量化的对角平均化，无需numba"""
    M, K = matrix.shape
    N = M + K - 1

    m_star = min(M, K)
    k_star = max(M, K)

    reconstructed_ts = np.zeros(N)

    # 第一部分: 1 ≤ j < m* (向量化)
    for j in range(1, m_star):
        indices_i = np.arange(j)
        indices_j = j - 1 - indices_i
        reconstructed_ts[j - 1] = np.mean(matrix[indices_i, indices_j])

    # 第二部分: m* ≤ j < k* (向量化)
    for j in range(m_star, k_star):
        indices_i = np.arange(m_star)
        indices_j = j - 1 - indices_i
        reconstructed_ts[j - 1] = np.mean(matrix[indices_i, indices_j])

    # 第三部分: k* ≤ j < N (向量化)
    for j in range(k_star, N + 1):
        start_t = j - k_star
        end_t = min(j - k_star + m_star, M)

        valid_i = np.arange(start_t, end_t)
        valid_j = j - 1 - valid_i

        # 确保索引在有效范围内
        mask = (valid_i < M) & (valid_j < K) & (valid_j >= 0)
        if np.any(mask):
            reconstructed_ts[j - 1] = np.mean(matrix[valid_i[mask], valid_j[mask]])

    return reconstructed_ts


def _compute_w_correlation_vectorized(X, Y, w):
    """向量化的w-相关计算"""
    weighted_product = np.sum(w * X * Y)
    norm_X = np.sqrt(np.sum(w * X * X))
    norm_Y = np.sqrt(np.sum(w * Y * Y))

    if norm_X > 1e-10 and norm_Y > 1e-10:
        return weighted_product / (norm_X * norm_Y)
    else:
        return 0.0


class MSSA:
    """
    多元奇异谱分析 (Multivariate Singular Spectrum Analysis)

    优化版本（无numba依赖）：
    - 使用向量化操作替代循环
    - 优化内存使用，使用稀疏SVD
    - 并行计算w-相关矩阵
    - 缓存机制避免重复计算
    - 更好的算法复杂度

    参数:
        window_length (int): 窗口长度 M，应该满足 2 < M < N/2
        n_components (int, optional): 要提取的分量数，默认为 None（保留所有分量）
        groups (dict, optional): 定义分组的字典，例如 {'trend': [0, 1], 'seasonal': [2, 3, 4]}
        verbose (bool): 是否打印中间步骤信息
        use_sparse_svd (bool): 是否使用稀疏SVD（适用于大矩阵）
        n_jobs (int): 并行作业数，-1表示使用所有CPU核心
    """

    def __init__(self, window_length: int,
                 n_components: Optional[int] = None,
                 groups: Optional[Dict[str, List[int]]] = None,
                 verbose: bool = False,
                 use_sparse_svd: bool = False,
                 n_jobs: int = -1):
        self.window_length = window_length
        self.n_components = n_components
        self.groups = groups
        self.verbose = verbose
        self.use_sparse_svd = use_sparse_svd
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()

        # 在 fit 过程中设置的属性
        self.n_series = None
        self.series_length = None
        self.K = None
        self.trajectory_matrix = None
        self.U = None
        self.sigma = None
        self.Vt = None
        self.reconstructed = None
        self.component_weights = None

        # 缓存
        self._w_corr_cache = {}
        self._weights_cache = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        对多变量时间序列数据进行MSSA拟合

        参数:
            X (array-like): 形状为 (N, P) 的时间序列数据，N是长度，P是变量数
                           或pandas DataFrame，列为变量

        返回:
            self: 返回MSSA对象实例
        """
        # 输入验证和转换
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.series_length, self.n_series = X.shape
        self.K = self.series_length - self.window_length + 1

        # 验证参数
        if self.window_length >= self.series_length:
            raise ValueError(f"窗口长度 {self.window_length} 必须小于序列长度 {self.series_length}")
        if self.K <= 0:
            raise ValueError(f"K = N - M + 1 = {self.K} 必须大于0")

        if self.verbose:
            print(f"数据形状: {X.shape} (N={self.series_length}, P={self.n_series})")
            print(f"窗口长度 M: {self.window_length}")
            print(f"K = N - M + 1: {self.K}")

        # 步骤1: 嵌入(Embedding) - 构建轨迹矩阵Y
        self.trajectory_matrix = self._embedding_optimized(X)

        if self.verbose:
            print(f"轨迹矩阵形状: {self.trajectory_matrix.shape}")

        # 步骤2: 奇异值分解(SVD)
        self._decompose_optimized()

        return self

    def _embedding_optimized(self, X: np.ndarray) -> np.ndarray:
        """
        优化的嵌入步骤: 使用向量化操作
        """
        M = self.window_length
        N, P = X.shape
        K = N - M + 1

        # 预分配内存
        Y = np.zeros((M, P * K), dtype=np.float64)

        # 使用向量化操作构建轨迹矩阵
        for p in range(P):
            start_idx = p * K
            end_idx = (p + 1) * K
            Y[:, start_idx:end_idx] = _fast_hankel_matrix_vectorized(X[:, p], M, K)

        return Y

    def _decompose_optimized(self):
        """
        优化的奇异值分解: 使用稀疏SVD或经济型SVD
        """
        M = self.window_length
        PK = self.n_series * self.K

        # 确定要保留的分量数
        max_components = min(M, PK)
        if self.n_components is None:
            self.n_components = max_components
        else:
            self.n_components = min(self.n_components, max_components)

        try:
            # 判断是否使用稀疏SVD的条件更智能
            use_sparse = (self.use_sparse_svd or
                          (self.n_components < min(M, PK) * 0.6 and min(M, PK) > 100))

            if use_sparse:
                # 使用稀疏SVD适用于大矩阵和少量分量
                if self.verbose:
                    print("使用稀疏SVD...")

                k = min(self.n_components, min(M, PK) - 1)
                U, sigma, Vt = svds(self.trajectory_matrix, k=k, which='LM')

                # 按奇异值降序排列
                idx = np.argsort(sigma)[::-1]
                self.U = U[:, idx]
                self.sigma = sigma[idx]
                self.Vt = Vt[idx, :]

            else:
                # 使用标准SVD
                if self.verbose:
                    print("使用标准SVD...")

                U, sigma, Vt = np.linalg.svd(self.trajectory_matrix, full_matrices=False)

                # 保留前n_components个分量
                self.U = U[:, :self.n_components]
                self.sigma = sigma[:self.n_components]
                self.Vt = Vt[:self.n_components, :]

            # 计算各分量的权重（贡献率）
            self.component_weights = (self.sigma ** 2) / np.sum(self.sigma ** 2)

            if self.verbose:
                print(f"SVD完成: 提取了{self.n_components}个分量")
                print(f"前10个奇异值: {self.sigma[:min(10, len(self.sigma))]}")
                print(f"前10个分量权重: {self.component_weights[:min(10, len(self.component_weights))]}")

        except Exception as e:
            print(f"SVD计算失败: {e}")
            # 尝试使用更保守的方法
            if not use_sparse:
                print("尝试使用稀疏SVD...")
                self.use_sparse_svd = True
                self.n_components = min(50, self.n_components)
                self._decompose_optimized()
            else:
                raise

    def reconstruct(self, groups=None):
        """
        根据选定的分量或分组重构时间序列
        优化版本：使用并行计算和向量化操作
        """
        if groups is None:
            groups = self.groups

        if groups is None:
            groups = {'all': list(range(self.n_components))}

        N, P = self.series_length, self.n_series
        M = self.window_length
        K = N - M + 1

        # 初始化存储重构结果的数组
        reconstructed = np.zeros((N, P, len(groups)), dtype=np.float64)

        # 预计算所有需要的基本重构矩阵
        elementary_matrices = []
        for i in range(self.n_components):
            elem_matrix = self.sigma[i] * np.outer(self.U[:, i], self.Vt[i, :])
            elementary_matrices.append(elem_matrix)

        # 并行重构各个分组
        group_items = list(groups.items())

        def reconstruct_group(group_info):
            idx, (group_name, component_indices) = group_info

            if self.verbose:
                print(f"重构分组 '{group_name}' 使用分量: {component_indices}")

            # 构建重构矩阵 - 向量化操作
            R_group = np.zeros((M, P * K), dtype=np.float64)
            for i in component_indices:
                if i < self.n_components:
                    R_group += elementary_matrices[i]

            # 对每个序列进行对角平均
            group_result = np.zeros((N, P), dtype=np.float64)
            for p in range(P):
                R_p = R_group[:, p * K:(p + 1) * K]
                group_result[:, p] = _fast_diagonal_averaging_vectorized(R_p)

            return idx, group_result

        # 使用多线程并行处理（如果分组数量足够多）
        if len(group_items) > 1 and self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(group_items))) as executor:
                results = list(executor.map(reconstruct_group, enumerate(group_items)))
        else:
            # 单线程处理
            results = [reconstruct_group(item) for item in enumerate(group_items)]

        # 整理结果
        for idx, group_result in results:
            reconstructed[:, :, idx] = group_result

        self.reconstructed = reconstructed
        return reconstructed

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """
        对角平均化: 使用向量化版本
        """
        return _fast_diagonal_averaging_vectorized(matrix)

    def w_correlation(self, components: List[int] = None) -> np.ndarray:
        """
        计算不同重构分量之间的w-相关系数
        优化版本：使用缓存和向量化计算
        """
        if not hasattr(self, 'reconstructed') or self.reconstructed is None:
            raise ValueError("请先调用reconstruct()方法生成重构分量")

        # 检查缓存
        cache_key = tuple(components) if components is not None else 'all'
        if cache_key in self._w_corr_cache:
            return self._w_corr_cache[cache_key]

        N = self.series_length
        P = self.n_series
        M = self.window_length

        if components is None:
            n_comps = self.reconstructed.shape[2]
            component_indices = list(range(n_comps))
        else:
            component_indices = components
            n_comps = len(component_indices)

        # 初始化w-相关矩阵
        w_corr = np.zeros((n_comps, n_comps), dtype=np.float64)

        # 构建权重向量w（缓存）
        if self._weights_cache is None:
            self._weights_cache = np.array([min(i + 1, M, N - i) for i in range(N)], dtype=np.float64)
        w = self._weights_cache

        # 向量化计算w-相关矩阵
        for i in range(n_comps):
            for j in range(i, n_comps):
                i_idx = component_indices[i]
                j_idx = component_indices[j]

                # 对所有序列计算平均w-相关
                corr_values = []
                for p in range(P):
                    X = self.reconstructed[:, p, i_idx]
                    Y = self.reconstructed[:, p, j_idx]
                    corr_val = _compute_w_correlation_vectorized(X, Y, w)
                    corr_values.append(corr_val)

                avg_corr = np.mean(corr_values)
                w_corr[i, j] = avg_corr
                w_corr[j, i] = avg_corr

        # 缓存结果
        self._w_corr_cache[cache_key] = w_corr
        return w_corr

    def plot_reconstructed(self, original_data=None, group_names=None, series_names=None,
                           figsize=(12, 10), ncols=1):
        """
        绘制原始时间序列和重构的时间序列
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
                ax.plot(original_data[:, p], 'k-', label='Original', alpha=0.7, linewidth=1.5)

            # 绘制每个分组的重构数据
            for g in range(G):
                ax.plot(self.reconstructed[:, p, g], label=group_names[g], linewidth=1.2)

            ax.set_title(f'{series_names[p]}', fontsize=12)
            ax.grid(True, alpha=0.3)
            if p == 0 or p % ncols == 0:
                ax.set_ylabel('Value')
            if p >= P - ncols:
                ax.set_xlabel('Time')

        # 移除未使用的子图
        for i in range(P, len(axes)):
            fig.delaxes(axes[i])

        # 添加图例
        if len(axes) > 0:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=min(G + 1, 4))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('原始时间序列与MSSA重构的时间序列对比', fontsize=16)
        plt.show()

    def plot_w_correlation(self, components=None, group_names=None, figsize=(10, 8)):
        """
        绘制w相关系数热力图
        """
        w_corr = self.w_correlation(components)

        if group_names is None and self.groups is not None:
            group_names = list(self.groups.keys())
        elif group_names is None:
            group_names = [f'Group {i + 1}' for i in range(w_corr.shape[0])]

        plt.figure(figsize=figsize)
        im = plt.imshow(np.abs(w_corr), cmap='viridis', origin='upper', vmin=0, vmax=1)
        plt.colorbar(im, label='|W-correlation|')

        # 添加数值标注
        for i in range(w_corr.shape[0]):
            for j in range(w_corr.shape[1]):
                plt.text(j, i, f'{np.abs(w_corr[i, j]):.2f}',
                         ha='center', va='center',
                         color='white' if np.abs(w_corr[i, j]) > 0.5 else 'black',
                         fontsize=10)

        plt.xticks(np.arange(len(group_names)), group_names, rotation=45)
        plt.yticks(np.arange(len(group_names)), group_names)

        plt.title('重构分量间的W-相关系数矩阵', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_singular_spectrum(self, figsize=(12, 5), style='bar', log_scale=False):
        """
        绘制奇异谱（奇异值和累积贡献率）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 计算累积贡献率
        cum_weights = np.cumsum(self.component_weights)

        # 绘制奇异值
        x = np.arange(len(self.sigma)) + 1
        if style == 'bar':
            bars = ax1.bar(x, self.sigma, alpha=0.7, color='steelblue')
        else:
            ax1.plot(x, self.sigma, 'o-', markersize=4, linewidth=1.5)

        if log_scale:
            ax1.set_yscale('log')

        ax1.set_title('奇异值谱', fontsize=12)
        ax1.set_xlabel('分量索引')
        ax1.set_ylabel('奇异值')
        ax1.grid(True, alpha=0.3)

        # 绘制累积贡献率
        ax2.plot(x, cum_weights, 'o-', color='red', markersize=4, linewidth=1.5)
        ax2.axhline(y=0.8, linestyle='--', color='gray', alpha=0.7, label='80%')
        ax2.axhline(y=0.9, linestyle='--', color='gray', alpha=0.7, label='90%')
        ax2.axhline(y=0.95, linestyle='--', color='gray', alpha=0.7, label='95%')

        ax2.set_title('累积贡献率', fontsize=12)
        ax2.set_xlabel('分量索引')
        ax2.set_ylabel('累积贡献率')
        ax2.set_ylim(0, 1.02)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def get_reconstruction_quality(self, original_data):
        """
        计算重构质量指标

        参数:
            original_data: 原始数据

        返回:
            dict: 包含各种质量指标的字典
        """
        if not hasattr(self, 'reconstructed') or self.reconstructed is None:
            raise ValueError("请先调用reconstruct()方法生成重构分量")

        # 计算总重构误差
        total_reconstruction = np.sum(self.reconstructed, axis=2)
        mse = np.mean((original_data - total_reconstruction) ** 2)
        rmse = np.sqrt(mse)

        # 计算相关系数
        correlation = np.corrcoef(original_data.flatten(), total_reconstruction.flatten())[0, 1]

        # 计算解释方差比
        total_var = np.var(original_data)
        explained_var = np.var(total_reconstruction)
        explained_var_ratio = explained_var / total_var

        return {
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation,
            'explained_variance_ratio': explained_var_ratio,
            'total_components': self.n_components,
            'cumulative_weights': np.cumsum(self.component_weights)
        }

    def clear_cache(self):
        """清除缓存以释放内存"""
        self._w_corr_cache = {}
        self._weights_cache = None
