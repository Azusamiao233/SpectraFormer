import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from mssa_transformer.config import DATA_DIR

import numpy as np  # 确保导入 numpy


# from scipy.linalg import svd # 你可以注释掉或移除 scipy.linalg.svd 的特定导入
# 如果你之前是 from scipy.linalg import hankel, svd
# 就改成 from scipy.linalg import hankel
# 然后 SVD 使用 np.linalg.svd

def ssa_decompose(time_series, window_length):
    """
        Perform Singular Spectrum Analysis (SSA) on a time series.

        Parameters:
            time_series (numpy array): The input time series.
            window_length (int): The embedding window length.

        Returns:
            tuple: Contains the trajectory matrix, singular values, and reconstructed components.
                   Returns (None, None, None) if window_length is too large.
    """
    N = len(time_series)
    if window_length <= 0 or window_length > N:
        print(f"Error: Window length ({window_length}) must be between 1 and series length ({N}).")
        return None, None, None

    K = N - window_length + 1
    # Step 1: Embedding (construct the trajectory matrix)
    # The Hankel matrix in scipy.linalg.hankel(c, r) uses c as the first column
    # and r as the last row. For SSA, the trajectory matrix is typically constructed
    # such that each row is a lagged version of the time series.
    # A more direct way to form the trajectory matrix X for SSA:
    # X_ij = y_{i+j-2} for i=1..L, j=1..K
    # Or, equivalently, each column is a segment of the time series.

    # Using a loop for clarity in constructing the trajectory matrix
    # trajectory_matrix = np.column_stack([time_series[i:i+window_length] for i in range(K)])
    # However, the Hankel matrix construction from the original code is also a valid way to form it,
    # though one needs to be careful about its orientation.
    # Let's stick to a common SSA trajectory matrix formulation:
    # Rows are lagged vectors.
    trajectory_matrix = np.zeros((window_length, K))
    # Or, columns are lagged vectors (more common for SVD in some texts, results in U, S, V.T where V's columns are eigenvectors)
    # trajectory_matrix = np.array([time_series[i:i+window_length] for i in range(K)]).T # This would be L x K

    # Let's use the definition where columns are lagged versions of the window:
    # X = [Y_1, Y_2, ..., Y_K] where Y_i = (y_i, y_{i+1}, ..., y_{i+L-1})^T
    # This results in an L x K matrix.
    for i in range(K):
        trajectory_matrix[:, i] = time_series[i:i + window_length]

    # Step 2: Singular Value Decomposition (SVD)
    try:
        # 主要修改点：
        # 1. 使用 np.linalg.svd
        # 2. 设置 full_matrices=False
        U, sigma, Vt = np.linalg.svd(trajectory_matrix, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"SVD computation failed: {e}")
        return None, None, None
    except MemoryError as e:
        print(f"MemoryError during SVD: {e}. The trajectory matrix (L={window_length}, K={K}) might be too large.")
        return None, None, None


    # Number of components is min(window_length, K), which is len(sigma)
    num_s_values = len(sigma)  # This will be min(L,K)
    reconstructed_components_list = []

    # U is L x num_s_values
    # Vt is num_s_values x K

    for r_idx in range(num_s_values):  # Iterate through each elementary matrix component
        Ur_col = U[:, r_idx].reshape(window_length, 1)
        Vr_row = Vt[r_idx, :].reshape(1, K)

        elementary_matrix = sigma[r_idx] * np.dot(Ur_col, Vr_row)  # This is L x K

        reconstructed_series_r = np.zeros(N)

        # Diagonal averaging (anti-diagonal averaging)
        for k_diag in range(N):  # Iterate over anti-diagonals of the reconstructed elementary matrix
            s = 0
            count = 0
            # Determine the range of (i, j) for the k-th anti-diagonal
            # i from 0 to L-1 (row index in elementary_matrix)
            # j from 0 to K-1 (col index in elementary_matrix)
            # k_diag = i + j (for the series index)
            # For elementary_matrix[row_idx, col_idx], series_idx = row_idx + col_idx

            # This loop structure for diagonal averaging might need careful review
            # A common way:
            # For each k from 0 to N-1 (index in the reconstructed series)
            #   sum_val = 0
            #   num_terms = 0
            #   For l from 0 to L-1 (row in trajectory/elementary matrix)
            #     j = k - l (column in trajectory/elementary matrix)
            #     If 0 <= j < K:
            #       sum_val += elementary_matrix[l, j]
            #       num_terms += 1
            #   If num_terms > 0: reconstructed_series_r[k] = sum_val / num_terms

            # The existing diagonal averaging logic from your previous code:
            for i_row in range(max(0, k_diag - K + 1), min(k_diag + 1, window_length)):
                j_col = k_diag - i_row
                if 0 <= j_col < K:  # Ensure j_col is a valid column index for elementary_matrix
                    s += elementary_matrix[i_row, j_col]
                    count += 1
            if count > 0:
                reconstructed_series_r[k_diag] = s / count
            # else: reconstructed_series_r[k_diag] = 0 # Should ideally not happen if k_diag covers 0..N-1
        reconstructed_components_list.append(reconstructed_series_r)

    return trajectory_matrix, sigma, reconstructed_components_list


def plot_ssa_results(time_series, components, singular_values, num_components_to_plot=3, feature_name="Feature"):
    """
    Plot the original time series, singular values (scree plot), and the first few reconstructed components.

    Parameters:
        time_series (numpy array): The original time series.
        components (list): The reconstructed components from SSA.
        singular_values (numpy array): Singular values from SVD.
        num_components_to_plot (int): Number of components to plot.
        feature_name (str): Name of the feature being analyzed.
    """
    if components is None or singular_values is None:
        print("Skipping plotting due to SSA decomposition failure.")
        return

    plt.figure(figsize=(15, 6 + 3 * num_components_to_plot))

    # Plot original time series
    plt.subplot(num_components_to_plot + 2, 1, 1)
    plt.plot(time_series, label=f'Original Time Series ({feature_name})', color='black')
    plt.title(f'Original Time Series: {feature_name}')
    plt.legend()
    plt.grid(True)

    # Plot singular values (Scree Plot)
    plt.subplot(num_components_to_plot + 2, 1, 2)
    plt.plot(range(1, len(singular_values) + 1), singular_values, 'o-', label='Singular Values')
    plt.title('Scree Plot (Singular Values)')
    plt.xlabel('Component Number')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.grid(True)

    # Plot reconstructed components
    actual_components_to_plot = min(num_components_to_plot, len(components))
    for i in range(actual_components_to_plot):
        plt.subplot(num_components_to_plot + 2, 1, i + 3)
        plt.plot(components[i], label=f'Reconstructed Component {i + 1}')
        plt.title(f'Reconstructed Component {i + 1}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # --- User Configuration ---
    input_file = str(DATA_DIR / "swat_test.csv")
    # 例如: input_file = 'series_data.csv'

    # 让用户选择要分析的特征列名或列号（从0开始，不包括第一列的序号）
    # 假设你的CSV有表头
    feature_to_analyze = 'FIT101'  # <--- !!! 请修改为你想要分析的特征列名 !!!
    # 或者，如果你的CSV没有表头，或者你想用列号（相对于特征列，即原始第二列是特征0）
    # feature_to_analyze = 0 # 分析第一个特征 (原始数据的第二列)

    window_length = 50  # <--- 你可以根据你的数据调整窗口长度
    num_components_to_plot = 3  # 要绘制的重构分量数量
    # --- End User Configuration ---

    try:
        # 1. 读取数据
        # 假设第一列是序号，我们将其作为索引
        df = pd.read_csv(input_file, index_col=0)
        print(f"成功加载数据，共 {df.shape[0]}行, {df.shape[1]}个特征 (不包括序号列).")
        print("特征列名:", df.columns.tolist())

        # 2. 选择要分析的特征
        if isinstance(feature_to_analyze, str):
            if feature_to_analyze not in df.columns:
                print(f"错误: 特征列 '{feature_to_analyze}' 在文件中未找到。")
                print(f"可用的特征列: {df.columns.tolist()}")
                exit()
            time_series_selected = df[feature_to_analyze].values
            selected_feature_name = feature_to_analyze
        elif isinstance(feature_to_analyze, int):
            if feature_to_analyze < 0 or feature_to_analyze >= df.shape[1]:
                print(f"错误: 特征列号 {feature_to_analyze} 超出范围 (0 到 {df.shape[1] - 1}).")
                exit()
            time_series_selected = df.iloc[:, feature_to_analyze].values
            selected_feature_name = df.columns[feature_to_analyze]
        else:
            print("错误: 'feature_to_analyze' 必须是列名字符串或列索引整数。")
            exit()

        print(f"\n选择分析的特征: '{selected_feature_name}'")
        print(f"序列长度: {len(time_series_selected)}")
        print(f"窗口长度 (L): {window_length}")

        # 3. 执行 SSA 分解
        trajectory_matrix, singular_values, reconstructed_components = ssa_decompose(time_series_selected,
                                                                                     window_length)

        if trajectory_matrix is not None:
            print(f"\n轨迹矩阵形状: {trajectory_matrix.shape}")
            print(f"奇异值数量: {len(singular_values)}")
            # print("奇异值:", singular_values) # 可以取消注释以查看奇异值
            print(f"重构分量数量: {len(reconstructed_components)}")

            # 4. 绘制结果
            plot_ssa_results(time_series_selected, reconstructed_components, singular_values,
                             num_components_to_plot=num_components_to_plot,
                             feature_name=selected_feature_name)
        else:
            print("SSA分解失败，无法进行绘图。")

    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。请确保文件路径正确。")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{input_file}' 为空。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
