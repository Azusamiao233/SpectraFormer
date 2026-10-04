import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import interpolate
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
from spectraformer.config import DATA_DIR, baseline_output_path

warnings.filterwarnings('ignore')


class TimeSeriesImputer:
    """时间序列插补方法集合"""

    @staticmethod
    def linear_interpolation(series):
        """线性插值"""
        if isinstance(series, pd.Series):
            return series.interpolate(method='linear')
        else:
            # 对numpy数组进行线性插值
            result = series.copy()
            mask = ~np.isnan(series)
            if np.sum(mask) < 2:  # 至少需要2个有效值
                return result

            indices = np.arange(len(series))
            valid_indices = indices[mask]
            valid_values = series[mask]

            # 使用线性插值
            f = interpolate.interp1d(valid_indices, valid_values,
                                     kind='linear', fill_value='extrapolate')
            missing_mask = np.isnan(series)
            result[missing_mask] = f(indices[missing_mask])

            return result

    @staticmethod
    def spline_interpolation(series, order=3):
        """样条插值"""
        if isinstance(series, pd.Series):
            return series.interpolate(method='spline', order=order)
        else:
            result = series.copy()
            mask = ~np.isnan(series)
            if np.sum(mask) < order + 1:  # 样条插值需要足够的点
                return TimeSeriesImputer.linear_interpolation(series)

            indices = np.arange(len(series))
            valid_indices = indices[mask]
            valid_values = series[mask]

            # 使用样条插值
            tck = interpolate.splrep(valid_indices, valid_values, k=min(order, len(valid_indices) - 1))
            missing_mask = np.isnan(series)
            result[missing_mask] = interpolate.splev(indices[missing_mask], tck)

            return result

    @staticmethod
    def polynomial_interpolation(series, order=3):
        """多项式插值"""
        result = series.copy()
        mask = ~np.isnan(series)
        if np.sum(mask) < order + 1:
            return TimeSeriesImputer.linear_interpolation(series)

        indices = np.arange(len(series))
        valid_indices = indices[mask]
        valid_values = series[mask]

        # 多项式拟合
        coeffs = np.polyfit(valid_indices, valid_values, min(order, len(valid_indices) - 1))
        poly_func = np.poly1d(coeffs)

        missing_mask = np.isnan(series)
        result[missing_mask] = poly_func(indices[missing_mask])

        return result

    @staticmethod
    def forward_fill(series):
        """前向填充（用前一个有效值填充）"""
        if isinstance(series, pd.Series):
            return series.fillna(method='ffill')
        else:
            result = series.copy()
            for i in range(1, len(result)):
                if np.isnan(result[i]) and not np.isnan(result[i - 1]):
                    result[i] = result[i - 1]
            return result

    @staticmethod
    def backward_fill(series):
        """后向填充（用后一个有效值填充）"""
        if isinstance(series, pd.Series):
            return series.fillna(method='bfill')
        else:
            result = series.copy()
            for i in range(len(result) - 2, -1, -1):
                if np.isnan(result[i]) and not np.isnan(result[i + 1]):
                    result[i] = result[i + 1]
            return result

    @staticmethod
    def moving_average(series, window=5):
        """移动平均插补"""
        result = series.copy()

        if isinstance(series, pd.Series):
            # 对每个缺失值，用前后窗口内的平均值填充
            for i in range(len(series)):
                if pd.isna(series.iloc[i]):
                    start = max(0, i - window // 2)
                    end = min(len(series), i + window // 2 + 1)
                    window_data = series.iloc[start:end]
                    if not window_data.isna().all():
                        result.iloc[i] = window_data.mean()
        else:
            for i in range(len(series)):
                if np.isnan(series[i]):
                    start = max(0, i - window // 2)
                    end = min(len(series), i + window // 2 + 1)
                    window_data = series[start:end]
                    valid_data = window_data[~np.isnan(window_data)]
                    if len(valid_data) > 0:
                        result[i] = np.mean(valid_data)

        return result

    @staticmethod
    def seasonal_decomposition_impute(series, period=12):
        """基于季节性分解的插补"""
        try:
            if isinstance(series, np.ndarray):
                series = pd.Series(series)

            # 先用线性插值填充一些缺失值以进行分解
            temp_filled = series.interpolate(method='linear')

            if len(temp_filled.dropna()) < 2 * period:
                return TimeSeriesImputer.linear_interpolation(series)

            # 季节性分解
            decomposition = seasonal_decompose(temp_filled, model='additive', period=period)

            # 重构序列
            reconstructed = decomposition.trend.fillna(method='ffill').fillna(method='bfill') + \
                            decomposition.seasonal + \
                            decomposition.resid.fillna(0)

            # 只在原始缺失位置使用重构值
            result = series.copy()
            missing_mask = series.isna()
            result[missing_mask] = reconstructed[missing_mask]

            return result.values if isinstance(series, pd.Series) else result

        except:
            return TimeSeriesImputer.linear_interpolation(series)

    @staticmethod
    def exponential_smoothing_impute(series, alpha=0.3):
        """指数平滑插补"""
        result = series.copy()

        if isinstance(series, np.ndarray):
            series_pd = pd.Series(series)
        else:
            series_pd = series

        # 找到第一个非缺失值作为起始点
        first_valid = series_pd.first_valid_index()
        if first_valid is None:
            return result

        # 指数平滑
        smoothed = [series_pd.loc[first_valid]]
        for i in range(first_valid + 1, len(series_pd)):
            if pd.isna(series_pd.iloc[i]):
                # 缺失值用平滑值估计
                smoothed_val = alpha * smoothed[-1] + (1 - alpha) * smoothed[-1]
            else:
                # 有效值更新平滑值
                smoothed_val = alpha * series_pd.iloc[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_val)

        # 填充缺失值
        for i, val in enumerate(smoothed):
            if pd.isna(series_pd.iloc[first_valid + i]):
                if isinstance(result, pd.Series):
                    result.iloc[first_valid + i] = val
                else:
                    result[first_valid + i] = val

        return result

    @staticmethod
    def kalman_filter_impute(series):
        """简单卡尔曼滤波插补"""
        result = series.copy()

        # 简化的卡尔曼滤波实现
        # 状态方程: x(k+1) = x(k) + w(k)
        # 观测方程: y(k) = x(k) + v(k)

        Q = 0.1  # 过程噪声方差
        R = 1.0  # 观测噪声方差

        # 初始化
        x_est = 0.0  # 状态估计
        P_est = 1.0  # 误差协方差估计

        valid_indices = []
        if isinstance(series, pd.Series):
            valid_indices = series.dropna().index.tolist()
            if len(valid_indices) > 0:
                x_est = series[valid_indices[0]]
        else:
            valid_indices = np.where(~np.isnan(series))[0]
            if len(valid_indices) > 0:
                x_est = series[valid_indices[0]]

        for i in range(len(series)):
            # 预测步
            x_pred = x_est
            P_pred = P_est + Q

            if isinstance(series, pd.Series):
                is_missing = pd.isna(series.iloc[i])
                obs_value = series.iloc[i]
            else:
                is_missing = np.isnan(series[i])
                obs_value = series[i]

            if not is_missing:
                # 更新步（有观测值）
                K = P_pred / (P_pred + R)  # 卡尔曼增益
                x_est = x_pred + K * (obs_value - x_pred)
                P_est = (1 - K) * P_pred
            else:
                # 无观测值，使用预测值
                x_est = x_pred
                P_est = P_pred
                if isinstance(result, pd.Series):
                    result.iloc[i] = x_est
                else:
                    result[i] = x_est

        return result


def compare_timeseries_imputation_methods(data_with_missing, complete_data, columns_to_impute=None):
    """
    比较多种时间序列插补方法的效果

    Parameters:
    data_with_missing: 带有缺失值的时间序列数据
    complete_data: 完整的时间序列数据
    columns_to_impute: 需要填补的列索引或列名列表
    """

    # 数据预处理
    original_columns = None
    if isinstance(data_with_missing, pd.DataFrame):
        original_columns = data_with_missing.columns.tolist()

        if columns_to_impute is not None and isinstance(columns_to_impute[0], str):
            columns_to_impute = [data_with_missing.columns.get_loc(col) for col in columns_to_impute]

        data_with_missing_array = data_with_missing.values
        complete_data_array = complete_data.values
    else:
        data_with_missing_array = data_with_missing
        complete_data_array = complete_data

    # 自动检测缺失列
    if columns_to_impute is None:
        missing_mask_full = np.isnan(data_with_missing_array)
        columns_with_missing = np.where(np.any(missing_mask_full, axis=0))[0]
        columns_to_impute = columns_with_missing.tolist()
        print(f"自动检测到有缺失值的列: {columns_to_impute}")

    print("开始进行时间序列插补方法比较...")
    print(f"数据形状: {data_with_missing_array.shape}")
    print(f"需要填补的列: {columns_to_impute}")

    # 统计缺失值信息
    missing_mask_selected = np.isnan(data_with_missing_array[:, columns_to_impute])
    total_missing = np.sum(missing_mask_selected)

    print(f"指定列中缺失值数量: {total_missing}")
    for i, col_idx in enumerate(columns_to_impute):
        col_missing = np.sum(missing_mask_selected[:, i])
        col_name = f"列{col_idx}" if original_columns is None else original_columns[col_idx]
        print(f"  {col_name}: {col_missing}个缺失值")

    print("-" * 60)

    # 准备评估数据
    true_missing_values = complete_data_array[:, columns_to_impute].copy()
    true_missing_values[~missing_mask_selected] = np.nan

    # 定义插补方法
    methods = {
        'Linear_Interpolation': TimeSeriesImputer.linear_interpolation,
        'Spline_Interpolation': lambda x: TimeSeriesImputer.spline_interpolation(x, order=3),
        'Polynomial_Interpolation': lambda x: TimeSeriesImputer.polynomial_interpolation(x, order=3),
        'Forward_Fill': TimeSeriesImputer.forward_fill,
        'Backward_Fill': TimeSeriesImputer.backward_fill,
        'Moving_Average': lambda x: TimeSeriesImputer.moving_average(x, window=5),
        'Seasonal_Decomposition': lambda x: TimeSeriesImputer.seasonal_decomposition_impute(x, period=12),
        'Exponential_Smoothing': lambda x: TimeSeriesImputer.exponential_smoothing_impute(x, alpha=0.3),
        'Kalman_Filter': TimeSeriesImputer.kalman_filter_impute
    }

    results = []
    filled_data_dict = {}

    # 对每种方法进行测试
    for method_name, method_func in methods.items():
        print(f"正在进行 {method_name} 插补...")

        try:
            # 复制原始数据
            data_filled = data_with_missing_array.copy()

            # 对每个指定列进行插补
            for col_idx in columns_to_impute:
                col_data = data_with_missing_array[:, col_idx].copy()
                filled_col_data = method_func(col_data)
                data_filled[:, col_idx] = filled_col_data

            # 评估结果
            result = evaluate_imputation(true_missing_values, data_filled[:, columns_to_impute], method_name)
            results.append(result)

            # 保存填补后的数据
            if original_columns is not None:
                filled_data_dict[method_name] = pd.DataFrame(data_filled, columns=original_columns)
            else:
                filled_data_dict[method_name] = data_filled

            print(f"{method_name} 完成 - R²: {result['R²']:.4f}, RMSE: {result['RMSE']:.4f}, MAE: {result['MAE']:.4f}")

        except Exception as e:
            print(f"{method_name} 插补失败: {str(e)}")
            continue

    print("-" * 60)

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # 按R²排序显示结果
        results_df_sorted = results_df.sort_values('R²', ascending=False)
        print("时间序列插补方法比较结果 (按R²排序):")
        print(results_df_sorted.to_string(index=False, float_format='%.4f'))

        # 找出最佳方法
        best_r2 = results_df.loc[results_df['R²'].idxmax(), 'Method']
        best_rmse = results_df.loc[results_df['RMSE'].idxmin(), 'Method']
        best_mae = results_df.loc[results_df['MAE'].idxmin(), 'Method']

        print(f"\n最佳方法:")
        print(f"R² (越大越好): {best_r2}")
        print(f"RMSE (越小越好): {best_rmse}")
        print(f"MAE (越小越好): {best_mae}")

    return {
        'results_df': results_df,
        'filled_data': filled_data_dict,
        'columns_imputed': columns_to_impute
    }


def evaluate_imputation(true_values, imputed_values, method_name):
    """评估插补效果"""
    mask = ~np.isnan(true_values)
    if np.sum(mask) == 0:
        return {
            'Method': method_name,
            'R²': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan
        }

    true_flat = true_values[mask]
    imputed_flat = imputed_values[mask]

    # 计算评估指标
    r2 = r2_score(true_flat, imputed_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, imputed_flat))
    mae = mean_absolute_error(true_flat, imputed_flat)

    return {
        'Method': method_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae
    }


# 数据读取函数
def load_data(missing_data_path, complete_data_path):
    """读取数据文件"""

    def read_file(file_path):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            return pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

    try:
        missing_data = read_file(missing_data_path)
        complete_data = read_file(complete_data_path)

        print(f"成功读取缺失数据: {missing_data.shape}")
        print(f"成功读取完整数据: {complete_data.shape}")

        if missing_data.shape != complete_data.shape:
            raise ValueError(f"数据形状不匹配: 缺失数据{missing_data.shape} vs 完整数据{complete_data.shape}")

        return missing_data, complete_data

    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None, None


# 使用示例
if __name__ == "__main__":
    # 请修改为你的数据文件路径
    missing_data_path = str(DATA_DIR / "train_wadi_continuous_missing_60.csv")
    complete_data_path = str(DATA_DIR / "train_wadi.csv")

    # 指定需要填补的列
    #columns_to_impute = ['column1', 'column2']  # 修改为你的列名
    # columns_to_impute = [0, 1]  # 或使用列索引
    columns_to_impute = None    # 或自动检测

    # 读取数据
    data_with_missing, complete_data = load_data(missing_data_path, complete_data_path)

    if data_with_missing is not None and complete_data is not None:
        # 运行时间序列插补方法比较
        results = compare_timeseries_imputation_methods(data_with_missing, complete_data, columns_to_impute)

        # 保存结果
        if len(results['results_df']) > 0:
            results['results_df'].to_csv(
                baseline_output_path("timeseries_imputation_results.csv"), index=False
            )
            print(f"\n结果已保存到 'timeseries_imputation_results.csv'")

            # 保存最佳方法的填补数据
            best_method = results['results_df'].loc[results['results_df']['R²'].idxmax(), 'Method']
            best_data = results['filled_data'][best_method]

            if isinstance(best_data, pd.DataFrame):
                best_data.to_csv(
                    baseline_output_path(f'data_filled_{best_method.lower()}.csv'),
                    index=False,
                )
            else:
                pd.DataFrame(best_data).to_csv(
                    baseline_output_path(f'data_filled_{best_method.lower()}.csv'),
                    index=False,
                )

            print(f"最佳方法 ({best_method}) 的填补数据已保存")

    else:
        print("数据读取失败，请检查文件路径和格式")
