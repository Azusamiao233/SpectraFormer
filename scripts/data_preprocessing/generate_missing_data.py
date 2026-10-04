import pandas as pd
import numpy as np
import random
from typing import List, Optional
from mssa_transformer.config import DATA_DIR, MASKED_DATA_DIR


def add_continuous_missing(input_file: str,
                           output_file: str,
                           columns: Optional[List[str]] = None,
                           missing_ratio: float = 0.1,
                           min_length: int = 5,
                           max_length: int = 20) -> None:
    """
    对CSV文件添加连续缺失值并保存

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    columns: 需要添加缺失值的列名列表，如果为None则处理所有列（除第一列）
    missing_ratio: 缺失值比例 (0-1之间)
    min_length: 连续缺失的最小长度
    max_length: 连续缺失的最大长度
    """
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据形状: {df.shape}")

    # 如果没有指定列，使用除第一列外的所有列
    if columns is None:
        columns = df.columns.tolist()[1:]  # 跳过第一列（序列号）
        print(f"未指定列，将处理除序列号外的所有列")

    print(f"\n将对以下列添加连续缺失: {columns}")
    print(f"缺失比例: {missing_ratio * 100:.1f}%")
    print(f"连续缺失长度范围: {min_length}-{max_length}")

    # 处理每一列
    for col in columns:
        if col not in df.columns:
            print(f"警告: 列 '{col}' 不存在，跳过处理")
            continue

        total_values = len(df)
        missing_count = int(total_values * missing_ratio)

        # 计算需要多少段连续缺失
        avg_length = (min_length + max_length) / 2
        num_segments = max(1, int(missing_count / avg_length))

        # 记录已经设置为缺失的位置
        missing_positions = set()

        for _ in range(num_segments):
            if len(missing_positions) >= missing_count:
                break

            # 随机选择连续缺失的长度
            length = random.randint(min_length, min(max_length, missing_count - len(missing_positions)))

            # 随机选择起始位置
            max_start = total_values - length
            if max_start <= 0:
                continue

            # 尝试找到一个没有重叠的位置
            attempts = 0
            while attempts < 100:
                start = random.randint(0, max_start)
                end = start + length

                # 检查是否与已有的缺失位置重叠
                if not any(pos in missing_positions for pos in range(start, end)):
                    # 添加连续缺失
                    for pos in range(start, end):
                        missing_positions.add(pos)
                        df.loc[pos, col] = np.nan
                    break

                attempts += 1

        actual_missing = len(missing_positions)
        print(f"  列 '{col}': 添加了 {actual_missing} 个连续缺失值 ({actual_missing / total_values * 100:.2f}%)")

    # 保存结果
    print(f"\n保存处理后的数据到: {output_file}")
    df.to_csv(output_file, index=False)

    # 统计缺失值情况
    print("\n缺失值统计汇总:")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    if len(missing_stats) > 0:
        for col, count in missing_stats.items():
            print(f"  {col}: {count} 个缺失值 ({count / len(df) * 100:.2f}%)")
    else:
        print("  没有缺失值")

    print(f"\n处理完成！")


def add_random_missing(input_file: str,
                       output_file: str,
                       columns: Optional[List[str]] = None,
                       missing_ratio: float = 0.1) -> None:
    """
    对CSV文件添加随机缺失值并保存

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    columns: 需要添加缺失值的列名列表，如果为None则处理所有列（除第一列）
    missing_ratio: 缺失值比例 (0-1之间)
    """
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据形状: {df.shape}")

    # 如果没有指定列，使用除第一列外的所有列
    if columns is None:
        columns = df.columns.tolist()[1:]  # 跳过第一列（序列号）
        print(f"未指定列，将处理除序列号外的所有列")

    print(f"\n将对以下列添加随机缺失: {columns}")
    print(f"缺失比例: {missing_ratio * 100:.1f}%")

    # 处理每一列
    for col in columns:
        if col not in df.columns:
            print(f"警告: 列 '{col}' 不存在，跳过处理")
            continue

        total_values = len(df)
        missing_count = int(total_values * missing_ratio)

        # 随机选择要设置为缺失的位置
        missing_indices = np.random.choice(total_values, missing_count, replace=False)

        # 设置缺失值
        df.loc[missing_indices, col] = np.nan

        print(f"  列 '{col}': 添加了 {missing_count} 个随机缺失值 ({missing_ratio * 100:.2f}%)")

    # 保存结果
    print(f"\n保存处理后的数据到: {output_file}")
    df.to_csv(output_file, index=False)

    # 统计缺失值情况
    print("\n缺失值统计汇总:")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    if len(missing_stats) > 0:
        for col, count in missing_stats.items():
            print(f"  {col}: {count} 个缺失值 ({count / len(df) * 100:.2f}%)")
    else:
        print("  没有缺失值")

    print(f"\n处理完成！")


def get_column_info(csv_file: str) -> None:
    """
    辅助函数：显示CSV文件的列信息

    参数:
    csv_file: CSV文件路径
    """
    df = pd.read_csv(csv_file)
    print(f"文件: {csv_file}")
    print(f"数据形状: {df.shape}")
    print(f"\n列信息:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        print(f"  [{i}] {col}: {dtype}, 非空值: {non_null}, 缺失值: {null_count}")


# 使用示例
if __name__ == "__main__":
    # 查看文件信息
    print("=== 查看原始文件信息 ===")
    get_column_info(str(DATA_DIR / "swat_test.csv"))

    print("\n" + "=" * 50 + "\n")
    """
    # 示例1: 对特定列添加连续缺失
    print("=== 示例1: 连续缺失处理 ===")
    add_continuous_missing(
        input_file=str(DATA_DIR / "swat_test.csv"),
        output_file=str(MASKED_DATA_DIR / "swat_test_missing.csv"),
        columns=["FIT101","LIT101"],  # 指定要处理的列
        missing_ratio=0.4,  # 15%的缺失
        min_length=17000,  # 连续缺失最少值
        max_length=18000  # 连续缺失最多值
    )
    """
    print("\n" + "=" * 50 + "\n")

    # 示例2: 对特定列添加随机缺失
    print("=== 示例2: 随机缺失处理 ===")
    add_random_missing(
        input_file=str(MASKED_DATA_DIR / "swat_test_missing.csv"),
        output_file=str(MASKED_DATA_DIR / "swat_test_random_missing.csv"),
        columns=["FIT101","LIT101"],  # 指定要处理的列
        missing_ratio=0.10  # 10%的缺失
    )

    print("\n" + "=" * 50 + "\n")



    # 示例3: 对所有列（除序列号）添加连续缺失
    #print("=== 示例3: 对所有列添加连续缺失 ===")
    # add_continuous_missing(
    #     input_file="test.csv",
    #     output_file="train_all_continuous.csv",
    #     columns=None,  # None表示处理所有列（除第一列）
    #     missing_ratio=0.05,
    #     min_length=20,
    #     max_length=100
    # )
