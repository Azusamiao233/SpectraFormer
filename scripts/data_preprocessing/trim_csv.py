import pandas as pd
import os
from spectraformer.config import DATA_DIR


def remove_last_n_lines(csv_file, num_lines):
    """
    使用pandas从CSV文件末尾删除指定数量的行

    参数:
        csv_file (str): CSV文件路径
        num_lines (int): 要删除的行数
    """
    if num_lines <= 0:
        print("请指定大于0的行数")
        return

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件 '{csv_file}' 不存在")
        return

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 检查文件是否有足够的行
        if len(df) <= num_lines:
            print(f"文件只有{len(df)}行，无法删除{num_lines}行")
            return

        # 保留除最后num_lines行之外的所有行
        df_trimmed = df.iloc[:-num_lines]

        # 写回CSV文件，不保留索引
        df_trimmed.to_csv(csv_file, index=False)

        print(f"成功从文件末尾删除了{num_lines}行")

    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")


if __name__ == "__main__":
    # 示例用法
    csv_path = str(DATA_DIR / "train_wadi.csv")
    lines_to_remove = 9000  # 要从末尾删除的行数

    remove_last_n_lines(csv_path, lines_to_remove)
