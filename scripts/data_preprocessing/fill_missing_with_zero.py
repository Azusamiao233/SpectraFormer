import pandas as pd
import os
from mssa_transformer.config import DATA_DIR, OUTPUTS_DIR

# 文件路径配置
INPUT_FILE = str(DATA_DIR / "train_continuous_missing_40.csv")
OUTPUT_FILE = str(OUTPUTS_DIR / "preprocessing" / "data_cleaned.csv")


def replace_missing_values_with_zero():
    """
    将CSV文件中的所有缺失值替换为0
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(INPUT_FILE):
            print(f"错误：找不到文件 {INPUT_FILE}")
            return

        # 读取CSV文件
        print(f"正在读取文件: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)

        # 显示原始数据信息
        print(f"原始数据形状: {df.shape}")
        print(f"缺失值总数: {df.isnull().sum().sum()}")

        if df.isnull().sum().sum() > 0:
            print("各列缺失值统计:")
            missing_counts = df.isnull().sum()
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count}")

        # 将所有缺失值替换为0
        df_filled = df.fillna(0)

        # 保存处理后的文件
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_filled.to_csv(OUTPUT_FILE, index=False)
        print(f"处理完成！文件已保存到: {OUTPUT_FILE}")
        print(f"处理后缺失值总数: {df_filled.isnull().sum().sum()}")

    except pd.errors.EmptyDataError:
        print(f"错误：文件 {INPUT_FILE} 为空")
    except pd.errors.ParserError:
        print(f"错误：无法解析文件 {INPUT_FILE}，请检查文件格式")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    replace_missing_values_with_zero()
