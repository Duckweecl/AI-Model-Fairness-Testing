import os
import pandas as pd

# 数据集路径
file_path = 'processed_communities_crime.csv'

# 获取当前工作目录
current_directory = os.getcwd()

# 检查文件是否存在
if os.path.exists(file_path):
    # 输出数据集路径
    print(f"Dataset file path: {os.path.abspath(file_path)}")

    # 加载数据集
    df = pd.read_csv(file_path)

    # 输出所有特征名称，排除第一个特征
    features = df.columns[1:]
    print("\nFeatures Categories (excluding the first feature):")
    print(features)

    # 统计最后两个二进制特征的1和0的比例
    last_two_features = df.columns[-2:]  # 获取最后两个特征
    for feature in last_two_features:
        feature_counts = df[feature].value_counts(normalize=True)  # 获取每个值的比例
        print(f"\nProportion of values in '{feature}':")
        print(feature_counts)
else:
    # 如果文件没有找到，输出寻找路径的提示
    print(f"File '{file_path}' not found.")
    print(f"Please check the file in the current directory: {current_directory}")
    print("If the file is located elsewhere, provide the correct path.")

