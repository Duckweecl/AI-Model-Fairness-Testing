import argparse
import pandas as pd
import random
import numpy as np
from tensorflow.keras.models import load_model
import os


def generate_random_pairs(dataset: pd.DataFrame, feature: str):
    """
    随机从 dataset 中挑选一条数据，复制为 A 和 B，
    其中 B 的指定 feature 从另一条数据中替换。
    确保下一次迭代不会选择到相同的数据。
    """
    used_indices = set()

    while len(used_indices) < len(dataset):
        available_indices = list(set(dataset.index) - used_indices)
        idx_A = random.choice(available_indices)
        A = dataset.loc[idx_A].copy()

        B = A.copy()

        available_indices.remove(idx_A)
        if not available_indices:
            break

        idx_B = random.choice(available_indices)
        B[feature] = dataset.loc[idx_B, feature]

        used_indices.add(idx_A)
        used_indices.add(idx_B)

        return pd.DataFrame([A, B], columns=dataset.columns)


parser = argparse.ArgumentParser(description='Evaluate feature pairs for prediction consistency.')
parser.add_argument('features', nargs='+', help='List of features to test')
args = parser.parse_args()

model_path = 'DNN/model_processed_adult.h5'
model = load_model(model_path)

dataset_path = 'dataset/processed_adult.csv'
dataset = pd.read_csv(dataset_path)

features = args.features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 存储所有运行的结果
all_experiment_results = []

# **外部循环**: 运行 10 次完整实验
for run in range(10):
    results = []

    for feature in features:
        idi_values = []

        for _ in range(200):  # 生成 50 组数据对
            result_df = generate_random_pairs(dataset, feature)

            expected_columns = model.input_shape[1]
            class_label_column = 'Class-label'

            result_df = result_df.drop(columns=[class_label_column])

            X = result_df.values
            preds = model.predict(X, verbose=0)

            binary_preds = (preds > 0.5).astype(int)

            # 计算 IDI 值
            idi = 1 if not np.array_equal(binary_preds[0], binary_preds[1]) else 0
            idi_values.append(idi)

        # 计算 IDI ratio
        idi_ratio = np.mean(idi_values)
        results.append((feature, idi_ratio))


        print(f" '{feature}' IDI ratio: {idi_ratio:.4f}")

    all_experiment_results.append(results)


