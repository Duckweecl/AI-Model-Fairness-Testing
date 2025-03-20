import argparse
import pandas as pd
import random
from tensorflow.keras.models import load_model
import numpy as np

model_path = 'DNN/model_processed_adult.h5'
dataset_path = 'dataset/processed_adult.csv'
class_label_column = 'Class-label'
budget = 100
samples_generate = 10
feature_generate = 10
def generate_choose_A(A_samples, model):

    A_with_preds = []

    for A in A_samples:

        preds = model.predict(A, verbose=0)

        diff = abs(preds - 0.5)
        A_with_preds.append((A, preds, diff))


    A_with_preds.sort(key=lambda x: x[2])


    best_A = A_with_preds[0][0]


    return best_A

def generate_random_B(A, feature: str, dataset: pd.DataFrame, class_label_column: str):


    A = pd.Series(A[0], index=dataset.drop(columns=[class_label_column]).columns)  # 删除标签列


    B_samples = []


    A_index = A.name


    previous_features = set()


    consecutive_retries = 0
    while len(B_samples) < feature_generate and consecutive_retries < 5:

        B = A.copy()


        available_indices = list(set(dataset.index) - {A_index})  # 排除掉 A 的索引
        idx_B = random.choice(available_indices)
        new_feature_value = dataset.loc[idx_B, feature]


        if new_feature_value not in previous_features:

            previous_features.add(new_feature_value)
            B[feature] = new_feature_value
            B = B.values.reshape(1, -1)
            B_samples.append(B)
            consecutive_retries = 0
        else:

            consecutive_retries += 1


    return B_samples

def generate_choose_B(A, B_samples, model):

    A_preds = model.predict(A, verbose=0)


    B_preds = [model.predict(B, verbose=0) for B in B_samples]


    if A_preds > 0.5:

        selected_B = B_samples[B_preds.index(min(B_preds))]
    else:

        selected_B = B_samples[B_preds.index(max(B_preds))]

    return selected_B

def generate_random_A(dataset: pd.DataFrame, label: str):

    dataset_without_label = dataset.drop(columns=[label])

    A_samples = []

    for _ in range(samples_generate):
        random_state = random.randint(1, len(dataset))
        A = dataset_without_label.sample(n=1, random_state=random_state).squeeze()

        A = A.values.reshape(1, -1)

        A_samples.append(A)

    return A_samples


parser = argparse.ArgumentParser(description='Evaluate feature pairs for prediction consistency.')
parser.add_argument('features', nargs='+', help='List of features to test')
args = parser.parse_args()






dataset = pd.read_csv(dataset_path)
model = load_model(model_path)
features = args.features


# 存储所有运行的结果
all_experiment_results = []



results = []

for feature in features:
        idi_values = []

        for _ in range(budget):  # 生成 50 组数据对

            result_df = generate_random_A(dataset, class_label_column)

            expected_columns = model.input_shape[1]

            A_random10 = generate_random_A(dataset, class_label_column)
            A = generate_choose_A(A_random10, model)

            B_random10 = generate_random_B(A, feature, dataset, class_label_column)
            B = generate_choose_B(A, B_random10, model)

            preds_A = model.predict(A, verbose=0)  # 假设模型的输入是 A 样本
            preds_B = model.predict(B, verbose=0)  # 假设模型的输入是 B 样本

            binary_preds_A = (preds_A > 0.5).astype(int)
            binary_preds_B = (preds_B > 0.5).astype(int)

            idi = 1 if not np.array_equal(binary_preds_A, binary_preds_B) else 0
            idi_values.append(idi)

        idi_ratio = np.mean(idi_values)
        results.append((feature, idi_ratio))

        print(f" '{feature}' IDI ratio: {idi_ratio:.4f}")

    all_experiment_results.append(results)
