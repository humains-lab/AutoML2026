from pipegenie.classification import PipegenieClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.io import arff
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str, help="Path to dataset")
parser.add_argument("seed", type=int, help="Random seed for the experiment")
args = parser.parse_args()

def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )

    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = pd.get_dummies(X)

    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)

    elif set(y.unique()) == {-1, 1}:
        y = y.map({-1: 0, 1: 1})

    df = pd.concat([X, y], axis=1)

    return df

dataset_path = args.filepath
dataset_name = os.path.basename(dataset_path)

seed = args.seed
np.random.seed = seed

print(f"-- Dataset {dataset_name} running (seed {seed}) --")

train_path = os.path.join(dataset_path, "train.arff")
test_path = os.path.join(dataset_path, "test.arff")

train_df = load_arff(train_path)
test_df = load_arff(test_path)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

model = PipegenieClassifier(
    generations=20,
    pop_size=40,
    elite_size=5,
    n_jobs=5,
    seed=seed,
    nderiv=20,
    outdir=f"results/raw_results/{dataset_name}/{seed}",
)

model.fit(X_train, y_train)
print(f"-- Dataset {dataset_name} training finished (seed {seed}) --")
model.ensemble_explainer(X_train, y_train)
print(f"-- Dataset {dataset_name} explainer finished (seed {seed}) --")