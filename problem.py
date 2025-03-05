import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Credit Card Fraud Detection - a binary classification problem'
_target_column_name = 'Class'
_ignore_column_names = []
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

# The roc_auc score is the default metric for the challenge
score_types = [
    rw.score_types.ROCAUC(name='roc_auc', precision=4),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
    return cv.split(X, y)


def csv_array_to_float(csv_array_string):
    return list(map(float, csv_array_string[1:-1].split(',')))


def _read_data(path, df_filename):
    df = pd.read_csv(os.path.join(path, 'data', df_filename))
    y_array = df[_target_column_name].values
    X_dict = df.drop(_target_column_name, axis=1)
    return X_dict, y_array


def get_train_data(path='.'):
    df_filename = 'train.csv'
    return _read_data(path, df_filename)


def get_test_data(path='.'):
    df_filename = 'test.csv'
    return _read_data(path, df_filename)
