from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import numpy as np


def log_transform_amount(X):
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X

def get_estimator():
    numeric_features = ["Amount"]
    pca_features = [f"V{i}" for i in range(1, 29)]

    preprocessor = ColumnTransformer([
        ("log_transform", FunctionTransformer(log_transform_amount, validate=False), numeric_features),
        ("scaler", StandardScaler(), numeric_features + pca_features)
    ], remainder="drop")

    pipe = make_pipeline(
        preprocessor,
        LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    )

    return pipe
