"""Model development utilities.

The primary model is intentionally simple: scaled logistic regression. A
shallow decision tree is used as a challenger to demonstrate benchmark-model
comparison rather than model-complexity chasing.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from .data import FEATURES


def build_logistic_model():
    prep = ColumnTransformer([
        ("scale", StandardScaler(), FEATURES),
    ], remainder="drop")
    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_challenger_model():
    return DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=150,
        class_weight=None,
        random_state=42,
    )
