from typing import NamedTuple
from kfp.dsl import component


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn"]
)
def load_data() -> NamedTuple("Outputs", [("X", list), ("y", list)]):

    from sklearn.datasets import load_iris

    data = load_iris()

    X = data.data.tolist()
    y = data.target.tolist()

    from collections import namedtuple

    outputs = namedtuple("Outputs", ["X", "y"])

    print("Dataset loaded successfully")

    return outputs(X, y)


@component(
    base_image="python:3.11",
    packages_to_install=["scikit-learn"]
)
def train_model(X: list, y: list):

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=200)

    model.fit(X, y)

    print("Model trained successfully")