"""
PyCaret AutoML module for MNIST dataset (torchvision).
Uses raw pixel values (784 features).
"""

import torch
import pandas as pd
from torchvision import datasets
from torchvision.transforms import ToTensor
from pycaret.classification import (
    setup,
    compare_models,
    predict_model,
    pull,
    save_model,
    load_model
)
from typing import Dict


def mnist_to_dataframe(dataset) -> pd.DataFrame:
    data = dataset.data.view(len(dataset), -1).numpy()
    targets = dataset.targets.numpy()

    df = pd.DataFrame(data)
    df["target"] = targets
    return df


def train_pycaret_mnist(
    model_name: str = "pycaret_mnist_best"
):

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    df = mnist_to_dataframe(train_dataset)

    setup(
        data=df,
        target="target",
        session_id=42,
        normalize=True
    )

    best_model = compare_models()
    leaderboard = pull()

    save_model(best_model, model_name)

    return {
        "best_model": str(best_model),
        "leaderboard": leaderboard.to_dict()
    }



def predict_pycaret_mnist(
    model_name: str,
    use_test_set: bool = True
) -> pd.DataFrame:

    model = load_model(model_name)

    dataset = datasets.MNIST(
        root="data",
        train=not use_test_set,
        download=True,
        transform=ToTensor()
    )

    df = mnist_to_dataframe(dataset).drop(columns=["target"])

    predictions = predict_model(model, data=df)

    return predictions[["Label", "Score"]]


def score_pycaret(
    model_name: str
) -> Dict:

    model = load_model(model_name)

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    df = mnist_to_dataframe(test_dataset)

    predictions = predict_model(model, data=df)

    accuracy = (predictions["Label"] == df["target"]).mean()

    return {
        "accuracy": accuracy,
        "num_samples": len(df)
    }
