import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.model import MnistModel1

MODELS_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str):
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found")

    model = MnistModel1(1, 20, 10).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def save_model(model, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, model_name))


def csv_to_dataloader(file, batch_size=32, has_target=True):
    df = pd.read_csv(file)

    if has_target:
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        y = torch.tensor(y, dtype=torch.long)
    else:
        if "label" in df.columns:
            df = df.drop(columns=["label"])
            X = df.values
        y = None

    X = torch.tensor(X, dtype=torch.float32)
    X = X.view(-1, 1, 28, 28)  

    dataset = TensorDataset(X, y) if has_target else TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=has_target)


def list_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return [m for m in os.listdir(MODELS_DIR) if m.endswith(".pth")]
