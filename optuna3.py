# src/optuna_search.py
import optuna
import torch
from torch import nn
from src.model import MnistModel1
from src.train import train_and_save
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial):

    hidden_units = trial.suggest_int("hidden_units", 8, 64)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)


    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)


    model = MnistModel1(
        input_shape=1,
        hidden_units=hidden_units,
        output_shape=10
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()


    trained_model = train_and_save(
        model,
        train_loader,
        test_loader,
        epochs=3,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )


    trained_model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = trained_model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def search(n_trials: int = 20):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_accuracy": study.best_value,
        "best_params": study.best_params
    }  

def test_search():
    results = search(n_trials=2)

    assert "best_accuracy" in results
    assert "best_params" in results

    best_acc = results["best_accuracy"]
    best_params = results["best_params"]

    assert isinstance(best_acc, float)
    assert 0.0 <= best_acc <= 1.0

    assert "hidden_units" in best_params
    assert "lr" in best_params

    print("Best accuracy:", best_acc)
    print("Best params:", best_params)
test_search()