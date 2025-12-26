from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import torch
from torch import nn

from src.train import train_and_save
from api.utils import (
    load_model,
    save_model,
    csv_to_dataloader,
    list_models,
    DEVICE
)

app = FastAPI(title="MNIST CNN API")


@app.post("/continue-train")
async def continue_train(
    model_name: str = Form(...),
    new_model_name: str = Form(...),
    train_input: UploadFile = File(...)
):
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    train_loader = csv_to_dataloader(train_input.file, has_target=True)

    # Use MNIST test set again (same as your training script)
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    test_loader = DataLoader(test_data, batch_size=32)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    trained_model = train_and_save(
        model,
        train_loader,
        test_loader,
        epochs=5,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )

    save_model(trained_model, new_model_name)

    return {
        "message": "Training completed",
        "saved_model": new_model_name
    }


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    input: UploadFile = File(...)
):
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    loader = csv_to_dataloader(input.file, has_target=False)

    model.eval()
    predictions = []

    with torch.inference_mode():
        for (X,) in loader:
            X = X.to(DEVICE)
            preds = model(X).argmax(dim=1)
            predictions.extend(preds.cpu().tolist())

    return {"predictions": predictions}


@app.get("/models", response_model=List[str])
def get_models():
    return list_models()
