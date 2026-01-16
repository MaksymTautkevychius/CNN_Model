from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import torch
from torch import nn
import sys
import os

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


# ============================================================================
# CONSOLE INTERFACE
# ============================================================================

def console_continue_train(model_name: str, new_model_name: str, train_input_path: str):
    print(f"\n{'='*60}")
    print("CONTINUE TRAINING")
    print(f"{'='*60}")
    
    try:
        print(f"Loading model: {model_name}...")
        model = load_model(model_name)
        print(" Model loaded successfully")
    except FileNotFoundError:
        print(f" Error: Model '{model_name}' not found")
        return
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    try:
        print(f"Loading training data from: {train_input_path}...")
        with open(train_input_path, 'rb') as f:
            train_loader = csv_to_dataloader(f, has_target=True)
        print(" Training data loaded successfully")
    except FileNotFoundError:
        print(f" Error: Training file '{train_input_path}' not found")
        return
    except Exception as e:
        print(f" Error loading training data: {e}")
        return

    print("Loading MNIST test dataset...")
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
    print("✓ Test data loaded successfully")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nStarting training on device: {DEVICE}")
    print(f"Training for 5 epochs...")
    
    trained_model = train_and_save(
        model,
        train_loader,
        test_loader,
        epochs=5,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE
    )

    print(f"\nSaving model as: {new_model_name}...")
    save_model(trained_model, new_model_name)
    print(f" Model saved successfully as '{new_model_name}'")
    print(f"{'='*60}\n")


def console_predict(model_name: str, input_path: str, output_path: str = None):
    print(f"\n{'='*60}")
    print("PREDICTION")
    print(f"{'='*60}")
    
    try:
        print(f"Loading model: {model_name}...")
        model = load_model(model_name)
        print(" Model loaded successfully")
    except FileNotFoundError:
        print(f" Error: Model '{model_name}' not found")
        return
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    try:
        print(f"Loading input data from: {input_path}...")
        with open(input_path, 'rb') as f:
            loader = csv_to_dataloader(f, has_target=False)
        print(" Input data loaded successfully")
    except FileNotFoundError:
        print(f" Error: Input file '{input_path}' not found")
        return
    except Exception as e:
        print(f" Error loading input data: {e}")
        return

    print("Making predictions...")
    model.eval()
    predictions = []

    with torch.inference_mode():
        for (X,) in loader:
            X = X.to(DEVICE)
            preds = model(X).argmax(dim=1)
            predictions.extend(preds.cpu().tolist())

    print(f" Generated {len(predictions)} predictions")
    
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write("prediction\n")
                for pred in predictions:
                    f.write(f"{pred}\n")
            print(f"✓ Predictions saved to: {output_path}")
        except Exception as e:
            print(f"✗ Error saving predictions: {e}")
    else:
        print(f"\nPredictions: {predictions[:20]}{'...' if len(predictions) > 20 else ''}")
    
    print(f"{'='*60}\n")
    return predictions


def console_list_models():
    print(f"\n{'='*60}")
    print("AVAILABLE MODELS")
    print(f"{'='*60}")
    
    models = list_models()
    
    if models:
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
    else:
        print("No models found")
    
    print(f"{'='*60}\n")
    return models


def console_menu():
    while True:
        print("\n" + "="*60)
        print("MNIST CNN CONSOLE INTERFACE")
        print("="*60)
        print("1. Continue training an existing model")
        print("2. Make predictions with a model")
        print("3. List all available models")
        print("4. Exit")
        print("="*60)
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            model_name = input("Enter existing model name: ").strip()
            new_model_name = input("Enter new model name to save: ").strip()
            train_input_path = input("Enter path to training CSV file: ").strip()
            console_continue_train(model_name, new_model_name, train_input_path)
            
        elif choice == "2":
            model_name = input("Enter model name: ").strip()
            input_path = input("Enter path to input CSV file: ").strip()
            save_choice = input("Save predictions to file? (y/n): ").strip().lower()
            
            if save_choice == 'y':
                output_path = input("Enter output file path: ").strip()
                console_predict(model_name, input_path, output_path)
            else:
                console_predict(model_name, input_path)
                
        elif choice == "3":
            console_list_models()
            
        elif choice == "4":
            print("\nExiting...")
            break
            
        else:
            print("\n Invalid option. Please select 1-4.")


def run_console():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        print("TO START API PLEASE USE uvicorn main:app --reload")
        
        if command == "train" and len(sys.argv) == 5:
            console_continue_train(sys.argv[2], sys.argv[3], sys.argv[4])
        elif command == "predict" and len(sys.argv) >= 4:
            output = sys.argv[4] if len(sys.argv) == 5 else None
            console_predict(sys.argv[2], sys.argv[3], output)
        elif command == "list":
            console_list_models()
        else:
            print("Usage:")
            print("  python main.py train <model_name> <new_model_name> <train_csv>")
            print("  python main.py predict <model_name> <input_csv> [output_file]")
            print("  python main.py list")
            print("  python main.py console  (for interactive menu)")
    else:
        console_menu()


if __name__ == "__main__":
    run_console()