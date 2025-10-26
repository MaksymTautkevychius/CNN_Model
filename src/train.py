import torch
from torch import nn
from tqdm.auto import tqdm
from src.serviceMethods import print_train_time,accuracy_fn
from timeit import default_timer as timer 
import matplotlib.pyplot as plt

train_losses = []
test_losses = []
test_accuracies = []

def train_and_save(model_0, train_dataloader, test_dataloader, epochs, loss_fn, optimizer, device="cuda") -> nn.Module:
    model_0.to(device)  

    for epoch in range(epochs):  
        print(f"Epoch: {epoch}\n-------")
        train_loss = 0
        model_0.train()

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)  
            y_pred = model_0(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

    


        ### Testing
        test_loss, test_acc = 0, 0
        model_0.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)  

                test_pred = model_0(X)

                test_loss += loss_fn(test_pred, y).item()  
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        test_losses.append(test_loss)  # Store test loss for plotting
        test_accuracies.append(test_acc)

        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
        if test_acc>=98.50:
            torch.save(model_0.state_dict(), "mnist_model.pth")
    return model_0


