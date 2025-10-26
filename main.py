import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.model import MnistModel1
from src.train import train_and_save
epochs=10
BATCH_SIZE = 32
loss_fn = nn.CrossEntropyLoss()
torch.manual_seed(42)



train_data = datasets.MNIST(
    root="data",
    train=True, 
    download=True, 
    transform=ToTensor(),
    target_transform=None 
)
class_names=train_data.classes

test_data = datasets.MNIST(
    root="data",
    train=False, 
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE,
    shuffle=True 
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False 
)
device="cuda"
model1 = MnistModel1(1,20,len(class_names)).to(device)
optimizer = torch.optim.SGD(params=model1.parameters(),lr=0.01)

NewModel= train_and_save(model1,train_dataloader,test_dataloader,50,loss_fn,optimizer)

