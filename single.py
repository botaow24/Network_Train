#Import PyTorch and Torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train(dataloader, model, loss_fn, optimizer):
    t_start = time.time()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print(batch)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    print(f"Time taken:{time.time()-t_start:>3f}")

def SingleTrain (epochs):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

def trainNetWork(dataloader, model, loss_fn, optimizer):
    t_start = time.time()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print(batch)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    print(f"Time taken:{time.time()-t_start:>3f}")

def NetworkTrain (epochs, num_devices):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    # Get cpu or gpu device for training.
device = "mps"

print(f"Using {device} device")

#from torchvision.models import resnet50, ResNet50_Weights
#Resmodel = resnet50(weights = ResNet50_Weights.DEFAULT)

#from torchvision.models import resnet34, ResNet34_Weights
#Resmodel = resnet34(weights = ResNet34_Weights.DEFAULT)

from torchvision.models import resnet18, ResNet18_Weights
Resmodel = resnet18(weights = ResNet18_Weights.DEFAULT)


num_classes = 10          # set output class number 
num_ftrs = Resmodel.fc.in_features
Resmodel.fc = nn.Linear(num_ftrs, num_classes)


model = Resmodel.to(device)   # set current net

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
training_dataSVHN = datasets.SVHN(
    root="data",
    split ='train',
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_dataSVHN = datasets.SVHN(
    root="data",
    split ='test',
    download=True,
    transform=ToTensor(),
)

batch_size = 384

# Create data loaders.
train_dataloader = DataLoader(training_dataSVHN, batch_size=batch_size)
test_dataloader = DataLoader(test_dataSVHN, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


epochs = 10
SingleTrain(epochs)
print("Done!")