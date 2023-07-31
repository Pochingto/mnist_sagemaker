
import os
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(2304, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.conv4(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.conv5(x)
        x = F.relu(x)
        # print(x.shape)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
    
def load_data(train_path, validation_path, batch_size):
    train = pd.read_csv(Path(train_path) / "train.csv", header=None)
    X_train = train.iloc[:, :-1].to_numpy()
    y_train = train.iloc[:, -1].to_numpy()
    validation = pd.read_csv(Path(validation_path) / "validation.csv", header=None)
    X_validation = validation.iloc[:, :-1].to_numpy()
    y_validation = validation.iloc[:, -1].to_numpy()
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).reshape((-1, 1, 28, 28)).float(), torch.from_numpy(y_train.astype("int")))
    valid_dataset = TensorDataset(torch.from_numpy(X_validation).reshape((-1, 1, 28, 28)).float(), torch.from_numpy(y_validation.astype("int")))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader
    
def train_one_epoch(epoch, model, device, optimizer, train_dataloader, valid_dataloader, loss_func):
    model.train()
    total_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y = F.one_hot(y, 10).to(device).float()
        
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(y, output)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().item()

    print(f"Train Epoch: {epoch} \t Avg. loss: {total_loss / len(train_dataloader.dataset):.6f}")
    
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)
            y = F.one_hot(y, 10).to(device).float()
            output = model(X)
            loss = loss_func(y, output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            valid_loss += loss.detach().item()

    valid_loss /= len(valid_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_dataloader.dataset),
        100. * correct / len(valid_dataloader.dataset)))
    
def train(base_directory, train_path, validation_path, epochs, batch_size):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model = Net().to(device).float()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    print(f"Train path: {train_path}")
    print(f"Validation path: {validation_path}")
    train_dataloader, valid_dataloader = load_data(train_path, validation_path, batch_size)
    
    print("Start training...")
    for epoch in range(1, epochs + 1):
        train_one_epoch(epoch, model, device, optimizer, train_dataloader, valid_dataloader, loss_func)
    
    
if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to the entry point
    # as script arguments. SageMaker will also provide a list of special parameters
    # that you can capture here. Here is the full list: 
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default="/opt/ml/")
    parser.add_argument("--train_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--validation_path", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args, _ = parser.parse_known_args()
    
    base_directory, epochs, train_path, validation_path, batch_size = args.base_directory, args.epochs, args.train_path, args.validation_path, args.batch_size
    train(base_directory, train_path, validation_path, epochs, batch_size)
    
