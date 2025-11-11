from typing import Tuple

from torchvision import datasets, transforms
import torch
from torch import nn
import time
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import random


def set_seed(seed=42):
    """
    Function to call before training if you want deterministic results.

    Use for testing.

    Example:
        set_seed()
        train(model, train_loader, epochs=32)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(batch_size: int = 32, batch_multiplier: int = 16, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    raw = datasets.QMNIST(root='./data', what='train', download=True, transform=transforms.ToTensor())

    # compute normalization transform
    transform_data = torch.cat([x for x, _ in raw], dim=0)
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(transform_data.mean(), transform_data.std())])

    # load normalized data
    qmnist_train = datasets.QMNIST(root='./data', what='train', download=True, transform=normalize_transform)
    qmnist_test = datasets.QMNIST(root='./data', what='test', download=True, transform=normalize_transform)

    # wrap in data loaders
    train_loader = torch.utils.data.DataLoader(qmnist_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(qmnist_test, batch_size=batch_multiplier * batch_size, shuffle=shuffle)

    return train_loader, test_loader

def train(model: nn.Module, train_loader: DataLoader, epochs: int = 16) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start = time.perf_counter()
    # train on data EPOCHS number of time
    for epoch in range(epochs):
        # initialize per epoch variables
        model.train()
        epoch_loss = 0.0

        # optimize parameters by batch-averaged loss gradient
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # output current loss
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
    # output training time
    end = time.perf_counter()
    print(f"Training time: {end - start:.2f} seconds")

    return model

def test(model: nn.Module, test_loader: DataLoader) -> float:
    print("Evaluating model performance...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # disable caching layer evaluations
        for images, labels in test_loader:
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {100 * correct / total:.2f}%")

    return accuracy

def save(model: nn.Module, name: str):
    torch.save(model.state_dict(), f"./models/{name}.pth")