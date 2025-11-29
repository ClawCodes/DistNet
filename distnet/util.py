import json
from pathlib import Path
from typing import Tuple

import time
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.distributed as dist

from distnet import DistributedSampler


def broadcast_model(model, src=0):
    # broadcast parameters in single call
    vec = parameters_to_vector(model.parameters())
    dist.broadcast(vec, src=src)
    vector_to_parameters(vec, model.parameters())

    # broadcast any buffers individually
    for buf in model.buffers():
        dist.broadcast(buf.data, src=src)

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

def load_cifar100(batch_size: int = 128, num_workers: int = 2):

    # Standard CIFAR-100 training augmentations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)),
    ])

    # Test/validation transform (no augmentation, deterministic)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


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

def load_cifar10(batch_size: int = 32, batch_multiplier: int = 16, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    # CIFAR-10 standard normalization (per-channel mean and std from entire dataset)
    # These values are commonly used for CIFAR-10
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2470, 0.2435, 0.2616])
    ])

    # load normalized data
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=normalize_transform)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=normalize_transform)

    # wrap in data loaders
    train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_multiplier * batch_size, shuffle=shuffle)

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

def distributed_train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, batch_size: int, epochs: int , outfile: Path) -> nn.Module:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    train_info = {
        "epochs": epochs,
        "batch_size": batch_size,
        "experiment": "speedup",
        "world_size": world_size,
        "rank": rank,
        "metrics": {
            "epoch_times": [],
            "losses": [],
            "accuracies": [],
            "comm_times": [],
            "compute_times": []
        }
    }

    sampler = DistributedSampler(
        dataset=train_loader.dataset,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        drop_last=False,
    )

    # Use local batch size: divide global batch by world size for strong scaling
    local_batch_size = batch_size // world_size
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,  # Keep false, sampler handles shuffling
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train on data EPOCHS number of time
    for epoch in range(epochs):
        epoch_start = time.perf_counter()   # Start timer for "each" epoch
        sampler.set_epoch(epoch)
        # initialize per epoch variables
        model.train()
        epoch_loss = 0.0

        # optimize parameters by batch-averaged loss gradient
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            model.reset_buckets()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.perf_counter() - epoch_start

        # Get communication time from model
        epoch_comm_time = model.get_comm_time()
        epoch_compute_time = epoch_time - epoch_comm_time

        train_info["metrics"]["losses"].append(avg_loss)
        train_info["metrics"]["epoch_times"].append(epoch_time)
        train_info["metrics"]["comm_times"].append(epoch_comm_time)
        train_info["metrics"]["compute_times"].append(epoch_compute_time)

        # Test after each epoch for convergence analysis
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                output = model(images)
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        train_info["metrics"]["accuracies"].append(accuracy)

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, time={epoch_time:.2f}s, acc={accuracy*100:.2f}%")

    # Write all the data to JSON at once when training finishes
    with open(outfile, "w") as f:
        json.dump(train_info, f, indent=4)

    return model

def distributed_test(model: nn.Module, test_loader: DataLoader, outfile: Path) -> float:
    print("Final evaluation...")
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
    print(f"Final test accuracy: {100 * accuracy:.2f}%")

    # Note: accuracies already saved during training in metrics
    # This is just final confirmation
    return accuracy

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