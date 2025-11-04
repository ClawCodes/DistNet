## --------------------------------------------------------------------------------------------------------------------
## localnet.py
## basic neural net for qmnist handwritten digit classification

## Imports and Constants-----------------------------------------------------------------------------------------------
import torch
from torch import nn, optim
from torchvision import datasets, transforms
BATCH_SIZE: int = 64
EPOCHS:     int = 8

## Prepare dataset ----------------------------------------------------------------------------------------------------
# load raw training data
raw = datasets.QMNIST(root='./data',what='train',download=True)

# compute normalization transform
transform_data = torch.cat([x for x, _ in raw], dim=0)
normalize_transform = torch.transform.Compose([
  transforms.ToTensor(),
  transforms.Normalize((transform_data.mean(),transform_data.std()))])

# load normalized data
qmnist_train = datasets.QMNIST(root='./data',what='train',download=True,transform=normalize_transform)
qmnist_test  = datasets.QMNIST(root='./data',what='test',download=True,transform=normalize_transform)

# wrap in data loaders
train_loader = torch.utils.data.DataLoader(qmnist_train, batchsize=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(qmnist_test, batchsize=16*BATCH_SIZE)

## Network Architecture -----------------------------------------------------------------------------------------------
class LocalNet(nn.Module):
  """ Fully-connected, 3 layer perceptron
  784 node input layer (flattened 28x28 pixel image), fully connected to hidden layers with 128 and 64 nodes each
  """
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Flatten(), # 28x28 to 784
        nn.Linear(784, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, 10))

  def forward(self, x):
    return self.model(x)

# instantiate model
model = LocalNet().to(torch.device('cpu'))

## Training -----------------------------------------------------------------------------------------------------------
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters())

# train on data EPOCHS number of time
for epoch in range(EPOCHS):
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