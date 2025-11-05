## --------------------------------------------------------------------------------------------------------------------
## localnet.py
## basic neural net for qmnist handwritten digit classification
from torch import nn

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
  
  def load(self, filepath: str):
    self.model.load_state_dict(torch.load(filepath))

  def forward(self, x):
    return self.model(x)

if (__name__ == "__main__"):
  # Test the neural network
  ## Imports and Constants---------------------------------------------------------------------------------------------
  import time
  import torch
  from torch import optim
  from torchvision import datasets, transforms
  BATCH_SIZE: int = 32
  EPOCHS:     int = 16

  ## Prepare dataset --------------------------------------------------------------------------------------------------
  print("Loading and normalizing datasets...")

  # load raw training data
  raw = datasets.QMNIST(root='./data',what='train',download=True,transform=transforms.ToTensor())

  # compute normalization transform
  transform_data = torch.cat([x for x, _ in raw], dim=0)
  normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(transform_data.mean(),transform_data.std())])

  # load normalized data
  qmnist_train = datasets.QMNIST(root='./data',what='train',download=True,transform=normalize_transform)
  qmnist_test  = datasets.QMNIST(root='./data',what='test',download=True,transform=normalize_transform)

  # wrap in data loaders
  train_loader = torch.utils.data.DataLoader(qmnist_train, batch_size=BATCH_SIZE, shuffle=True)
  test_loader  = torch.utils.data.DataLoader(qmnist_test, batch_size=16*BATCH_SIZE)

  ## Create Model and Perform Training ------------------------------------------------------------------------------
  # instantiate model
  model = LocalNet().to(torch.device('cpu'))

  print("Training model...")
  # loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  start = time.perf_counter()
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
    # output current loss
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
  # output training time
  end = time.perf_counter()
  print(f"Training time: {end - start:.2f} seconds")

  ## Evaluation -------------------------------------------------------------------------------------------------------
  print("Evaluating model performance...")
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad(): # disable caching layer evaluations
    for images, labels in test_loader:
      output = model(images)
      pred = output.argmax(dim=1)
      correct += (pred == labels).sum().item()
      total += labels.size(0)
  print(f"Test accuracy: {100*correct/total:.2f}%")

  ## Save Trained Model -----------------------------------------------------------------------------------------------
  torch.save(model.state_dict(), "./distnet/localnet.pth")