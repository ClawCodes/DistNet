## --------------------------------------------------------------------------------------------------------------------
## localnet.py
## basic neural net for qmnist handwritten digit classification
import torch
from torch import nn

from distnet.distnet import DistNet


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


class DistLocalNet(DistNet):
  """ Fully-connected, 3 layer perceptron
  784 node input layer (flattened 28x28 pixel image), fully connected to hidden layers with 128 and 64 nodes each

  After initialization call model.register_grad_hook(you_function) to register a per param gradient hook.
  """

  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Flatten(),  # 28x28 to 784
      nn.Linear(784, 128),
      nn.GELU(),
      nn.Linear(128, 64),
      nn.GELU(),
      nn.Linear(64, 10))

  def load(self, filepath: str):
    self.model.load_state_dict(torch.load(filepath))

  def forward(self, x):
    return self.model(x)
