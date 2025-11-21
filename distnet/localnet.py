## --------------------------------------------------------------------------------------------------------------------
## localnet.py
## basic neural net for qmnist handwritten digit classification
import torch
from torch import nn
from distnet.ring_allreduce import ring_reduce
from distnet.bucket import Bucket
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
    self.grad_params = [param for param in self.parameters() if param.requires_grad]
    # Setup buckets.
    bucket_cap_mb = 5 # maybe change this
    cap_bytes = bucket_cap_mb * 1024 * 1024
    self.buckets = []
    curr_bucket = []
    curr_bytes = 0

    for param in self.grad_params:
        #Fill buckets
        size = param.numel() * param.element_size()
        if curr_bucket and curr_bytes + size > cap_bytes:
            self.buckets.append(Bucket(curr_bucket))
            curr_bucket, curr_bytes = [], 0
        curr_bucket.append(param)
        curr_bytes += size
        # It might be better to just register gradient hooks here to save a second loop over parameters
        #param.register_hook(make_hook(param))
    # Add last unfilled bucket
    if curr_bucket:
        self.buckets.append(Bucket(curr_bucket)) 

  # Ring all reduce hook will look something like this, this should probably be moved to main.py
  def dist_hook(self, parameter, grad):
    for bucket in self.buckets:
      if id(parameter) in bucket.param_ids:
        #Add gradient to the bucket
        bucket.add_grad(parameter)
        if bucket.is_ready():
          ring_reduce(bucket.tensor)
          bucket.scatter_to_params()
          break
    return grad

  def load(self, filepath: str):
    self.model.load_state_dict(torch.load(filepath))

  def forward(self, x):
    return self.model(x)
