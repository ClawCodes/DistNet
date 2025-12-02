## --------------------------------------------------------------------------------------------------------------------
## localnet.py
## basic neural net for qmnist handwritten digit classification
import torch
from torch import nn
from distnet.ring_allreduce import ring_reduce
from distnet.bucket import Bucket

from distnet.distnet import DistNet
from distnet.resnet import ResNetCIFAR


## Network Architecture -----------------------------------------------------------------------------------------------
class LocalNet(nn.Module):
  """ Fully-connected, 4 layer perceptron for CIFAR-10
  3072 node input layer (flattened 32x32x3 RGB image), fully connected to hidden layers with 512, 256, and 128 nodes
  """
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Flatten(), # 32x32x3 to 3072
        nn.Linear(3072, 512),
        nn.GELU(),
        # nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.GELU(),
        # nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 10))
  
  def load(self, filepath: str):
    self.model.load_state_dict(torch.load(filepath))

  def forward(self, x):
    return self.model(x)


class DistCNN(DistNet):
  """ Convolutional Neural Network for CIFAR-10
  Architecture: Conv(32) -> Conv(64) -> Conv(128) -> FC(256) -> FC(10)
  Input: 32x32x3 RGB images
  Parameters: ~0.68M (~2.73MB)

  After initialization call model.register_grad_hook(you_function) to register a per param gradient hook.
  """

  def __init__(self, bucket_size=5): # bucket size is 5 mb by default
    super().__init__(bucket_size)
    self.model = nn.Sequential(
      # Conv Block 1: 32x32x3 -> 16x16x32
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.GELU(),
      nn.MaxPool2d(2, 2),

      # Conv Block 2: 16x16x32 -> 8x8x64
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.GELU(),
      nn.MaxPool2d(2, 2),

      # Conv Block 3: 8x8x64 -> 4x4x128
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.GELU(),
      nn.MaxPool2d(2, 2),

      # Flatten and FC layers: 4x4x128 = 2048 -> 256 -> 10
      nn.Flatten(),
      nn.Linear(2048, 256),
      nn.GELU(),
      # nn.Dropout(0.2),
      nn.Linear(256, 10))

    self.grad_params = [param for param in self.parameters() if param.requires_grad]

    # Setup buckets
    bucket_cap_mb = bucket_size
    cap_bytes = bucket_cap_mb * 1024 * 1024
    self.buckets = []
    curr_bucket = []
    curr_bytes = 0

    for param in self.grad_params:
        size = param.numel() * param.element_size()
        if curr_bucket and (curr_bytes + size > cap_bytes):
            # finalize current bucket
            self.buckets.append(Bucket(list(curr_bucket)))
            curr_bucket, curr_bytes = [], 0
        curr_bucket.append(param)
        curr_bytes += size

    # add last bucket if any
    if curr_bucket:
        self.buckets.append(Bucket(list(curr_bucket)))

    # Set for quickly looking up bucket for a parameter
    self.param_id_to_bucket = {}
    for b in self.buckets:
        for p in b.params:
            self.param_id_to_bucket[id(p)] = b

    # Counters for debugging
    self.hookFireCount = 0
    self.reduceFireCount = 0

    # Communication time tracking
    self.comm_time = 0.0

  def dist_hook(self, parameter, grad):
    self.hookFireCount+=1
    bucket = self.param_id_to_bucket.get(id(parameter))
    if bucket is None:
      raise RuntimeError("Parameter not found in any bucket")
    bucket.add_grad_tensor(parameter, grad)
    if bucket.is_ready():
      self.reduceFireCount+=1
      comm_time = ring_reduce(bucket.tensor)
      self.comm_time += comm_time
      bucket.scatter_to_params()
    return grad

  def reset_buckets(self):
    for b in self.buckets:
      b.reset()

  def get_comm_time(self) -> float:
    """Get accumulated communication time and reset counter."""
    time = self.comm_time
    self.comm_time = 0.0
    return time

  def forward(self, x):
    return self.model(x)


class DistLocalNet(DistNet):
  """ Fully-connected, 4 layer perceptron for CIFAR-10
  3072 node input layer (flattened 32x32x3 RGB image), fully connected to hidden layers with 512, 256, and 128 nodes

  After initialization call model.register_grad_hook(you_function) to register a per param gradient hook.
  """

  def __init__(self, bucket_size=5): # bucket size is 5 mb by default
    super().__init__(bucket_size)
    self.model = nn.Sequential(
      nn.Flatten(),  # 32x32x3 to 3072
      nn.Linear(3072, 512),
      nn.GELU(),
      # nn.Dropout(0.2),
      nn.Linear(512, 256),
      nn.GELU(),
      # nn.Dropout(0.2),
      nn.Linear(256, 128),
      nn.GELU(),
      nn.Linear(128, 10))

    self.grad_params = [param for param in self.parameters() if param.requires_grad]

    # Setup buckets
    bucket_cap_mb = bucket_size
    cap_bytes = bucket_cap_mb * 1024 * 1024
    self.buckets = []
    curr_bucket = []
    curr_bytes = 0

    for param in self.grad_params:
        size = param.numel() * param.element_size()
        if curr_bucket and (curr_bytes + size > cap_bytes):
            # finalize current bucket
            self.buckets.append(Bucket(list(curr_bucket)))
            curr_bucket, curr_bytes = [], 0
        curr_bucket.append(param)
        curr_bytes += size

    # add last bucket if any
    if curr_bucket:
        self.buckets.append(Bucket(list(curr_bucket)))

    # Set for quickly looking up bucket for a parameter
    self.param_id_to_bucket = {}
    for b in self.buckets:
        for p in b.params:
            self.param_id_to_bucket[id(p)] = b

    # Counters for debugging
    self.hookFireCount = 0
    self.reduceFireCount = 0

    # Communication time tracking
    self.comm_time = 0.0

    # Register hooks
    #for param in self.grad_params:
    #    param.register_hook(self.dist_hook(param))

  # Ring all reduce hook will look something like this, this should probably be moved to main.py
  def dist_hook(self, parameter, grad):
    #def hook(grad):
    self.hookFireCount+=1
    #if self.hookFireCount < 10:  print('Firing hook: {}'.format(self.hookFireCount))
    bucket = self.param_id_to_bucket.get(id(parameter))
    if bucket is None:
      # shouldn't happen if buckets constructed correctly
      raise RuntimeError("Parameter not found in any bucket")
    #Add gradient to the bucket
    #if self.hookFireCount < 10: print('Bucket size: {}'.format(len(bucket.params)))
    bucket.add_grad_tensor(parameter, grad)
    if bucket.is_ready():
      #if self.hookFireCount < 19: print('Firing reduce on hook fire: {}'.format(self.hookFireCount))
      self.reduceFireCount+=1
      comm_time = ring_reduce(bucket.tensor)
      self.comm_time += comm_time
      bucket.scatter_to_params()
      #return grad
    return grad

  def reset_buckets(self):
    for b in self.buckets:
      b.reset()

  def get_comm_time(self) -> float:
    """Get accumulated communication time and reset counter."""
    time = self.comm_time
    self.comm_time = 0.0
    return time

  def load(self, filepath: str):
    self.model.load_state_dict(torch.load(filepath))

  def forward(self, x):
    return self.model(x)


class DistResNet(DistNet):
  """ ResNet-32 for CIFAR-10
  Architecture: ResNet with BasicBlocks
  Input: 32x32x3 RGB images
  Parameters: ~3.35M (~13.4MB)

  After initialization call model.register_grad_hook(you_function) to register a per param gradient hook.
  """

  def __init__(self, bucket_size=5):
    super().__init__(bucket_size)
    self.model = ResNetCIFAR(num_blocks_per_layer=(5, 5, 5), num_classes=10, base_channels=64)

    self.grad_params = [param for param in self.parameters() if param.requires_grad]

    # Setup buckets
    bucket_cap_mb = bucket_size
    cap_bytes = bucket_cap_mb * 1024 * 1024
    self.buckets = []
    curr_bucket = []
    curr_bytes = 0

    for param in self.grad_params:
        size = param.numel() * param.element_size()
        if curr_bucket and (curr_bytes + size > cap_bytes):
            # finalize current bucket
            self.buckets.append(Bucket(list(curr_bucket)))
            curr_bucket, curr_bytes = [], 0
        curr_bucket.append(param)
        curr_bytes += size

    # add last bucket if any
    if curr_bucket:
        self.buckets.append(Bucket(list(curr_bucket)))

    # Set for quickly looking up bucket for a parameter
    self.param_id_to_bucket = {}
    for b in self.buckets:
        for p in b.params:
            self.param_id_to_bucket[id(p)] = b

    # Counters for debugging
    self.hookFireCount = 0
    self.reduceFireCount = 0

    # Communication time tracking
    self.comm_time = 0.0

  def dist_hook(self, parameter, grad):
    self.hookFireCount+=1
    bucket = self.param_id_to_bucket.get(id(parameter))
    if bucket is None:
      raise RuntimeError("Parameter not found in any bucket")
    bucket.add_grad_tensor(parameter, grad)
    if bucket.is_ready():
      self.reduceFireCount+=1
      comm_time = ring_reduce(bucket.tensor)
      self.comm_time += comm_time
      bucket.scatter_to_params()
    return grad

  def reset_buckets(self):
    for b in self.buckets:
      b.reset()

  def get_comm_time(self) -> float:
    """Get accumulated communication time and reset counter."""
    time = self.comm_time
    self.comm_time = 0.0
    return time

  def forward(self, x):
    return self.model(x)


class ScalableNet(DistNet):
  def __init__(self, bucket_size, hidden_dim=512, layers=8, num_classes=10):
    super().__init__(bucket_size)
    self.flatten = nn.Flatten()
    in_dim = 32 * 32 * 3 # using CIFAR

    modules = []
    current_dim = in_dim

    for i in range(layers):
      modules.append(nn.Linear(current_dim, hidden_dim))
      modules.append(nn.ReLU())
      current_dim = hidden_dim

    modules.append(nn.Linear(hidden_dim, num_classes))
    self.model = nn.Sequential(*modules)
    self.init_comms() # Must be called after self.model is initialized

  def forward(self, x):
    x = self.flatten(x)
    return self.model(x)