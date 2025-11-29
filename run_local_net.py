import torch

from distnet.localnet import LocalNet
from distnet.util import load_cifar10, train, test, save

BATCH_SIZE: int = 32
EPOCHS: int = 16

if __name__ == "__main__":
  train_loader, test_loader = load_cifar10(batch_size=BATCH_SIZE)

  model = LocalNet().to(torch.device('cpu'))

  train(model, train_loader, epochs=EPOCHS)

  test(model, test_loader)

  save(model, 'localnet')

