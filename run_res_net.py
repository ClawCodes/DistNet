import torch

from distnet.localnet import ScalableNet
from distnet.resnet import ResNetCIFAR
from distnet.util import train, test, save, load_cifar100, load_cifar10

BATCH_SIZE: int = 32
EPOCHS: int = 16

if __name__ == "__main__":
  # train_loader, test_loader = load_cifar100(batch_size=BATCH_SIZE)

  train_loader, test_loader = load_cifar10(batch_size=BATCH_SIZE)

  model = ScalableNet(5, layers=3).to(torch.device('cpu'))

  train(model, train_loader, epochs=EPOCHS)

  test(model, test_loader)

  # save(model, 'resnet_cifar')

