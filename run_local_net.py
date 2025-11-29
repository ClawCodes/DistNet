import torch

from distnet.localnet import LocalNet
from distnet.resnet import ResNetCIFAR
from distnet.util import load_mnist, train, test, save, load_cifar100

BATCH_SIZE: int = 32
EPOCHS: int = 16

if __name__ == "__main__":
  # train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)
  #
  # model = LocalNet().to(torch.device('cpu'))
  #
  # train(model, train_loader, epochs=EPOCHS)
  #
  # test(model, test_loader)
  #
  # save(model, 'localnet')

  train_loader, test_loader = load_cifar100(batch_size=BATCH_SIZE)

  model = ResNetCIFAR().to(torch.device('cpu'))

  train(model, train_loader, epochs=EPOCHS)

  test(model, test_loader)

  save(model, 'resnet_cifar')

