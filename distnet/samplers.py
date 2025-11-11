from typing import Iterator, Optional, Sized
import math
import torch
from torch.utils.data import Dataset, Sampler


class DistributedSampler(Sampler[int]): # Inherit Sampler
    """
    Args:
        world_size: Total number of processes (nodes/GPUs) participating 
                    in distributed training (# of nodes * GPU/node)
        rank:       Rank of the current process (0 to world_size-1)
        drop_last:  If True, drops the tail of the data to make it evenly
                    divisible across the number of replicas. Default: False.

    Example:
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
        sampler = DistributedSampler(dataset, world_size=4, rank=0)
        loader  = torch.utils.data.DataLoader(dataset, sampler=sampler)
        
        # Training loop:
        for epoch in range(num_epochs):
             sampler.set_epoch(epoch)  # Ensure different shuffle each epoch
             for batch in loader:
                 # training
    """

    def __init__(
        self,
        dataset: Dataset,
        world_size: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(dataset, Sized):  # Check if dataset has __len__ method
            raise TypeError("Dataset must implement __len__ method")

        if rank >= world_size or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the range [0, {world_size - 1}]"
            )

        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        if self.drop_last and len(self.dataset) % self.world_size != 0:
            self.num_samples = math.floor(len(self.dataset) / self.world_size)
        else:   # Padding required
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)

        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # randperm() return a tensor with random permutation of 0 to 
            # len(dataset)-1
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                # eg, [0, ..., 9] + [0, 1, 2] = [0, ..., 9, 0, 1, 2]
                indices += indices[:padding_size]
        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Key logic: sharding
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples

        return iter(indices)    # return iterator

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
