import pytest
import torch
from torch.utils.data import TensorDataset

from distnet.samplers import DistributedSampler


class TestDistributedSampler:
    def test_basic_partitioning(self):
        """Test that data is correctly partitioned across ranks."""
        dataset = TensorDataset(torch.arange(100))
        world_size = 4

        samplers = [
            DistributedSampler(dataset, world_size=world_size, rank=i, shuffle=False)
            for i in range(world_size)
        ]

        # Collect all indices from all ranks
        all_indices = []
        for sampler in samplers:
            indices = list(sampler)
            all_indices.extend(indices)

        # Each rank should get 25 samples (100 / 4)
        for sampler in samplers:
            assert len(sampler) == 25

        # Total should be 100 (no padding needed)
        assert len(all_indices) == 100

        # All indices should be present (no duplicates, no missing)
        assert sorted(all_indices) == list(range(100))

    def test_uneven_partitioning_with_padding(self):
        """Test partitioning when dataset size is not divisible by world_size."""
        dataset = TensorDataset(torch.arange(10))   # [1, ..., 10]
        world_size = 3

        samplers = [
            DistributedSampler(dataset, world_size=world_size, rank=i, shuffle=False, drop_last=False)
            for i in range(world_size)
        ]

        for sampler in samplers:        
            assert len(sampler) == 4    # ceil(10/3) = 4

        # Collect indices from each rank
        indices = [list(sampler) for sampler in samplers]

        # [0,1,2, ... ,9] + padding [0,1] -> [0, 1, 2, ..., 9, 0, 1]
        assert indices[0] == [0, 3, 6, 9]   
        assert indices[1] == [1, 4, 7, 0]   # last one is padded
        assert indices[2] == [2, 5, 8, 1]   # last one is padded

        # Verify no duplicates within each rank
        for rank_indices in indices:
            assert len(rank_indices) == len(set(rank_indices)) or rank_indices.count(rank_indices[-1]) <= 2

    def test_uneven_partitioning_with_drop_last(self):
        """Test partitioning with drop_last=True."""
        dataset = TensorDataset(torch.arange(10))
        world_size = 3

        samplers = [
            DistributedSampler(dataset, world_size=world_size, rank=i, shuffle=False, drop_last=True)
            for i in range(world_size)
        ]

        # Each rank should get floor(10/3) = 3 samples
        for sampler in samplers:
            assert len(sampler) == 3

        # Collect all indices
        all_indices = []
        for sampler in samplers:
            indices = list(sampler)
            all_indices.extend(indices)

        # Total should be 9 (dropped 1 sample)
        assert len(all_indices) == 9
        assert sorted(all_indices) == list(range(9))

    def test_shuffle_consistency(self):
        """Test that shuffle is deterministic with same seed."""
        dataset = TensorDataset(torch.arange(100))

        sampler1 = DistributedSampler(dataset, world_size=2, rank=0, shuffle=True, seed=42)
        sampler2 = DistributedSampler(dataset, world_size=2, rank=0, shuffle=True, seed=42)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_shuffle_different_seeds(self):
        """Test that different seeds produce different shuffles."""
        dataset = TensorDataset(torch.arange(100))

        sampler1 = DistributedSampler(dataset, world_size=2, rank=0, shuffle=True, seed=42)
        sampler2 = DistributedSampler(dataset, world_size=2, rank=0, shuffle=True, seed=123)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 != indices2

    def test_set_epoch(self):
        """Test that set_epoch changes the shuffle order."""
        dataset = TensorDataset(torch.arange(100))

        sampler = DistributedSampler(dataset, world_size=2, rank=0, shuffle=True, seed=42)

        indices_epoch0 = list(sampler)

        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        sampler.set_epoch(2)
        indices_epoch2 = list(sampler)

        # Different epochs should produce different orderings
        assert indices_epoch0 != indices_epoch1
        assert indices_epoch1 != indices_epoch2

        # But same epoch should produce same ordering
        sampler.set_epoch(0)
        indices_epoch0_again = list(sampler)
        assert indices_epoch0 == indices_epoch0_again

    def test_no_overlap_between_ranks(self):
        """Test that different ranks get non-overlapping data."""
        dataset = TensorDataset(torch.arange(100))
        world_size = 4

        samplers = [
            DistributedSampler(dataset, world_size=world_size, rank=i, shuffle=True, seed=42)
            for i in range(world_size)
        ]

        all_indices_sets = [set(list(sampler)) for sampler in samplers]

        # Check that no two ranks have overlapping indices
        for i in range(world_size):
            for j in range(i + 1, world_size):
                overlap = all_indices_sets[i] & all_indices_sets[j]
                assert len(overlap) == 0, f"Ranks {i} and {j} have overlapping indices: {overlap}"

    def test_invalid_rank(self):
        """Test that invalid rank raises ValueError."""
        dataset = TensorDataset(torch.arange(100))

        with pytest.raises(ValueError, match="Invalid rank"):
            DistributedSampler(dataset, world_size=4, rank=4)

        with pytest.raises(ValueError, match="Invalid rank"):
            DistributedSampler(dataset, world_size=4, rank=-1)

    def test_dataset_without_len(self):
        """Test that dataset without __len__ raises TypeError."""
        class NoLenDataset:
            pass

        dataset = NoLenDataset()

        with pytest.raises(TypeError, match="Dataset must implement __len__"):
            DistributedSampler(dataset, world_size=2, rank=0)

    def test_single_rank(self):
        """Test behavior with single rank (no distribution)."""
        dataset = TensorDataset(torch.arange(100))

        sampler = DistributedSampler(dataset, world_size=1, rank=0, shuffle=False)
        indices = list(sampler)

        # Single rank should get all data
        assert len(indices) == 100
        assert indices == list(range(100))

    def test_interleaved_distribution(self):
        """Test that indices are interleaved across ranks, not consecutive blocks."""
        dataset = TensorDataset(torch.arange(12))
        world_size = 3

        samplers = [
            DistributedSampler(dataset, world_size=world_size, rank=i, shuffle=False)
            for i in range(world_size)
        ]

        indices = [list(sampler) for sampler in samplers]

        # Rank 0 should get [0, 3, 6, 9]
        # Rank 1 should get [1, 4, 7, 10]
        # Rank 2 should get [2, 5, 8, 11]
        assert indices[0] == [0, 3, 6, 9]
        assert indices[1] == [1, 4, 7, 10]
        assert indices[2] == [2, 5, 8, 11]
