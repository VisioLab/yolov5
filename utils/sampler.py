from typing import Sequence, TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler
import torch.distributed as dist

T_co = TypeVar("T_co", covariant=True)


class DistributedWeightedSampler(Sampler[T_co]):
    """Sampler that draws a weighted sample from a subset of the data.

    Each worker is assigned a fixed subset of the dataset. The size of
    the sample is always the size of the subset.

    Args:
        weights: weights for random sampling
        replacement: whether to replace the samples
        seed: random seed
    """

    rank: int
    world_size: int
    weights: torch.Tensor
    size: int
    seed: int
    epoch: int
    generator: torch.Generator
    replacement: bool = True

    def __init__(
        self,
        weights: Sequence[float],
        replacement: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.replacement = replacement
        self.generator = torch.Generator()
        if seed is None:
            self.seed = self.generator.initial_seed()
        else:
            self.seed = seed
        self.size = len(weights[self.rank::self.world_size])
        self.weights = torch.as_tensor(weights)

    def __iter__(self) -> Iterator[T_co]:
        self.generator.manual_seed(self.seed + self.epoch)
        rand_tensor = torch.multinomial(
            self.weights, len(self.weights), self.replacement, generator=self.generator
        )
        rand_tensor = rand_tensor[self.rank::self.world_size]
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch):
        """Ensure that each replica gets a different random sample."""
        self.epoch = epoch


if __name__ == '__main__':
    from unittest.mock import patch
    weights = [1/2, 1/2, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    with patch('utils.sampler.dist.get_world_size', lambda: 2), patch('utils.sampler.dist.get_rank', lambda: 0):
        sampler = DistributedWeightedSampler(weights)
        samples = []
        for e in range(5000):
            sampler.set_epoch(e)
            samples.extend(list(sampler))
        assert round(sum(samples.count(x) for x in range(2)) / 1000) == round(sum(samples.count(x) for x in range(2, 8)) / 1000)
    print('Distributed sampler test succeeded.')
