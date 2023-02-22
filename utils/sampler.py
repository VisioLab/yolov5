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
    """

    rank: int
    world_size: int
    subset: torch.Tensor
    replacement: bool
    generator: Optional[torch.Generator]

    def __init__(
        self,
        weights: Sequence[float],
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.world_size = 2 # dist.get_world_size()
        self.rank = 0 # dist.get_rank()

        self.replacement = replacement
        self.generator = generator
        self.subset = torch.as_tensor(weights[self.rank :: self.world_size])

    def __iter__(self) -> Iterator[T_co]:
        rand_tensor = torch.multinomial(
            self.subset, len(self.subset), self.replacement, generator=self.generator
        )
        # get indices respective to original dataset (not subset)
        rand_tensor = rand_tensor * self.world_size + self.rank
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return len(self.subset)

    def set_epoch(self, epoch):
        ...
