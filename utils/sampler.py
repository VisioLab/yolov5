from dataclasses import dataclass
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)


@dataclass
class DistributedWeightedSampler(Sampler[T_co]):
    """Sampler that draws a weighted sample from a subset of the data.

    Each worker is assigned a fixed subset of the dataset. The size of
    the sample is always the size of the subset.

    Args:
        weights: weights for random sampling
        replacement: whether to replace the samples
    """
    weights: torch.Tensor
    replacement: bool = True
    generator: Optional[torch.Generator] = None

    def __post_init__(self) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        self.subset = self.weights[rank::num_replicas]

    def __iter__(self) -> Iterator[T_co]:
        rand_tensor = torch.multinomial(self.subset, len(self.subset), self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return len(self.subset)
