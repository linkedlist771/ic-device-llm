import random
from typing import Callable, List, Any


def ratio_uniform_sampler(input_list: List[Any], ratio: float) -> List[int]:
    assert 0 <= ratio <= 1, "Ratio must be between 0 and 1"
    assert len(input_list) > 0, "Input list must not be empty"

    sample_size = int(len(input_list) * ratio)
    return random.sample(range(len(input_list)), sample_size)


def random_sampler(input_list: List[Any], num_samples: int) -> List[int]:
    assert num_samples <= len(input_list), "Number of samples cannot exceed list length"
    return random.sample(range(len(input_list)), num_samples)


def sequential_sampler(input_list: List[Any], step: int) -> List[int]:
    return list(range(0, len(input_list), step))


def interval_sampler(input_list: List[Any], interval: int) -> List[int]:
    assert interval > 0, "Interval must be greater than 0"
    return list(range(0, len(input_list), interval))


def get_sampler(sampler_name: str) -> Callable:
    samplers = {
        "ratio_uniform": ratio_uniform_sampler,
        "random": random_sampler,
        "sequential": sequential_sampler,
        "interval": interval_sampler,
    }

    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    return samplers[sampler_name]
