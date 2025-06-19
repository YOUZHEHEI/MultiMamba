"""
Fixed torch_utils.py with proper single GPU support
"""
import os
import random
from typing import Callable, Optional

import numpy as np
import torch

# === Randomness ===


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """
    Fixed worker init function that handles both distributed and single GPU setups.
    
    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    # Handle both distributed and single GPU setups
    global_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    process_seed = torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


# === BFloat16 Support ===


def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist

        return (
            (torch.version.cuda is not None)
            and torch.cuda.is_bf16_supported()
            and (packaging.version.parse(torch.version.cuda).release >= (11, 0))
            and dist.is_nccl_available()
            and (nccl.version() >= (2, 10))
        )

    except Exception:
        return False


def check_bloat16_supported_single_gpu() -> bool:
    """Check BFloat16 support for single GPU setup (without distributed requirements)."""
    try:
        return (
            torch.version.cuda is not None
            and torch.cuda.is_bf16_supported()
        )
    except Exception:
        return False