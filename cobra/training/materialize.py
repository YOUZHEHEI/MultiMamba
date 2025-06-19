"""
Updated materialize.py for training strategies with single GPU support
"""
import os
from typing import Callable, Optional

import torch

from cobra.models.vlms import CobraVLM
from cobra.training.strategies import FSDPStrategy, TrainingStrategy
from cobra.training.strategies.single_gpu import SingleGPUStrategy

# Registry =>> Maps ID --> {cls(), kwargs} :: supports FSDP and SingleGPU
TRAIN_STRATEGIES = {
    "fsdp-shard-grad-op": {"cls": FSDPStrategy, "kwargs": {"sharding_strategy": "shard-grad-op"}},
    "fsdp-full-shard": {"cls": FSDPStrategy, "kwargs": {"sharding_strategy": "full-shard"}},
    "single-gpu": {"cls": SingleGPUStrategy, "kwargs": {}},
}


def get_train_strategy(
    train_strategy: str,
    vlm: CobraVLM,
    device_id: int,
    epochs: int,
    max_steps: Optional[int],
    global_batch_size: int,
    per_device_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    enable_gradient_checkpointing: bool = True,
    enable_mixed_precision_training: bool = True,
    reduce_in_full_precision: bool = False,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> TrainingStrategy:
    
    # Auto-detect single GPU setup and override strategy if needed
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1 and train_strategy.startswith("fsdp"):
        print(f"[INFO] Detected single GPU setup, switching from '{train_strategy}' to 'single-gpu' strategy")
        train_strategy = "single-gpu"
    
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]
        strategy = strategy_cfg["cls"](
            vlm=vlm,
            device_id=device_id,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            **strategy_cfg["kwargs"],
        )
        return strategy
    else:
        raise ValueError(f"Train Strategy `{train_strategy}` is not supported!")