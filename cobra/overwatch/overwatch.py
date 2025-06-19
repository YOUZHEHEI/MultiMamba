"""
Fixed overwatch.py for proper single GPU support
"""
import logging
import logging.config
import os
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

# Overwatch Default Format String
RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {**{0: "[*] "}, **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]}}

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that wraps logging & `accelerate.PartialState`."""
        # Initialize logger first
        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        
        # Try to initialize accelerate PartialState with proper error handling
        try:
            from accelerate import PartialState
            self.distributed_state = PartialState()
        except Exception as e:
            # If accelerate initialization fails, fall back to PureOverwatch behavior
            self.logger.warning(f"Could not initialize accelerate PartialState: {e}")
            self.logger.info("Falling back to single GPU mode")
            self.distributed_state = None

        # Logger Delegation (for convenience)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        if self.distributed_state is not None:
            self.logger.setLevel(logging.INFO if self.distributed_state.is_main_process else logging.ERROR)
        else:
            self.logger.setLevel(logging.INFO)

    def rank_zero_only(self) -> Callable[..., Any]:
        if self.distributed_state is not None:
            return self.distributed_state.on_main_process
        else:
            # Return identity function for single GPU
            def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
                return fn
            return identity

    def is_rank_zero(self) -> bool:
        if self.distributed_state is not None:
            return self.distributed_state.is_main_process
        else:
            return True

    def rank(self) -> int:
        if self.distributed_state is not None:
            return self.distributed_state.process_index
        else:
            return 0

    def world_size(self) -> int:
        if self.distributed_state is not None:
            return self.distributed_state.num_processes
        else:
            return 1


class PureOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that just wraps logging."""
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def rank_zero_only() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    """Initialize overwatch with better single GPU detection."""
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    # Check if we have the minimum required environment variables for distributed training
    has_distributed_env = all(
        env_var in os.environ 
        for env_var in ["WORLD_SIZE", "RANK"]
    )
    
    # If WORLD_SIZE is 1 or we don't have distributed environment, use PureOverwatch
    if world_size == 1 or not has_distributed_env:
        return PureOverwatch(name)
    
    # For multi-GPU setups, try DistributedOverwatch but fall back to PureOverwatch if it fails
    try:
        return DistributedOverwatch(name)
    except Exception as e:
        print(f"Warning: Could not initialize distributed training: {e}")
        print("Falling back to single GPU mode")
        return PureOverwatch(name)