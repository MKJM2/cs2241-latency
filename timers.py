import time
import torch
from typing import Optional
from contextlib import ContextDecorator

NS_PER_MS = 1_000_000.0
NS_PER_S = 1_000_000_000.0

# --- Timing Utilities ---


class TimerError(Exception):
    """Custom exception for timer errors."""

    pass


class CudaTimer(ContextDecorator):
    """
    Context manager for measuring GPU execution time using CUDA events.
    Requires PyTorch. Provides time in nanoseconds.
    """

    def __init__(self, device: "torch.device"):
        if not torch.cuda.is_available():
            raise TimerError("CUDA is not available or PyTorch not installed.")
        if device.type != "cuda":
            raise TimerError(f"CudaTimer requires a CUDA device, got {device.type}")
        self.device = device
        self.start_event: Optional["torch.cuda.Event"] = None
        self.end_event: Optional["torch.cuda.Event"] = None
        self.elapsed_ns: Optional[float] = None

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        # Ensure previous CUDA operations are done? Consider if needed.
        # torch.cuda.synchronize(self.device)
        self.start_event.record(stream=torch.cuda.current_stream(self.device))
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.start_event is None or self.end_event is None:
            raise TimerError("Timer not started properly")  # Defensive
        self.end_event.record(stream=torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)  # Crucial: Wait for GPU to finish
        # elapsed_time returns milliseconds
        elapsed_ms = self.start_event.elapsed_time(self.end_event)
        self.elapsed_ns = elapsed_ms * NS_PER_MS

    def get_elapsed_ns(self) -> int:
        if self.elapsed_ns is None:
            raise TimerError("Timer was not used correctly or exited with error.")
        return int(self.elapsed_ns)


class CpuTimer(ContextDecorator):
    """Context manager for measuring CPU execution time using time.perf_counter_ns."""

    def __init__(self):
        self.start_ns: Optional[int] = None
        self.end_ns: Optional[int] = None
        self.elapsed_ns: Optional[int] = None

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.end_ns = time.perf_counter_ns()
        if self.start_ns is not None:
            self.elapsed_ns = self.end_ns - self.start_ns

    def get_elapsed_ns(self) -> int:
        if self.elapsed_ns is None:
            raise TimerError("Timer was not used correctly or exited with error.")
        return self.elapsed_ns
