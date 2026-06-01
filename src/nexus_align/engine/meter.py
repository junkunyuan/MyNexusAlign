"""Training meters: windowed metric tracking and hardware monitoring."""

import os
import time
import threading
from collections import deque
from dataclasses import dataclass

import torch
import psutil
import pynvml

GB = 1024**3


@dataclass
class Meter:
    """A single windowed metric: a bounded value history plus its running mean.

    precision/notation control display: notation "e" prints scientific form
    (precision=2 -> 2.34e-02), "f" prints fixed-point (precision=4 -> 0.0234).
    """

    window_size: int
    precision: int = 2
    notation: str = "e"
    report_mean: bool = True

    def __post_init__(self) -> None:
        self.data: deque = deque(maxlen=self.window_size)
        self.mean: float | None = None

    @property
    def latest(self) -> float | None:
        return self.data[-1] if self.data else None

    def update(self, value: int | float) -> None:
        """Append a value and refresh the window mean."""
        self.data.append(value)
        self.mean = sum(self.data) / len(self.data)

    def update_peak(self, value: int | float) -> None:
        """Append the running maximum (windows of size 1 keep a single peak)."""
        self.update(value if not self.data else max(self.data[-1], value))

    def reset(self) -> None:
        """Clear the history while keeping the window size."""
        self.data.clear()
        self.mean = None


class WindowMeter:
    """A timer and tracker for training information over a sliding window."""

    def __init__(self, hardware: bool = True) -> None:
        self.meters: dict[str, Meter] = {}
        self.hardware_meters: set[str] = set()
        self.timing_meters = {"epoch", "step"}

        # Cumulative counters (plain values, not windowed meters).
        self.epoch = 0
        self.step = 0
        self.total_step = 0
        self.current_train_steps = 0  # steps in this run
        self.exp_start_time = None
        self._epoch_start = None
        self._step_start = None

        # Hardware monitoring (deps lazily imported; only set up when requested).
        self._hardware_monitoring = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._pynvml = None
        self._psutil = None
        self.device_id = None
        self.nvml_handle = None
        if hardware:
            self._setup_hardware()

    # ----------------------------------------
    # Meter Utilities
    # ----------------------------------------
    def add_new_meter(
        self,
        meter: str,
        window_size: int,
        precision: int = 2,
        notation: str = "e",
        report_mean: bool = True,
    ) -> None:
        """Register a new windowed meter.

        ``precision`` is the number of mantissa digits in scientific notation
        (``notation``="e", the default for experiment metrics) or the decimal
        places in fixed-point (``notation``="f").
        """
        self.meters[meter] = Meter(window_size, precision, notation, report_mean)

    def update(self, meter: str, value: int | float) -> None:
        """Append a value to a meter. Only int and float are accepted."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"❌ Expected int or float, got {type(value).__name__}")
        self.meters[meter].update(value)

    def update_peak(self, meter: str, value: int | float) -> None:
        """Append the running maximum of a meter."""
        self.meters[meter].update_peak(value)

    def _experiment_meters(self):
        """Yield (name, meter) for user metrics, excluding hardware/timing meters."""
        for name, meter in self.meters.items():
            if name not in self.hardware_meters and name not in self.timing_meters:
                yield name, meter

    def latest_metrics(self) -> dict:
        """Return the latest value of each experiment metric (0 if empty)."""
        return {
            name: (meter.latest if meter.latest is not None else "NaN")
            for name, meter in self._experiment_meters()
        }

    # ----------------------------------------
    # Hardware Monitoring
    # ----------------------------------------
    def _setup_hardware(self) -> None:
        """Resolve the NVML handle, and start sampling."""
        self._psutil = psutil
        self._pynvml = pynvml
        psutil.cpu_percent(interval=None)

        self.device_id = torch.cuda.current_device()
        pynvml.nvmlInit()
        physical_idx = self.device_id
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            physical_idx = [int(x) for x in visible.split(",")][self.device_id]
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

        self.hardware_meters = self._init_hardware_meters()
        self.start_hardware_monitoring()

    def _init_hardware_meters(self) -> set[str]:
        """Register GPU/CPU memory and utilization meters; return their names."""
        log_window = 3600  # seconds of history kept for non-peak meters
        names = [
            "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total",  # GB
            "gpu_util", "gpu_util_peak",                      # %
            "cpu_mem_used", "cpu_mem_peak", "cpu_mem_total",  # GB
            "cpu_util", "cpu_util_peak",                      # %
        ]
        for name in names:
            precision = 0 if "util" in name else 1
            window = 1 if ("peak" in name or "total" in name) else log_window
            self.add_new_meter(name, window_size=window, precision=precision, notation="f")

        self.update("gpu_mem_total", self._pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).total / GB)
        self.update("cpu_mem_total", self._psutil.virtual_memory().total / GB)
        return set(names)

    def _monitor_hardware(self) -> None:
        """Sample GPU/CPU usage."""
        nvml, ps = self._pynvml, self._psutil
        while not self._stop_monitoring.is_set():
            gpu_mem = nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / GB
            self.update("gpu_mem_used", gpu_mem)
            self.update_peak("gpu_mem_peak", gpu_mem)

            gpu_util = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            if gpu_util is not None:
                self.update("gpu_util", gpu_util)
                self.update_peak("gpu_util_peak", gpu_util)

            cpu_mem = ps.virtual_memory().used / GB
            self.update("cpu_mem_used", cpu_mem)
            self.update_peak("cpu_mem_peak", cpu_mem)

            cpu_util = ps.cpu_percent(interval=None)
            self.update("cpu_util", cpu_util)
            self.update_peak("cpu_util_peak", cpu_util)

            if self._stop_monitoring.wait(1):  # sleep 1s, exit early if stopped
                break

    def start_hardware_monitoring(self) -> None:
        """Start hardware monitoring in a background thread."""
        if not self._hardware_monitoring:
            self._hardware_monitoring = True
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_hardware, daemon=True
            )
            self._monitor_thread.start()

    def stop_hardware_monitoring(self) -> None:
        """Stop the monitoring thread and release NVML."""
        if self._hardware_monitoring:
            self._stop_monitoring.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1)
            self._hardware_monitoring = False
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()

    # ----------------------------------------
    # Epoch and Step Meters
    # ----------------------------------------
    def add_epoch_step(self, epoch_window: int = 5, step_window: int = 100) -> None:
        """Register the epoch and step timing meters."""
        self.add_new_meter("epoch", window_size=epoch_window, precision=1, notation="f")
        self.add_new_meter("step", window_size=step_window, precision=1, notation="f")

    def update_train_state(self, train_state: dict) -> None:
        """Restore the epoch/step/total_step counters (e.g. after resuming)."""
        self.epoch = train_state.get("epoch", 0)
        self.step = train_state.get("step", 0)
        self.total_step = train_state.get("total_step", 0)

    def start(self, meter: str) -> None:
        """Call it at the beginning of every epoch/step."""
        if meter not in self.timing_meters:
            raise ValueError("❌ Meter must be 'step' or 'epoch'")
        now = time.time()
        if meter == "epoch":
            self._epoch_start = now
            if self.exp_start_time is None:
                self.exp_start_time = now
        else:
            self._step_start = now

    def end(self, meter: str) -> None:
        """Call it at the end of every epoch/step."""
        if meter not in self.timing_meters:
            raise ValueError("❌ meter must be 'step' or 'epoch'")
        if meter == "epoch":
            self.epoch += 1
            self.update("epoch", time.time() - self._epoch_start)
            self.step = 0
            self.meters["step"].reset()
        else:
            self.step += 1
            self.total_step += 1
            self.current_train_steps += 1
            self.update("step", time.time() - self._step_start)

    # ----------------------------------------
    # Print Meter Information
    # ----------------------------------------
    def _fmt(self, meter: str, which: str = "latest", unit: str = "") -> str:
        """Format a meter's latest value or mean using its precision/notation."""
        m = self.meters[meter]
        val = m.latest if which == "latest" else m.mean
        if not isinstance(val, (int, float)):
            return "N/A"
        return f"{val:.{m.precision}{m.notation}}{unit}"

    def _hardware_info(self) -> list[str]:
        """One summary line per hardware meter group (used/mean/peak[/total])."""
        # label, used, peak, total (None if absent), unit
        specs = [
            ("gpu_mem", "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total", "G"),
            ("gpu_uti", "gpu_util", "gpu_util_peak", None, "%"),
            ("cpu_mem", "cpu_mem_used", "cpu_mem_peak", "cpu_mem_total", "G"),
            ("cpu_uti", "cpu_util", "cpu_util_peak", None, "%"),
        ]
        lines = []
        for label, used, peak, total, unit in specs:
            line = (
                f"{label}: {self._fmt(used, 'latest', unit)} "
                f"(mean: {self._fmt(used, 'mean', unit)}, "
                f"peak: {self._fmt(peak, 'latest', unit)}"
            )
            if total is not None:
                line += f", total: {self._fmt(total, 'latest', unit)}"
            lines.append(line + ")")
        return lines

    def info(self, train_info: bool = True, show_metrics: bool = True) -> str:
        """Build a human-readable summary string of the meters."""
        out = ""

        if train_info:
            infos = [
                f"epoch: {self.epoch}",
                f"step: {self.step}",
                f"total_step: {self.total_step}",
            ]
            for key in ("epoch", "step"):
                infos.append(f"{key}_avg_time_cost: {self._fmt(key, 'mean', 's')}")
            if self._hardware_monitoring:
                infos += self._hardware_info()
            out += "\n📊 " + "  |  ".join(infos)

        if show_metrics:
            infos = []
            for name, meter in self._experiment_meters():
                latest = self._fmt(name, "latest")
                if meter.report_mean:
                    infos.append(f"{name}: {latest} ({self._fmt(name, 'mean')})")
                else:
                    infos.append(f"{name}: {latest}")
            out += "\n📊 " + "  |  ".join(infos)

        return out
