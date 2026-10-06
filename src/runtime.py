"""Устройство, тип весов, сид и память — всё, что зависит от железа.

Код обучения не знает, на чём он запущен: устройство выбирается здесь по
params.yaml. Смена железа — правка конфига, а не кода.
"""

import os
import random
import sys

import numpy as np
import psutil
import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(name: str) -> torch.device:
    """auto -> cuda, если есть; иначе mps; иначе cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise SystemExit(f"model.dtype = {name!r}, допустимо: {sorted(DTYPES)}")
    return DTYPES[name]


def set_seed(seed: int) -> None:
    """Один сид на всё, что тянет случайность: инициализацию A в LoRA,
    dropout, порядок примеров. Без него два прогона с одним конфигом —
    два разных эксперимента, и сравнивать их нельзя."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def allocated_bytes(device: torch.device) -> int:
    """Аллокатор ускорителя; на CPU — пик RSS процесса, фиксируемый ОС."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated()
    if os.name == "posix":
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak_rss if sys.platform == "darwin" else peak_rss * 1024
    return psutil.Process().memory_info().rss


def memory_metric(device: torch.device) -> str:
    return {
        "mps": "torch.mps.driver_allocated_memory",
        "cuda": "torch.cuda.memory_allocated",
    }.get(device.type, "ru_maxrss (пик RSS процесса)" if os.name == "posix" else "RSS процесса")
