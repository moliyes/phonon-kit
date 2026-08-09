from __future__ import annotations

from typing import Protocol

import numpy as np


class ForceProvider(Protocol):
    """Small extension point for force calculators used by the runner."""

    def prepare(self) -> None: ...

    def run_or_submit(self, *, wait: bool = False) -> str: ...

    def collect(self) -> tuple[np.ndarray, np.ndarray | None]: ...

