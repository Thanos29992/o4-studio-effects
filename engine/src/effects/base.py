from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models.model_manager import ModelManager


class BaseEffect(ABC):
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        ...

    def cleanup(self) -> None:
        pass
