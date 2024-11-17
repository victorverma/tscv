import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for models."""

    @property
    @abstractmethod
    def label(self) -> str:
        """The label of the model."""
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass
