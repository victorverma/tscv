import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
     """Abstract base class for metrics."""

     @abstractmethod
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
         """
         Compute the value of the metric using the observed response values and their predictions.

         :param y_obs: NumPy array of shape (n,) containing the observed response values.
         :param y_pred: NumPy array of shape (n,) containing the predictions of the observed response values.
         :return: Float that equals the computed value of the metric.
         """
         pass
