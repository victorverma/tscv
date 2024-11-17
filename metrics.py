import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix

class Metric(ABC):
     """Abstract base class for metrics."""
     @abstractmethod
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
         """
         Compute the value of the metric using the test response values and their predictions.

         :param y_obs: NumPy array of shape (n,) containing the test response values.
         :param y_pred: NumPy array of shape (n,) containing the predictions of the test response values.
         :return: Float that equals the computed value of the metric.
         """
         pass

class MSE(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          return np.mean((y_obs - y_pred) ** 2)

class MAE(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          return np.mean(np.abs(y_obs - y_pred))

class EventRate(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          tn, fp, fn, tp = confusion_matrix(y_obs, y_pred).ravel()
          return (tp + fn) / (tp + fn + fp + tn)

class AlarmRate(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          tn, fp, fn, tp = confusion_matrix(y_obs, y_pred).ravel()
          return (tp + fp) / (tp + fn + fp + tn)

class TPR(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          _, _, fn, tp = confusion_matrix(y_obs, y_pred).ravel()
          return tp / (tp + fn)

class FPR(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          tn, fp, _, _ = confusion_matrix(y_obs, y_pred).ravel()
          return fp / (fp + tn)

class TSS(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          tn, fp, fn, tp = confusion_matrix(y_obs, y_pred).ravel()
          return tp / (tp + fn) - fp / (fp + tn)

class Precision(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          _, fp, _, tp = confusion_matrix(y_obs, y_pred).ravel()
          return tp / (tp + fp)
