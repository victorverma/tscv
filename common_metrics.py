import numpy as np
from metric import Metric
from sklearn.metrics import confusion_matrix

# Regression Metrics ###########################################################

class MSE(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          return np.mean((y_obs - y_pred) ** 2)

class MAE(Metric):
     def evaluate(self, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
          return np.mean(np.abs(y_obs - y_pred))

# Classification Metrics #######################################################

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
