import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from metric import Metric
from model import Model
from multiprocessing import get_context
from preprocessor import Preprocessor
from typing import Optional

class ParallelizationMode(Enum):
    NONE = "none"
    SCRIPT = "script"

class TimeSeriesCrossValidator:
    """Time series cross-validation framework."""

    def __init__(
            self,
            x: np.ndarray, y: np.ndarray,
            train_starts: list[int], train_size: int, test_starts: list[int], test_size: int,
            metrics: list[Metric]
        ) -> None:
        """
        Initialize the cross-validation scheme with data, training/test window creation information, and performance metrics.

        Parameters:
        :param x: NumPy array of shape (n, d) containing covariate data.
        :param y: NumPy array of shape (n,) containing outcome data.
        :param train_starts: List of start indices of the training windows.
        :param train_size: Integer giving the size of a training window.
        :param test_starts: List of start indices of the test windows.
        :param test_size: Integer giving the size of a test window.
        :param metrics: List of Metric instances for the performance metrics.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if x.shape[0] != y.size:
            raise ValueError("The number of rows in x must equal the length of y.")
        if len(train_starts) != len(test_starts):
            raise ValueError("train_starts and test_starts must have the same length.")
        if any(test_starts[k] < train_starts[k] + train_size for k in range(len(train_starts))):
            raise ValueError("Each test set must start after the corresponding training set.")
        if test_starts[-1] + test_size > x.shape[0]:
            raise ValueError("The index of the last test observation exceeds the index of the last observation.")
        self.x = x
        self.y = y
        self.window_pairs = []
        for train_start, test_start in zip(train_starts, test_starts):
            train_indices = range(train_start, train_start + train_size)
            test_indices = range(test_start, test_start + test_size)
            self.window_pairs.append((train_indices, test_indices))
        self.metrics = metrics

    def _process_window_pair(self, window_pair: tuple[range, range], preprocessor: Preprocessor, models: list[Model]) -> pd.DataFrame:
        """
        Process a single training-test window pair: preprocess the data, fit the model, make predictions, and compute metrics.

        :param window_pair: Tuple whose two entries contain the training and test window indices.
        :param preprocessor: Preprocessor instance for preprocessing data before model fitting.
        :param model: Model instance for the model to try.
        :return: Pandas data frame containing the values of the performance metrics for each model.
        """
        train_indices, test_indices = window_pair
        x_train, x_test = self.x[train_indices], self.x[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]

        x_train, y_train = preprocessor.fit_transform(x_train, y_train)
        x_test, y_test = preprocessor.transform(x_test, y_test)

        evaluations = []
        for model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            evaluation = {
                "train_start": train_indices[0], "train_end": train_indices[-1],
                "test_start": test_indices[0], "test_end": test_indices[-1],
                "model": model.label
            }
            for metric in self.metrics:
                evaluation[metric.__class__.__name__] = metric.evaluate(y_test, y_pred)
            evaluations.append(evaluation)

        return pd.DataFrame(evaluations)

    def cross_validate(
            self,
            preprocessor: Preprocessor,
            models: list[Model],
            parallelize: Optional[ParallelizationMode] = ParallelizationMode.NONE.value,
            max_workers: Optional[int] = None
        ) -> pd.DataFrame:
        """
        Perform time series cross-validation with a given preprocessor and given models.

        :param preprocessor: Preprocessor instance for preprocessing data before model fitting.
        :param models: List of Model instances for the models to try.
        :param parallelize: ParallelizationMode value indicating whether/how to process training-test window pairs in parallel.
        :param max_workers: Integer giving the maximum number of workers to use when parallelization is desired.
        :return: Pandas data frame containing the values of the performance metrics for each pair of test window and model.
        """
        if parallelize != ParallelizationMode.NONE.value and max_workers is None:
            raise ValueError("max_workers cannot be None if parallelize isn't None.")

        evaluations = []
        if parallelize == ParallelizationMode.NONE.value:
            evaluations = [self._process_window_pair(window_pair, preprocessor, models) for window_pair in self.window_pairs]
        elif parallelize == ParallelizationMode.SCRIPT.value:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("fork")) as executor:
                future_to_window_pair = {
                    executor.submit(self._process_window_pair, window_pair, preprocessor, models): window_pair for window_pair in self.window_pairs
                }
                for future in as_completed(future_to_window_pair):
                    evaluations.append(future.result())

        return pd.concat(evaluations)