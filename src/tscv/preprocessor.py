import numpy as np
from abc import ABC, abstractmethod
from typing import final

class Preprocessor(ABC):
    """Abstract base class for preprocessors."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the preprocessor to training data.

        :param x: NumPy array of shape (n, d) containing covariate training data.
        :param y: NumPy array of shape (n,) containing response training data.
        """
        pass

    @abstractmethod
    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform data with the preprocessor.

        :param x: NumPy array of shape (n, d) containing covariate data.
        :param y: NumPy array of shape (n,) containing response data.
        :return: Tuple whose two entries are the values of x and y after preprocessing.
        """
        pass

    @final
    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor to training data and then transform the data with the fitted preprocessor.

        :param x: NumPy array of shape (n, d) containing covariate data.
        :param y: NumPy array of shape (n,) containing response data.
        :return: Tuple whose two entries are the values of x and y after preprocessing.
        """
        self.fit(x, y)
        return self.transform(x, y)

    @final
    def _embed(x: np.ndarray, r: int) -> np.ndarray:
        """
        Concatenate consecutive rows of an array. This mimics the functionality of stats::embed() in R.

        :param x: NumPy array of shape (n, d) whose rows are to be concatenated.
        :param r: Integer equal to the number of consecutive rows to concatenate.
        :return: NumPy array of shape (n - r + 1, r * d) resulting from concatenating rows.
        """
        n, d = x.shape
        if n < r:
            raise ValueError("x must have at least r rows.")

        if r == 1:
            return x
        x_ = np.zeros((n - r + 1, r * d))
        for i in range(n - r + 1):
            x_[i] = x[i:(i + r)][::-1].flatten()
        return x_
