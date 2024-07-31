from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

class TimeSeriesCrossValidator:
    """Time series cross-validation framework."""
    def __init__(self, train_starts, train_sizes, test_sizes, preprocessor=None, max_workers=1):
        if not (len(train_starts) == len(train_sizes) == len(test_sizes)):
            raise ValueError("train_starts, train_sizes, and test_sizes must have the same length.")
        if preprocessor is not None:
            if not (hasattr(preprocessor, "fit_transform") and callable(getattr(preprocessor, "fit_transform")) and
                    hasattr(preprocessor, "transform") and callable(getattr(preprocessor, "transform"))):
                raise TypeError("preprocessor must have 'fit_transform' and 'transform' methods.")
        self.train_starts = train_starts
        self.train_sizes = train_sizes
        self.test_sizes = test_sizes
        self.preprocessor = preprocessor
        self.max_workers = max_workers

    def split(self, x, y):
        """Generate indices for splitting data into training and test sets."""
        splits = []
        for train_start, train_size, test_size in zip(self.train_starts, self.train_sizes, self.test_sizes):
            if train_start + train_size + test_size > len(x):
                raise ValueError("Train and test indices exceed dataset length.")
            train_indices = range(train_start, train_start + train_size)
            test_indices = range(train_start + train_size, train_start + train_size + test_size)
            splits.append((train_indices, test_indices))
        return splits

    def evaluate(self, y_test, y_pred, metrics):
        """Evaluate predictions from the fitted model using specified metrics."""
        evaluation = {}
        for metric in metrics:
            evaluation[metric.__class__.__name__] = metric.evaluate(y_test, y_pred)
        return evaluation

    def _process_split(self, split, x, y, model, metrics):
        """Process a single split: fit the model, make predictions, and evaluate."""
        train_indices, test_indices = split
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        if self.preprocessor:
            x_train = self.preprocessor.fit_transform(x_train)
            x_test = self.preprocessor.transform(x_test)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return self.evaluate(y_test, y_pred, metrics)

    def cross_validate(self, model, x, y, metrics):
        """Fit the model, make predictions, and calculate metrics."""
        if not (hasattr(model, "fit") and callable(getattr(model, "fit")) and
                hasattr(model, "predict") and callable(getattr(model, "predict"))):
            raise TypeError("model must have 'fit' and 'predict' methods.")
        if not metrics:
            raise ValueError("at least one metric must be provided.")
        for index, metric in enumerate(metrics):
            if not (hasattr(metric, "evaluate") and callable(getattr(metric, "evaluate"))):
                raise TypeError(f"metric {index} must have an 'evaluate' method.")
        splits = self.split(x, y)
        evaluations = []
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=get_context("fork")) as executor:
            future_to_split = {executor.submit(self._process_split, split, x, y, model, metrics): split for split in splits}
            for future in as_completed(future_to_split):
                evaluation = future.result()
                evaluations.append(evaluation)
        return evaluations