class TimeSeriesCrossValidator:
    """Time series cross-validation framework."""
    def __init__(self, train_starts, train_sizes, test_sizes, preprocessor=None):
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

    def split(self, X, y):
        """Generate indices for splitting data into training and test sets."""
        splits = []
        for train_start, train_size, test_size in zip(self.train_starts, self.train_sizes, self.test_sizes):
            if train_start + train_size + test_size > len(X):
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

    def cross_validate(self, model, X, y, metrics):
        """Fit the model, make predictions, and calculate metrics."""
        if not (hasattr(model, "fit") and callable(getattr(model, "fit")) and
                hasattr(model, "predict") and callable(getattr(model, "predict"))):
            raise TypeError("model must have 'fit' and 'predict' methods.")
        if not metrics:
            raise ValueError("at least one metric must be provided.")
        for index, metric in enumerate(metrics):
            if not (hasattr(metric, "evaluate") and callable(getattr(metric, "evaluate"))):
                raise TypeError(f"metric {index} must have an 'evaluate' method.")
        evaluations = []
        for train_indices, test_indices in self.split(X, y):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            if self.preprocessor:
                X_train = self.preprocessor.fit_transform(X_train)
                X_test = self.preprocessor.transform(X_test)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluation = self.evaluate(y_test, y_pred, metrics)
            evaluations.append(evaluation)        
        return evaluations