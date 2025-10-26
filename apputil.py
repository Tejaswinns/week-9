import pandas as pd
import numpy as np


class GroupEstimate(object):
    """
    A class for estimating values based on categorical groups.

    This class fits a model on categorical data and predicts estimates
    (mean or median) for new observations based on group membership.
    Supports handling missing groups with a default category fallback.
    """

    def __init__(self, estimate):
        """
        Initialize the GroupEstimate model.

        Parameters:
        estimate (str): The estimation method, either 'mean' or 'median'.
        """
        self.estimate = estimate
        self.grouped_data = None
        estimate_methods = {
            'mean': self._mean_estimate,
            'median': self._median_estimate
        }
        if estimate not in estimate_methods:
            raise ValueError(f"Estimate method '{estimate}' not recognized.")
        self.estimate_method = estimate_methods[estimate]

    def _mean_estimate(self, data):
        """Calculate the mean of the data."""
        return data.mean()

    def _median_estimate(self, data):
        """Calculate the median of the data."""
        return data.median()

    def fit(self, X, y, default_category=None):
        """
        Fit the model on training data.

        Parameters:
        X (array-like or DataFrame): Categorical features.
        y (array-like): Target values.
        default_category (str, optional): Column name for default grouping
                                          in case of missing combinations.
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.grouped_data = X.copy()
        self.grouped_data['target'] = y
        # Group by all columns and compute estimate per group
        self.grouped_data = self.grouped_data.groupby(list(X.columns))['target'].apply(self.estimate_method).reset_index()
        self.default_category = default_category
        if default_category is not None:
            # Compute estimates per default category for fallback
            self.default_grouped = self.grouped_data.groupby(default_category)['target'].apply(self.estimate_method).reset_index()
        return None

    def predict(self, X):
        """
        Predict estimates for new data.

        Parameters:
        X (array-like or DataFrame): Observations to predict.

        Returns:
        np.ndarray: Predicted estimates, with NaN for missing groups.
        """
        X = pd.DataFrame(X)
        # Merge with grouped data to get predictions
        predictions = X.merge(self.grouped_data, on=list(X.columns), how='left')
        if self.default_category is not None:
            # Fill missing predictions with default category estimates
            missing_mask = predictions['target'].isna()
            if missing_mask.any():
                default_merge = X.loc[missing_mask, [self.default_category]].merge(self.default_grouped, on=self.default_category, how='left')
                predictions.loc[missing_mask, 'target'] = default_merge['target']
        missing_count = predictions['target'].isna().sum()
        if missing_count > 0:
            print(f"Number of missing groups: {missing_count}")
        return np.array(predictions['target'].tolist())    
    