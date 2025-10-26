"""Utility helpers for week-9 exercises.

Provides GroupEstimate class for computing group-level estimates (mean or median)
and predicting them for new data.
"""

import pandas as pd
import numpy as np


class GroupEstimate:
    """Estimate group-level target values and predict by joining groups.

    Parameters
    ----------
    estimate : {'mean', 'median'}
        Aggregation type for each group.

    Attributes
    ----------
    grouped_data : pd.DataFrame or None
        Grouped data with aggregated targets.
    columns_ : list or None
        Grouping column names.
    """

    def __init__(self, estimate):
        # validate estimate choice
        if estimate not in ["mean", "median"]:
            raise ValueError(f"Estimate must be 'mean' or 'median', got '{estimate}'")
        self.estimate = estimate
        self.grouped_data = None
        self.columns_ = None

    def fit(self, X, y):
        """Compute grouped estimates from X and y.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature table for grouping.
        y : array-like or pd.Series
            Target values to aggregate.
        """
        # Ensure inputs are DataFrame/Series for consistent operations
        X = pd.DataFrame(X)
        y = pd.Series(y)

        # Basic sanity check: rows must match
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")

        # Remember the columns used for grouping so predict can use them
        self.columns_ = list(X.columns)

        # Create a combined DataFrame so we can group by all feature columns
        df = X.copy()
        df["target"] = y

        # Group and aggregate using the requested estimator
        if self.estimate == "mean":
            self.grouped_data = (
                df.groupby(self.columns_)["target"].mean().reset_index()
            )
        else:
            self.grouped_data = (
                df.groupby(self.columns_)["target"].median().reset_index()
            )

    def predict(self, X):

        """Predict group-level estimates for X."""

        # Ensure input is a DataFrame with correct columns
        X = pd.DataFrame(X, columns=self.columns_) if isinstance(X, np.ndarray) else pd.DataFrame(X)
        X.columns = self.columns_  # enforce column names

        # Merge with grouped data to get predictions
        preds = X.merge(self.grouped_data, on=self.columns_, how="left")

        missing_count = preds["target"].isna().sum()
        if missing_count > 0:
         print(f"Number of missing groups: {missing_count}")

        return preds["target"].to_numpy()
