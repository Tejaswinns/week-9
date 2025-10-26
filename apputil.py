import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
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
        return data.mean()

    def _median_estimate(self, data):
        return data.median()

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.grouped_data = X.copy()
        self.grouped_data['target'] = y
        self.grouped_data = self.grouped_data.groupby(list(X.columns))['target'].apply(self.estimate_method).reset_index()
        return None

    def predict(self, X):
        X = pd.DataFrame(X)
        predictions = X.merge(self.grouped_data, on=list(X.columns), how='left')
        missing_count = predictions['target'].isna().sum()
        if missing_count > 0:
            print(f"Number of missing groups: {missing_count}")
        return predictions['target'].tolist()    
    