import pandas as pd
from ..factory.data_processor_factory import DataProcessorFactory

@DataProcessorFactory.register('spearman')
class SpearmanProcessor:
    def __init__(self, n=10, target_column=None):
        self.n = n
        self.target_column = target_column

    @classmethod
    def from_config(cls, config):
        n = config.get('n', 10)
        target_column = config.get('target_column')
        return cls(n=n, target_column=target_column)

    def process(self, data):
        # Ensure the target column exists
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")

        # Calculate Spearman's rank correlation coefficient
        corr_matrix = data.corr(method='spearman')
        scores = abs(corr_matrix[self.target_column])
        scores = scores[scores.index != self.target_column]  

        # Get the column names of the top N most correlated features
        top_n_features = scores.sort_values(ascending=False).head(self.n-1).index.tolist()

        # Include the target column and return
        selected_features = [self.target_column] + top_n_features
        print(selected_features)

        # Return a new DataFrame with only the selected columns, maintaining the original format
        return data[selected_features]