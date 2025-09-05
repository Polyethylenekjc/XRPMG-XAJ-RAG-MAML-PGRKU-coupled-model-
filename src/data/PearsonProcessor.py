import pandas as pd
from ..factory.data_processor_factory import DataProcessorFactory

@DataProcessorFactory.register('pearson')
class PearsonProcessor:
    def __init__(self, n=10):
        self.n = n

    @classmethod
    def from_config(cls, config):
        n = config.get('n', 10)
        return cls(n=n)

    def process(self, data, target_column=None):
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        corr_matrix = data.corr(method='pearson')
        scores = abs(corr_matrix[target_column])
        scores = scores[scores.index != target_column]

        top_n_features = scores.sort_values(ascending=False).head(self.n-1).index.tolist()
        selected_features = [target_column] + top_n_features

        return data[selected_features]