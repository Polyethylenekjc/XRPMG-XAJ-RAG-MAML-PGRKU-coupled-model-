import pandas as pd

from statsmodels.tsa.stattools import grangercausalitytests
from ..factory.data_processor_factory import DataProcessorFactory


@DataProcessorFactory.register('granger')
class GrangerCausalityProcessor:
    def __init__(self, maxlag=5, limit=10):
        self.maxlag = maxlag
        self.limit = limit

    @classmethod
    def from_config(cls, config):
        maxlag = config.get('maxlag', 5)
        limit = config.get('limit', 10)
        return cls(maxlag=maxlag, limit=limit)


    def process(self, data: pd.DataFrame, target_column=None):
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        results = []
        for col in data.columns:
            if col == target_column:
                continue
            try:
                df_pair = data[[target_column, col]].dropna()
                if len(df_pair) < self.maxlag + 1:
                    print(f"Insufficient data for {col} and {target_column}")
                    continue

                # Note: grangercausalitytests requires the dependent variable first, followed by the independent variable
                test_result = grangercausalitytests(df_pair, maxlag=self.maxlag, verbose=False)

                # Extract the minimum p-value (from ssr_ftest)
                min_p_value = min(
                    test_result[lag][0]['ssr_ftest'][1]
                    for lag in test_result
                )
                results.append((col, min_p_value))
            except Exception as e:
                print(f"Error processing {col}: {e}")

        results.sort(key=lambda x: x[1])
        top_features = [col for col, _ in results[:self.limit-1]]
        return data[top_features + [target_column]]