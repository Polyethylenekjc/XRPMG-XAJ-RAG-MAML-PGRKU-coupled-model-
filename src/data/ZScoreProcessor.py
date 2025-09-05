import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..factory.data_processor_factory import DataProcessorFactory

@DataProcessorFactory.register('zscore')
class ZScoreProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    @classmethod
    def from_config(cls, config):
        return cls()

    def process(self, data, target_column=None):
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)