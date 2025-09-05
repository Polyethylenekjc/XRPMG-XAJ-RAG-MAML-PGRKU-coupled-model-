import pandas as pd
from .dataprocessorBase import DataProcessorBase
from src.factory.data_processor_factory import DataProcessorFactory

@DataProcessorFactory.register("limit_processor")
class LimitProcessor:
    def __init__(self, limit=365):
        self.limit = limit

    @classmethod
    def from_config(cls, config):
        limit = config.get("limit", 365)
        return cls(limit)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns the first 'limit' rows of the DataFrame
        return df.head(self.limit)