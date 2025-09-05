# src/data/dataprocessorBase.py
from abc import ABC, abstractmethod
import pandas as pd
from src.utils.Logger import Logger

class DataProcessorBase(ABC):
    def __init__(self, module_name: str):
        self.logger = Logger
        self.module_name = module_name or self.__class__.__name__

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Starting data processing for module: {self.module_name}", module=self.module_name)
        try:
            result = self._process(df)
            self.logger.info(f"Data processing completed for module: {self.module_name}", module=self.module_name)
            return result
        except Exception as e:
            self.logger.error(f"Data processing failed for module: {self.module_name}, error details: {str(e)}", module=self.module_name)
            raise

    @abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class CombinedDataProcessor(DataProcessorBase):
    def __init__(self, processors: list, module_name: str = "CombinedDataProcessor"):
        super().__init__(module_name)
        self.processors = processors

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        for proc in self.processors:
            df = proc.process(df)
        return df