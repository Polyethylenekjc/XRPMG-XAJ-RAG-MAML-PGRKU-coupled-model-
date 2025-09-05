# src/analyzer/analyzerBase.py
import os
from typing import Any, Dict, Optional
from src.utils.Logger import Logger

class AnalyzerBase:
    """Base class for analyzers, defining basic interfaces and common attributes for analyzers"""
    
    def __init__(self, config=None):
        """
        Initialize the analyzer

        Args:
            config: Analyzer configuration parameters
        """
        self.res = None  # Used to store analysis results
        self.config = config or {}  # Use empty dict as default value
        self.logger = Logger

    def analyze(self, predictions: Any, true_values: Any, **kwargs: Any) -> None:
        """
        Analyze prediction results

        Args:
            predictions: Dictionary containing model prediction results
            true_values: Ground truth values
            **kwargs: Variable keyword arguments to support special requirements of different analyzers
        """
        self.logger.info(f"Start analyzing prediction results, prediction data: {predictions.keys()}", module=self.__class__.__name__)
        try:
            self._analyze(predictions, true_values, **kwargs)
            self.logger.info("Prediction result analysis completed", module=self.__class__.__name__)
        except Exception as e:
            self.logger.error(f"Failed to analyze prediction results, error details: {str(e)}", module=self.__class__.__name__)
            raise

    def _analyze(self, predictions: Any, true_values: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Subclasses must implement the _analyze method")

    def save(self, dataset_name: str, model_name: str, path=None) -> None:
        """
        Save analysis results to specified path

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            path: Save path, if None then use default path
        """
        if path is None:
            path = self.config.get('output', {}).get('analysis_result_path', 'res/analysis/')
        
        final_path = os.path.join(path, dataset_name, model_name)
        os.makedirs(final_path, exist_ok=True)
        self.logger.info(f"Saving analysis results to: {final_path}", module=self.__class__.__name__)
        self._save_to_file(final_path, model_name)
        
    def _save_to_file(self, path: str, model_name: str) -> None:
        """
        Abstract method for actual file saving, implemented by subclasses for specific saving logic

        Args:
            path: Complete file save path
            model_name: Name of the model
        """
        raise NotImplementedError("Subclasses must implement the _save_to_file method")