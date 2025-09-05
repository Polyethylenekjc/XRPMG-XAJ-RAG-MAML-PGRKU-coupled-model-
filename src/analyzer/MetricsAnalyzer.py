# src/analyzer/MetricsAnalyzer.py
from src.analyzer.analyzerBase import AnalyzerBase
import pandas as pd
from typing import Any, Dict, Optional
import numpy as np
from src.factory.analyzer_factory import AnalyzerFactory
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import os

@AnalyzerFactory.register('metrics_analyzer')
class MetricsAnalyzer(AnalyzerBase):
    def __init__(self, config=None):
        """
        Initializes the analyzer.

        Args:
            config: Configuration parameters for the analyzer.
        """
        super().__init__(config)
        self.res = None

    def analyze(self, predictions: Any, true_values: Any, **kwargs):
        """
        Analyzes the model's predictions and true values, calculating various metrics.

        Args:
            predictions: A dictionary containing the model's prediction results.
            true_values: True values (supports list or numpy array).
            **kwargs: Variable keyword arguments to support specific requirements
                      for different analyzers.
        """
        # Convert to numpy array for uniformity
        true_values_array = np.squeeze(np.array(true_values))

        # Validate that predictions is a non-empty dictionary
        if not predictions or not isinstance(predictions, dict):
            raise ValueError("predictions must be a non-empty dictionary")

        # Initialize the result storage structure
        self.res = {}

        for model_name, preds in predictions.items():
            preds_array = np.squeeze(np.array(preds))

            # Calculate basic metrics
            mae = mean_absolute_error(true_values_array, preds_array)
            
            # Add epsilon to prevent division by zero in MAPE
            epsilon = 1e-8
            mape = mean_absolute_percentage_error(true_values_array + epsilon, preds_array + epsilon)

            # Calculate NSE (Nash-Sutcliffe Efficiency)
            mean_true = np.mean(true_values_array)
            numerator = np.sum((true_values_array - preds_array) ** 2)
            denominator = np.sum((true_values_array - mean_true) ** 2)
            nse = 1 - (numerator / denominator) if denominator != 0 else 0

            # Calculate R2 (Coefficient of Determination)
            r2 = r2_score(true_values_array, preds_array)

            # Calculate metrics required for Taylor diagram
            # Correlation coefficient
            correlation_matrix = np.corrcoef(true_values_array, preds_array)
            correlation = correlation_matrix[0, 1]
            
            # Standard deviation of predictions and true values
            std_dev_pred = np.std(preds_array)
            std_dev_true = np.std(true_values_array)
            
            # Calculate KGE (Kling-Gupta Efficiency)
            alpha = std_dev_pred / std_dev_true if std_dev_true != 0 else 0
            beta = np.mean(preds_array) / np.mean(true_values_array) if np.mean(true_values_array) != 0 else 0
            kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

            # Store results
            self.res = {
                'MAE': mae,
                'MAPE': mape,
                'NSE': nse,
                'KGE': kge,
                'R2': r2,
                'Correlation': correlation,
                'StdDev': std_dev_pred,
                'RefStdDev': std_dev_true
            }

    def _save_to_file(self, path: str, model_name: str) -> None:
        """
        Abstract method for saving the file, to be implemented by subclasses.

        Args:
            path: The complete file save path.
            model_name: The name of the model.
        """
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        if not self.res:
            raise ValueError("Results not initialized. Cannot save results.")
            
        # Generate a metrics file for a single model, formatted to be compatible with SummaryAnalyzer
        model_metrics_path = os.path.join(path, f'{model_name}_metrics.csv')
        result_dict = self.res
        if result_dict:
            # Add 'Model' and 'Location' fields to match the format expected by SummaryAnalyzer
            row = {'Model': model_name, 'Location': os.path.basename(path)}
            row.update(result_dict)
            df = pd.DataFrame([row])
            df.to_csv(model_metrics_path, index=False)