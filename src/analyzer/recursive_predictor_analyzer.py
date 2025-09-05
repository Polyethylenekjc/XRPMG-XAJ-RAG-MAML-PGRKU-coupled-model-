import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Any, Dict
from src.analyzer.analyzerBase import AnalyzerBase
from src.factory.analyzer_factory import AnalyzerFactory
from src.utils.Logger import Logger


@AnalyzerFactory.register('recursive_predictor_analyzer')
class RecursivePredictorAnalyzer(AnalyzerBase):
    """
    Recursive Prediction Analyzer
    
    This analyzer performs recursive prediction by using the model's output
    as input for subsequent steps:
    1. Uses the first `window_size` time steps as initial input.
    2. Predicts the next time step's value.
    3. The latest `window_size - 1` time steps and the new prediction are used to form a new input window.
    4. This process is repeated until the predicted length matches the true values.
    5. Calculates evaluation metrics and visualizes the prediction results.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config)
        self.logger = Logger()
        self.window_size = kwargs.get("window_size", (config or {}).get("window_size", 7))
        self.warmup_ratio = kwargs.get("warmup_ratio", (config or {}).get("warmup_ratio", 0.5))

    def _analyze(self, predictions: Dict[str, Any], true_values: Any, model=None, X_test=None, **kwargs) -> None:
        """
        Executes the recursive prediction analysis.
        
        Args:
            predictions: Model prediction results.
            true_values: The true values.
            model: The model instance.
            X_test: The test data.
        """
        if model is None or X_test is None:
            raise ValueError("Recursive prediction analysis requires a model and test data.")

        self.logger.info("Starting recursive prediction analysis", module=self.__class__.__name__)
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        if hasattr(true_values, 'values'):
            true_values = true_values.values
            
        if len(true_values.shape) > 1:
            true_values = true_values.flatten()
            
        self.window_size = self._infer_window_size(X_test, model)
            
        prediction_length = len(true_values)
        
        multi_step = self._infer_multi_step_from_predictions(predictions)
        print(f"Multi-step prediction steps: {multi_step}")
        
        warmup_length = int(prediction_length * self.warmup_ratio)
        
        recursive_predictions = self._recursive_predict_with_warmup(model, X_test, prediction_length, warmup_length, multi_step)
        
        metrics = self._calculate_metrics(recursive_predictions, true_values[:len(recursive_predictions)])
        
        self.res = {
            "recursive_predictions": recursive_predictions,
            "true_values": true_values[:len(recursive_predictions)],
            "metrics": metrics
        }
        
        self.logger.info(f"Recursive prediction analysis complete, prediction length: {len(recursive_predictions)}", module=self.__class__.__name__)

    def _infer_window_size(self, X_test, model=None):
        """
        Automatically infers the window size.
        
        Args:
            X_test: The test data.
            model: The model instance.
            
        Returns:
            int: The inferred window_size.
        """
        if X_test is not None and len(X_test.shape) >= 2:
            if X_test.shape[1] < X_test.shape[2] if len(X_test.shape) > 2 else True:
                return X_test.shape[1]
            elif len(X_test.shape) > 2:
                return X_test.shape[2]
        
        if model is not None:
            if hasattr(model, 'seq_length'):
                return model.seq_length
            elif hasattr(model, 'input_size'):
                return model.input_size
            elif hasattr(model, 'config') and 'seq_length' in model.config:
                return model.config['seq_length']
        
        return self.window_size

    def _infer_multi_step_from_predictions(self, predictions):
        """
        Automatically infers the number of multi-step prediction steps from the prediction results.
        
        Args:
            predictions: Dictionary of model prediction results.
            
        Returns:
            int: The number of multi-step prediction steps.
        """
        if isinstance(predictions, dict) and len(predictions) > 0:
            first_prediction = next(iter(predictions.values()))
            
            if not isinstance(first_prediction, np.ndarray):
                first_prediction = np.array(first_prediction)
            
            if len(first_prediction) == 0:
                return 1
                
            if len(first_prediction.shape) >= 2 and first_prediction.shape[0] > 0:
                return first_prediction.shape[1]
        
        return 1

    def _recursive_predict_with_warmup(self, model, X_test, prediction_length, warmup_length, multi_step=1):
        """
        Performs recursive prediction with a warmup phase.
        
        Args:
            model: The model instance.
            X_test: The test data.
            prediction_length: The total prediction length.
            warmup_length: The length of the warmup data.
            multi_step: The number of multi-step prediction steps.
            
        Returns:
            list: The recursive prediction results.
        """
        model.eval()
        
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        
        predictions = []
        
        # Phase 1: Warmup prediction using real data
        for i in range(min(warmup_length, prediction_length)):
            if i + self.window_size > len(X_test):
                break
                
            current_input = X_test[i:i+self.window_size].copy()
            
            with torch.no_grad():
                if torch.is_tensor(current_input):
                    input_tensor = current_input
                else:
                    input_tensor = torch.from_numpy(current_input).float()
                
                input_tensor = input_tensor.to(device)
                
                # During warmup, use single-step prediction for accuracy
                prediction = model(input_tensor)
                
                if hasattr(prediction, 'cpu'):
                    prediction = prediction.cpu().numpy()
                
                if len(prediction.shape) > 1:
                    prediction = prediction.flatten()[0]
                else:
                    prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0 else prediction
                    
                predictions.append(prediction)
        
        # Phase 2: Recursive prediction using predicted data
        if warmup_length < prediction_length:
            if warmup_length + self.window_size > len(X_test):
                return np.array(predictions)
                
            current_input = X_test[warmup_length:warmup_length+self.window_size].copy()
            
            i = warmup_length
            while i < prediction_length:
                if current_input is None or len(current_input) == 0:
                    break
                    
                with torch.no_grad():
                    if torch.is_tensor(current_input):
                        input_tensor = current_input
                    else:
                        input_tensor = torch.from_numpy(current_input).float()
                    
                    input_tensor = input_tensor.to(device)
                    
                    if multi_step > 1:
                        prediction = model(input_tensor)
                        
                        if hasattr(prediction, 'cpu'):
                            prediction = prediction.cpu().numpy()
                        
                        if len(prediction.shape) == 1:
                            prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0 else prediction
                            predictions.append(prediction)
                            
                            new_input = np.roll(current_input, -1, axis=0)
                            new_input[-1] = prediction
                            current_input = new_input
                            i += 1
                        else:
                            if len(prediction.shape) > 1:
                                multi_predictions = prediction[0] if isinstance(prediction, np.ndarray) else prediction.flatten()
                                
                                for j in range(min(len(multi_predictions), multi_step, prediction_length - i)):
                                    predictions.append(multi_predictions[j])
                                    i += 1
                                    
                                    if i < prediction_length:
                                        new_input = np.roll(current_input, -1, axis=0)
                                        new_input[-1] = multi_predictions[j]
                                        current_input = new_input
                                        
                                    if i >= prediction_length:
                                        break
                    else:
                        prediction = model(input_tensor)
                        
                        if hasattr(prediction, 'cpu'):
                            prediction = prediction.cpu().numpy()
                        
                        if len(prediction.shape) > 1:
                            prediction = prediction.flatten()[0]
                        else:
                            prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0 else prediction
                            
                        predictions.append(prediction)
                        
                        if i < prediction_length - 1:
                            new_input = np.roll(current_input, -1, axis=0)
                            new_input[-1] = prediction
                            current_input = new_input
                        
                        i += 1
                        
                        if i >= prediction_length:
                            break
                
        return np.array(predictions)

    def _recursive_predict(self, model, X_test, prediction_length):
        """
        Performs recursive prediction (original method, for backward compatibility).
        
        Args:
            model: The model instance.
            X_test: The test data.
            prediction_length: The prediction length.
            
        Returns:
            list: The recursive prediction results.
        """
        model.eval()
        
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        
        self.window_size = self._infer_window_size(X_test, model)
        
        multi_step = self._infer_multi_step(X_test)
        
        predictions = []
        
        if len(X_test) < self.window_size:
            return np.array([])
            
        current_input = X_test[0:self.window_size].copy()
        
        i = 0
        while i < prediction_length and len(current_input) == self.window_size:
            with torch.no_grad():
                if torch.is_tensor(current_input):
                    input_tensor = current_input
                else:
                    input_tensor = torch.from_numpy(current_input).float()
                
                input_tensor = input_tensor.to(device)
                
                prediction = model(input_tensor)
                
                if hasattr(prediction, 'cpu'):
                    prediction = prediction.cpu().numpy()
                
                if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                    multi_predictions = prediction[0] if isinstance(prediction, np.ndarray) else prediction.flatten()
                    steps_predicted = len(multi_predictions)
                    
                    for j in range(min(steps_predicted, prediction_length - i)):
                        predictions.append(multi_predictions[j])
                        i += 1
                        
                        if i < prediction_length:
                            new_input = np.roll(current_input, -1, axis=0)
                            new_input[-1] = multi_predictions[j]
                            current_input = new_input
                            
                        if i >= prediction_length:
                            break
                else:
                    if len(prediction.shape) > 1:
                        prediction = prediction.flatten()[0]
                    else:
                        prediction = prediction[0] if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0 else prediction
                        
                    predictions.append(prediction)
                    
                    if i < prediction_length - 1:
                        new_input = np.roll(current_input, -1, axis=0)
                        new_input[-1] = prediction
                        current_input = new_input
                    
                    i += 1
                    
                    if i >= prediction_length:
                        break
            
        return np.array(predictions)

    def _infer_multi_step(self, X_test):
        """
        Automatically infers the number of multi-step prediction steps.
        
        Args:
            X_test: The test data.
            
        Returns:
            int: The number of multi-step prediction steps.
        """
        if X_test is not None and len(X_test.shape) >= 2:
            return X_test.shape[1]
        else:
            return 1

    def _calculate_metrics(self, predictions, true_values):
        """
        Calculates evaluation metrics.
        
        Args:
            predictions: The predicted values.
            true_values: The true values.
            
        Returns:
            dict: A dictionary containing various evaluation metrics.
        """
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        min_len = min(len(predictions), len(true_values))
        if min_len == 0:
            return {"MAPE": np.inf, "NSE": -np.inf}
            
        predictions = predictions[:min_len]
        true_values = true_values[:min_len]
        
        non_zero_true_values = true_values != 0
        
        if np.sum(non_zero_true_values) > 0:
            mape = np.mean(np.abs((true_values[non_zero_true_values] - predictions[non_zero_true_values]) / 
                                 true_values[non_zero_true_values])) * 100
        else:
            mape = np.inf
            
        numerator = np.sum((true_values - predictions) ** 2)
        denominator = np.sum((true_values - np.mean(true_values)) ** 2)
        nse = 1 - (numerator / denominator) if denominator != 0 else -np.inf
        
        return {
            "MAPE": mape,
            "NSE": nse
        }

    def _save_to_file(self, path: str, model_name: str) -> None:
        """
        Saves the analysis results to a file.
        
        Args:
            path: The save path.
            model_name: The name of the model.
        """
        if self.res is None:
            self.logger.warning("No results to save", module=self.__class__.__name__)
            return
            
        os.makedirs(path, exist_ok=True)
        
        df = pd.DataFrame({
            'True_Values': self.res['true_values'],
            'Recursive_Predictions': self.res['recursive_predictions']
        })
        df.to_csv(os.path.join(path, 'recursive_predictions.csv'), index=False)
        
        metrics_df = pd.DataFrame([self.res['metrics']])
        metrics_df.to_csv(os.path.join(path, 'recursive_metrics.csv'), index=False)
        
        self._plot_predictions(path)
        
        self.logger.info(f"Recursive prediction analysis results saved to: {path}", module=self.__class__.__name__)

    def _plot_predictions(self, path: str) -> None:
        """
        Plots the comparison of prediction results.
        
        Args:
            path: The save path.
        """
        if self.res is None:
            return
            
        true_values = self.res['true_values']
        predictions = self.res['recursive_predictions']
        metrics = self.res['metrics']
        
        if len(true_values) == 0 or len(predictions) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, label='True Values', linewidth=2)
        plt.plot(predictions, label='Recursive Predictions', linewidth=2, linestyle='--')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.title(f'Recursive Prediction vs True Values\nMAPE: {metrics["MAPE"]:.2f}%, NSE: {metrics["NSE"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'recursive_prediction_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()