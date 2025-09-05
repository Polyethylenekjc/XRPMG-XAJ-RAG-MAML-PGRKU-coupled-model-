# src/analyzer/mape_interval_analyzer.py
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_percentage_error
from src.analyzer.analyzerBase import AnalyzerBase
from src.factory.analyzer_factory import AnalyzerFactory
from typing import Any, Dict, Optional

@AnalyzerFactory.register('mape_interval_analyzer')
class MapeAndIntervalAnalyzer(AnalyzerBase):
    """
    Analyzer supporting MAPE, D1/D10 MAPE, and PICP/PINAW metric analysis and saving
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.res = {
            'mape': {},
            'picp_pinaw': {}
        }
        self.models: Optional[list] = None

    def analyze(self, predictions: dict, true_values, locations=None, models=None, alpha_values=None, **kwargs):
        """
        Perform MAPE, D1/D10 MAPE, PICP/PINAW analysis and save results
        """
        if not isinstance(true_values, dict):
            combined_true_values = {}
            for key in predictions:
                combined_true_values[key] = true_values[:len(predictions[key])]
            true_values = combined_true_values

        parsed_predictions = {}
        parsed_true_values = {}

        for key in predictions:
            parts = key.split('_')
            if len(parts) < 2:
                raise ValueError(f"Key '{key}' format is incorrect, should be model_location")

            model = '_'.join(parts[:-1])
            location = parts[-1]

            pred_tensor = predictions[key]
            pred = pred_tensor.cpu().numpy() if hasattr(pred_tensor, 'is_cuda') and pred_tensor.is_cuda else pred_tensor.numpy() if hasattr(pred_tensor, 'numpy') else pred_tensor

            truth_tensor = true_values.get(key)
            truth = truth_tensor.cpu().numpy() if hasattr(truth_tensor, 'is_cuda') and truth_tensor.is_cuda else truth_tensor.numpy() if hasattr(truth_tensor, 'numpy') else truth_tensor

            parsed_predictions.setdefault(location, {})[model] = pred
            parsed_true_values.setdefault(location, {})[model] = truth

        predictions = parsed_predictions
        true_values = parsed_true_values

        if locations is None:
            locations = list(predictions.keys())
        if models is None:
            if len(locations) > 0:
                models = list(next(iter(predictions.values())).keys())
            else:
                models = []

        self.models = models
        if alpha_values is None:
            alpha_values = [0.05, 0.1, 0.2]

        mape_results = self._analyze_mape(predictions, true_values, locations, models)
        self.res['mape'] = mape_results

        picp_pinaw_results = self._analyze_picp_pinaw(predictions, true_values, locations, models, alpha_values)
        self.res['picp_pinaw'] = picp_pinaw_results

    def _analyze_mape(self, predictions: dict, true_values: dict, locations, models):
        results = {city: [] for city in locations}
        for city in locations:
            for model in models:
                pred = predictions[city][model]
                truth = true_values[city][model]
                all_mape = self._calculate_mape(truth, pred)
                d1_threshold = np.percentile(truth, 10)
                d10_threshold = np.percentile(truth, 90)
                d1_mask = truth <= d1_threshold
                d10_mask = truth >= d10_threshold
                d1_mape = self._calculate_mape(truth[d1_mask], pred[d1_mask]) if np.any(d1_mask) else np.nan
                d10_mape = self._calculate_mape(truth[d10_mask], pred[d10_mask]) if np.any(d10_mask) else np.nan
                results[city].append([all_mape, d1_mape, d10_mape])
        return results

    def _calculate_mape(self, y_true, y_pred):
        epsilon = 1e-8
        return mean_absolute_percentage_error(y_true + epsilon, y_pred + epsilon)

    def _analyze_picp_pinaw(self, predictions: dict, true_values: dict, locations, models, alpha_values):
        results = {loc: {} for loc in locations}
        for location in locations:
            results[location] = {}
            for model in models:
                pred = predictions[location][model]
                truth = true_values[location][model]
                std_dev = np.std(truth - pred)
                res = {}
                for alpha in alpha_values:
                    z_score = norm.ppf(1 - alpha / 2)
                    lower = pred - z_score * std_dev
                    upper = pred + z_score * std_dev
                    picp, pinaw = self._calculate_picp_pinaw(truth, lower, upper)
                    res[alpha] = {'PICP': picp, 'PINAW': pinaw}
                results[location][model] = res
        return results

    def _calculate_picp_pinaw(self, truth, lower, upper):
        in_interval = np.sum((lower <= truth) & (truth <= upper))
        picp = in_interval / len(truth)
        interval_widths = upper - lower
        true_range = np.max(truth) - np.min(truth)
        if true_range == 0:
            true_range = 1.0
        pinaw = np.mean(interval_widths) / true_range
        return picp, pinaw

    def _save_to_file(self, path: str, model_name: str) -> None:
        """Save analysis results to file"""
        os.makedirs(path, exist_ok=True)
        mape_df = self._generate_mape_dataframe()
        mape_output_path = os.path.join(path, "mape_comparison.csv")
        mape_df.to_csv(mape_output_path, index=False)
        print(f'MAPE results saved to {mape_output_path}')
        picp_pinaw_df = self._generate_picp_pinaw_dataframe()
        picp_pinaw_output_path = os.path.join(path, "picp_pinaw_comparison.csv")
        picp_pinaw_df.to_csv(picp_pinaw_output_path, index=False)
        print(f'PICP/PINAW results saved to {picp_pinaw_output_path}')

    def _generate_mape_dataframe(self):
        data = []
        if self.res['mape'] and self.models:
            for city, models_data in self.res['mape'].items():
                for i, model_name in enumerate(self.models):
                    mape_values = models_data[i]
                    data.append({
                        'Location': city,
                        'Model': model_name,
                        'All result': mape_values[0],
                        'D1': mape_values[1],
                        'D10': mape_values[2]
                    })
        return pd.DataFrame(data)

    def _generate_picp_pinaw_dataframe(self):
        data = []
        if self.res['picp_pinaw']:
            for location, models_data in self.res['picp_pinaw'].items():
                for model, alphas in models_data.items():
                    for alpha, metrics in alphas.items():
                        row = {
                            'Location': location,
                            'Model': model,
                            'Alpha': f'{alpha:.2f}',
                            'PICP': metrics['PICP'],
                            'PINAW': metrics['PINAW']
                        }
                        data.append(row)
        return pd.DataFrame(data)

    def _plot_mape_results(self, df, output_dir='pic'):
        # This function is deprecated as plotting is handled by SummaryAnalyzer
        pass

    def _plot_picp_pinaw_results(self, df, output_dir='pic'):
        # This function is deprecated as plotting is handled by SummaryAnalyzer
        pass