import matplotlib.pyplot as plt
import numpy as np
from src.analyzer.analyzerBase import AnalyzerBase
from ..factory.analyzer_factory import AnalyzerFactory
import os


@AnalyzerFactory.register('plot_analyzer')
class PlotAnalyzer(AnalyzerBase):
    def __init__(self, plot_type='line'):
        super().__init__()
        self.plot_type = plot_type
        self.figures = []

    @classmethod
    def from_config(cls, config):
        plot_type = config.get('plot_type', 'line')
        return cls(plot_type=plot_type)

    def analyze(self, predictions, true_values, **kwargs):
        if self.plot_type == 'line':
            fig, ax = plt.subplots()
            
            for model_name, preds in predictions.items():
                if isinstance(preds, (list, np.ndarray)) and len(preds) > 0 and isinstance(preds[0], (list, np.ndarray)):
                    preds_avg = np.mean(preds, axis=1)
                    ax.plot(preds_avg, label=f'Prediction - {model_name}')
                else:
                    ax.plot(preds, label=f'Prediction - {model_name}')

                if isinstance(true_values, (list, np.ndarray)) and len(true_values) > 0:
                    true_values = true_values[:,0]

            ax.plot(true_values, label='True Value')
            ax.legend()

            self.figures.append(fig)
        else:
            raise ValueError(f"Unsupported plot type: {self.plot_type}")

    def _save_to_file(self, path: str, model_name: str) -> None:
        """
        Abstract method for saving files, to be implemented by subclasses.
        Supports SVG format only.

        Args:
            path: The full file path to save the file (including filename and extension).
        """
        if not self.figures:
            raise ValueError("No figures to save")

        path = os.path.join(path, f'{model_name}_plot.svg')

        for i, fig in enumerate(self.figures):
            fig.savefig(path, format='svg')
            plt.close(fig)

        print(f"Saved {len(self.figures)} SVG figures to {path}")