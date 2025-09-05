from typing import Tuple
import numpy as np
import pandas as pd


class TimeWindowSplitter:
    def __init__(self, window_size=7, forecast_horizon=1):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    @classmethod
    def from_config(cls, config):
        window_size = config.get('window_size', 7)
        forecast_horizon = config.get('forecast_horizon', 1)
        return cls(window_size=window_size, forecast_horizon=forecast_horizon)
    
    def split(self, X: pd.DataFrame, y: pd.DataFrame, window=10, test_size=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the dataset using a sliding time window, with a default split ratio of 8:2,
        without shuffling the data.
        
        Parameters:
            X (pd.DataFrame or np.ndarray): Feature data
            y (pd.DataFrame or np.ndarray): Target variable
            window (int): The size of the time window
            test_size (float): The proportion of the dataset to include in the test split
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if len(X) != len(y):
            raise ValueError("The lengths of `X` and `y` must be the same")
        
        data_len = len(X)
        train_size = int(data_len * (1 - test_size))
        
        # Automatically convert to numpy arrays (compatible with DataFrame)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.DataFrame) else y

        # Build training set inputs and outputs
        X_train, y_train = [], []
        for i in range(window, train_size - self.forecast_horizon + 1):
            X_train.append(X_np[i - window:i])
            y_train.append(y_np[i:i + self.forecast_horizon])
        
        # Build test set inputs and outputs
        X_test, y_test = [], []
        for i in range(train_size, data_len - self.forecast_horizon + 1):
            if i - window < 0:
                continue  # Skip data points that don't have a full window
            X_test.append(X_np[i - window:i])
            y_test.append(y_np[i:i + self.forecast_horizon])
        
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
