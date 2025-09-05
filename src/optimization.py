import yaml
import numpy as np
import pandas as pd
import os
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from src.pipeline import MLExperimentPipeline
import copy
import logging

class SequentialBayesianOptimizer:
    def __init__(self, config_path='Config/opt.yaml', results_path='res/optimization_results.csv'):
        """
        Initialize the sequential Bayesian optimizer.
        
        Args:
            config_path (str): The path to the configuration file.
            results_path (str): The path to save the results.
        """
        self.config_path = config_path
        self.results_path = results_path
        self.results_df = pd.DataFrame(columns=[
            'model', 'iteration', 'mse', 'parameters'
        ])
        
        # Create the results directory
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # If the results file already exists, load it
        if os.path.exists(results_path):
            self.results_df = pd.read_csv(results_path)
    
    def load_config(self):
        """
        Load the configuration file.
        
        Returns:
            dict: The configuration dictionary.
        """
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_all_models(self, config):
        """
        Get a list of all models from the configuration.
        
        Args:
            config (dict): The configuration dictionary.
            
        Returns:
            list: A list of model names.
        """
        return list(config['models'].keys())
    
    def enable_single_model(self, config, target_model):
        """
        Set the configuration to enable only the specified model and disable others.
        
        Args:
            config (dict): The configuration dictionary.
            target_model (str): The name of the target model.
            
        Returns:
            dict: The updated configuration.
        """
        new_config = copy.deepcopy(config)
        for model_name in new_config['models']:
            new_config['models'][model_name]['enabled'] = (model_name == target_model)
        return new_config
    
    def convert_params_to_python_types(self, params):
        """
        Convert numpy types to native Python types.
        
        Args:
            params (dict): The parameters dictionary.
            
        Returns:
            dict: The dictionary with converted parameter types.
        """
        converted_params = {}
        for key, value in params.items():
            if isinstance(value, np.integer):
                converted_params[key] = int(value)
            elif isinstance(value, np.floating):
                converted_params[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted_params[key] = value.tolist()
            else:
                converted_params[key] = value
        return converted_params
    
    def define_search_space(self, model_name):
        """
        Define the search space based on the model name.
        
        Args:
            model_name (str): The model name.
            
        Returns:
            list: The parameter search space.
        """
        if model_name in ['lstm_time_series_model', 'gru_time_series_model']:
            return [
                Integer(1, 3, name='num_layers'),
                Integer(32, 128, name='hidden_dim'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'KAN':
            return [
                Real(0.001, 0.1, name='grid_eps', prior='log-uniform'),
                Integer(3, 10, name='grid_size'),
                Integer(2, 5, name='spline_order'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'grku_time_series_model':
            return [
                Real(0.001, 0.1, name='grid_eps', prior='log-uniform'),
                Integer(1, 5, name='num_layers'),
                Integer(32, 256, name='hidden_dim'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'attention_gru':
            return [
                Integer(1, 3, name='num_layers'),
                Integer(32, 128, name='hidden_dim'),
                Real(0.0, 0.5, name='dropout'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'cnn_transformer':
            return [
                Integer(1, 3, name='num_layers'),
                Integer(4, 128, name='hidden_dim'),  # Ensure it is divisible by num_heads
                Integer(1, 8, name='num_heads'),    # Restrict the range to ensure constraints are met
                Real(0.0, 0.5, name='dropout'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'eklt_model':
            return [
                Integer(1, 3, name='lstm_num_layers'),
                Integer(32, 128, name='lstm_hidden_dim'),
                Integer(1, 3, name='transformer_num_layers'),
                Integer(2, 8, name='transformer_nhead'),
                Real(0.0, 0.5, name='dropout'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'simple_pytorch_model':
            return [
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        elif model_name == 'vmdi_lstm_ed_model':
            return [
                Integer(1, 3, name='num_layers'),
                Integer(32, 128, name='hidden_dim'),
                Real(0.0, 0.5, name='dropout'),
                Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform')
            ]
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def update_config_with_params(self, config, model_name, params):
        """
        Update the configuration with optimization parameters.
        
        Args:
            config (dict): The original configuration.
            model_name (str): The model name.
            params (dict): The parameters dictionary.
            
        Returns:
            dict: The updated configuration.
        """
        # Convert parameter types
        converted_params = self.convert_params_to_python_types(params)
        
        new_config = copy.deepcopy(config)
        model_config = new_config['models'][model_name]
        
        if model_name in ['lstm_time_series_model', 'gru_time_series_model']:
            model_config['params']['num_layers'] = converted_params['num_layers']
            model_config['params']['hidden_dim'] = converted_params['hidden_dim']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'KAN':
            model_config['params']['grid_eps'] = converted_params['grid_eps']
            model_config['params']['grid_size'] = converted_params['grid_size']
            model_config['params']['spline_order'] = converted_params['spline_order']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'grku_time_series_model':
            model_config['params']['grid_eps'] = converted_params['grid_eps']
            model_config['params']['num_layers'] = converted_params['num_layers']
            model_config['params']['hidden_dim'] = converted_params['hidden_dim']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'attention_gru':
            model_config['params']['num_layers'] = converted_params['num_layers']
            model_config['params']['hidden_dim'] = converted_params['hidden_dim']
            model_config['params']['dropout'] = converted_params['dropout']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'cnn_transformer':
            model_config['params']['num_layers'] = converted_params['num_layers']
            model_config['params']['hidden_dim'] = converted_params['hidden_dim']
            # Ensure hidden_dim is divisible by num_heads
            num_heads = converted_params['num_heads']
            hidden_dim = converted_params['hidden_dim']
            adjusted_hidden_dim = (hidden_dim // num_heads) * num_heads
            if adjusted_hidden_dim == 0:
                adjusted_hidden_dim = num_heads
            model_config['params']['hidden_dim'] = adjusted_hidden_dim
            model_config['params']['num_heads'] = num_heads
            model_config['params']['dropout'] = converted_params['dropout']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'eklt_model':
            model_config['params']['lstm_num_layers'] = converted_params['lstm_num_layers']
            model_config['params']['lstm_hidden_dim'] = converted_params['lstm_hidden_dim']
            model_config['params']['transformer_num_layers'] = converted_params['transformer_num_layers']
            model_config['params']['transformer_nhead'] = converted_params['transformer_nhead']
            # Ensure LSTM hidden dim is equal to Transformer d_model
            hidden_dim = converted_params['lstm_hidden_dim']
            model_config['params']['lstm_hidden_dim'] = hidden_dim
            model_config['params']['transformer_d_model'] = hidden_dim
            model_config['params']['dropout'] = converted_params['dropout']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'simple_pytorch_model':
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        elif model_name == 'vmdi_lstm_ed_model':
            model_config['params']['num_layers'] = converted_params['num_layers']
            model_config['params']['hidden_dim'] = converted_params['hidden_dim']
            model_config['params']['dropout'] = converted_params['dropout']
            model_config['trainer']['learning_rate'] = converted_params['learning_rate']
            
        return new_config
    
    def create_objective_function(self, model_name, config):
        """
        Create the objective function for Bayesian optimization.
        
        Args:
            model_name (str): The model name.
            config (dict): The base configuration.
            
        Returns:
            function: The objective function.
            list: The dimensions of the search space.
        """
        dimensions = self.define_search_space(model_name)
        
        @use_named_args(dimensions)
        def objective(**params):
            # Update configuration parameters
            updated_config = self.update_config_with_params(config, model_name, params)
            
            try:
                # Run the experiment
                pipeline = MLExperimentPipeline(updated_config)
                _, total_mse = pipeline.runExperiment()
                print(f"{model_name} - MSE: {total_mse}")
                return total_mse
            except Exception as e:
                logging.error(f"Experiment failed to run: {e}")
                return np.inf  # Return infinity to indicate failure
        
        return objective, dimensions
    
    def optimize_single_model(self, model_name, n_calls=30):
        """
        Optimize a single model.
        
        Args:
            model_name (str): The model name.
            n_calls (int): The number of optimization iterations.
        """
        print(f"Starting optimization for model: {model_name}")
        
        # Load configuration and enable a single model
        config = self.load_config()
        config = self.enable_single_model(config, model_name)
        
        # Create the objective function
        objective_function, dimensions = self.create_objective_function(model_name, config)
        
        # Execute Bayesian optimization
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Get the best parameters
        param_names = [d.name for d in dimensions]
        best_params = dict(zip(param_names, result.x))
        
        # Convert best parameter types
        best_params = self.convert_params_to_python_types(best_params)
        
        print(f"Model {model_name} optimization complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best MSE: {result.fun}")
        
        # Save results to DataFrame
        new_row = {
            'model': model_name,
            'iteration': 'final',
            'mse': result.fun,
            'parameters': str(best_params)
        }
        self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        self.results_df.to_csv(self.results_path, index=False)
        
        return best_params, result.fun
    
    def optimize_all_models(self, model_names=None, n_calls_per_model=30):
        """
        Sequentially optimize all models.
        
        Args:
            model_names (list): A list of models to optimize; if None, all models are optimized.
            n_calls_per_model (int): The number of optimization iterations for each model.
        """
        # Load configuration and get the list of models
        config = self.load_config()
        all_models = self.get_all_models(config)
        
        if model_names is None:
            model_names = all_models
        
        print(f"List of models to be optimized: {model_names}")
        
        # Sequentially optimize each model
        for model_name in model_names:
            if model_name not in all_models:
                print(f"Warning: Model {model_name} does not exist in the configuration, skipping")
                continue
                
            # Check if the model has already been optimized
            existing_results = self.results_df[self.results_df['model'] == model_name]
            if not existing_results.empty:
                print(f"Model {model_name} already has optimization results, skipping")
                continue
            
            # Optimize a single model
            try:
                best_params, best_mse = self.optimize_single_model(model_name, n_calls_per_model)
                print(f"Model {model_name} optimization complete, best MSE: {best_mse}")
            except Exception as e:
                print(f"An error occurred during optimization for model {model_name}: {e}")
        
        print("All models optimization complete!")
        print(f"Results saved to: {self.results_path}")

def main():
    """
    Main function
    """
    optimizer = SequentialBayesianOptimizer(
        config_path='Config/opt.yaml',
        results_path='res/optimization_results.csv'
    )
    
    # Optimize all enabled models
    optimizer.optimize_all_models(n_calls_per_model=30,model_names=["eklt_model"])

if __name__ == "__main__":
    main()
