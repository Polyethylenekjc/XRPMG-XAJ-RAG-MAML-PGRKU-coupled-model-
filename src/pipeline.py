import numpy as np
import pandas as pd
import os
from src.analyzer.SummaryAnalyzer import SummaryAnalyzer
from src.data.dataprocessorBase import CombinedDataProcessor
from src.factory.data_processor_factory import DataProcessorFactory
from src.factory.model_factory import ModelFactory
from src.factory.analyzer_factory import AnalyzerFactory
from src.factory.trainer_factory import TrainerFactory
from src.trainer.base_model_trainer import BaseModelTrainer
from src.utils.time_window_split import TimeWindowSplitter
from src.utils.Logger import Logger 


class MLExperimentPipeline:
    def __init__(self, config):
        """Initializes the experiment pipeline, configuring data processors, models, and analyzers"""
        self.logger = Logger() # Modified to be an instance
        self.config = config
        self.data_processors = self._create_data_processors()
        self.models = self._build_models()
        self.analyzers = self._create_analyzers()
        self._X_test_cache = {}

        # Create the time window splitter
        self.time_window_splitter = TimeWindowSplitter(
            window_size=self.config.get("window", 15),
            forecast_horizon=self.config.get("forecast_horizon", 3)
        )

    def _get_X_test(self, dataset_name):
        """Gets the cached test feature data"""
        if dataset_name not in self._X_test_cache:
            raise ValueError(f"X_test data not cached for dataset {dataset_name}")
        return self._X_test_cache[dataset_name]

    def _create_data_processors(self):
        """
        Creates multiple data processors based on the configuration
        
        Returns:
            dict: A mapping from dataset names to data processor instances
        """
        data_configs = self.config.get("data", {}).get("datasets", [])
        data_processors = {}
        
        self.logger.info("Starting to create data processors", module="MLExperimentPipeline")
        
        for dataset_config in data_configs:
            # New: Check if dataset is enabled
            if not dataset_config.get("enabled", True):
                continue
                
            dataset_name = dataset_config["name"]
            self.logger.info(f"Creating data processor for dataset {dataset_name}", module="MLExperimentPipeline")
            
            if 'processors' not in dataset_config:
                raise ValueError(f"No processor configuration found for dataset {dataset_name}")
                
            processor_names = [p['name'] for p in dataset_config['processors']]
            self.logger.info(f"Dataset {dataset_name} will use the following processors: {', '.join(processor_names)}", 
                             module="MLExperimentPipeline")
            
            data_processor = CombinedDataProcessor(DataProcessorFactory.create_all(dataset_config['processors']))
            data_processors[dataset_name] = data_processor
        
        self.logger.info("Data processors created successfully", module="MLExperimentPipeline")
        for dataset_name, processor in data_processors.items():
            self.logger.info(f"- {dataset_name}: {type(processor).__name__}", module="MLExperimentPipeline")
            
        return data_processors

    def _build_models(self):
        """
        Builds enabled models based on the configuration
        
        Returns:
            dict: A mapping from model names to model instances
        """
        models = {}
        
        self.logger.info("Starting to build models", module="MLExperimentPipeline")
        
        for model_name, model_config in self.config.get("models", {}).items():
            if not model_config.get("enabled"):
                continue
                
            params = model_config.get("params", {})
            model = ModelFactory.create(model_name, **params)
            models[model_name] = model
            self.logger.info(f"- Enabling model {model_name} with parameters: {params}", module="MLExperimentPipeline")
        
        if not models:
            self.logger.warning("No models enabled", module="MLExperimentPipeline")
        else:
            self.logger.info(f"Successfully built {len(models)} models: {', '.join(models.keys())}", module="MLExperimentPipeline")
            print("\nModel Details:")
            for model_name, model in models.items():
                print(f"\n{model_name}:")
                print(model)
                
        return models

    def _create_analyzers(self):
        """
        Creates analyzer instances based on the configuration
        
        Returns:
            list: A list of the created analyzer instances
        """
        analyzers_config = self.config.get("analyzers", [])
        filtered_configs = []
        
        self.logger.info("Starting to create analyzers", module="MLExperimentPipeline")
        
        # Filter out SHAP analyzer configurations for non-specified models
        for config in analyzers_config:
            if config.get("name") == "shap_analyzer":
                model_name = config.get("params", {}).get("model_name")
                if model_name and model_name in self.models:
                    filtered_configs.append(config)
                    self.logger.info(f"- Enabling SHAP analyzer (model: {model_name})", module="MLExperimentPipeline")
                else:
                    self.logger.warning(f"- Skipping SHAP analyzer: Model {model_name} does not exist or is not configured", module="MLExperimentPipeline")
            else:
                filtered_configs.append(config)
                self.logger.info(f"- Enabling {config.get('name')} analyzer", module="MLExperimentPipeline")
        
        analyzers = AnalyzerFactory.create_all(filtered_configs)
        
        if not analyzers:
            self.logger.warning("No analyzers created", module="MLExperimentPipeline")
        else:
            self.logger.info(f"Successfully created {len(analyzers)} analyzers: {[type(a).__name__ for a in analyzers]}", 
                             module="MLExperimentPipeline")
        
        return analyzers

    def runExperiment(self):
        """
        Runs the entire experiment pipeline
        
        Returns:
            tuple: (all_predictions, total_mse) All predictions for all datasets and models, and the total MSE sum
        """
        all_predictions = {} # Stores prediction results and corresponding true values for all datasets and models
        self._feature_names_cache = {} # New: Used to cache feature column names

        # Process each dataset
        for dataset_name, data_processor in self.data_processors.items():
            # Can re-check configuration here for safety
            dataset_config = next((d for d in self.config["data"]["datasets"] if d["name"] == dataset_name), None)
            if dataset_config and not dataset_config.get("enabled", True):
                continue
                
            self.logger.info(f"Processing dataset: {dataset_name}", module="MLExperimentPipeline")
            df_processed = self.load_and_process_data(dataset_name, data_processor)
            features, target_col, X, y, window, test_size = self.prepare_features_and_split(df_processed, dataset_name)
            self.true = y
            
            # Time window splitting
            X_train, y_train, X_test, y_test = self.time_window_splitter.split(X, y, window=window, test_size=test_size)

            # Cache X_test and feature_names
            self._X_test_cache[dataset_name] = X_test
            self._feature_names_cache[dataset_name] = features # New: Cache column names

            # Uniformly convert to numpy arrays
            X_train, X_test = self._convert_to_array(X_train), self._convert_to_array(X_test)
            y_train, y_test = self._convert_to_array(y_train), self._convert_to_array(y_test)

            # Ensure y_train and y_test are 2D
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1) # type: ignore
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1) # type: ignore

            predictions = self.train_and_predict(X_train, y_train, X_test, y_test, dataset_name, data_processor, target_col)
            all_predictions[dataset_name] = predictions

            # New: Save predictions for each model
            for model_name, result in predictions.items():
                prediction = result['prediction']
                true_values = result['true_values']
                try:
                    self._save_prediction_to_csv(dataset_name, model_name, prediction, true_values)
                except Exception as e:
                    self.logger.error(f"Error saving predictions for {model_name} on {dataset_name}: {str(e)}", module="MLExperimentPipeline")

        # Add logging: show complete prediction status for all dataset and model combinations
        self.logger.info("Experiment completed, here is a complete overview of the predictions:", module="MLExperimentPipeline")
        summary_info = {}
        for dataset_name, predictions in all_predictions.items():
            model_names = list(predictions.keys())
            summary_info[dataset_name] = model_names
            # Output a concise version to the console
            print(f"Dataset '{dataset_name}' has been used to predict with the following models: {', '.join(model_names)}")

        self.analyze_results(all_predictions)
        
        # Calculate the total MSE sum of all predictions
        total_mse = 0.0
        for dataset_predictions in all_predictions.values():
            for model_result in dataset_predictions.values():
                predictions = model_result['prediction']
                true_values = model_result['true_values']
                
                # Ensure consistent array shapes
                min_length = min(len(predictions), len(true_values))
                predictions = predictions[:min_length]
                true_values = true_values[:min_length]
                
                # Calculate MSE
                mse = np.mean((predictions - true_values) ** 2)
                total_mse += mse

        return all_predictions, total_mse # Return all prediction results and the total MSE sum

    def train_and_predict(self, X_train, y_train, X_test, y_test, dataset_name, data_processor, target_col):
        """
        Trains the model and makes predictions
        
        Args:
            X_train: Training feature data
            y_train: Training target data
            X_test: Test feature data
            y_test: Test target data
            dataset_name: Dataset name
            data_processor: Data processor instance
            target_col: Target column name
            
        Returns:
            dict: Prediction results and corresponding true values for each model under the current dataset
        """
        predictions = {} # Stores prediction results and corresponding true values for each model under the current dataset

        for model_name, model in self.models.items():
            self.logger.info(f"Training model: {model_name} on dataset: {dataset_name}", module="MLExperimentPipeline")

            model_config = self.config["models"][model_name]
            trainer_config = model_config.get("trainer", {}) # Get the trainer configuration specific to the model

            # Dynamically create a trainer based on the trainer.type field
            trainer_class_name = trainer_config.get("type")
            if not trainer_class_name:
                raise ValueError(f"Missing 'type' in trainer configuration for {model_name}")

            try:
                # Use TrainerFactory to create the trainer
                trainer = TrainerFactory.create(trainer_class_name, model=model, config=trainer_config)
            except ValueError as e:
                self.logger.error(f"Error creating trainer for {model_name}: {e}", module="MLExperimentPipeline")
                continue # Skip this model

            self.logger.info(f"Created trainer of type: {trainer_class_name} for {model_name}", module="MLExperimentPipeline")

            # If training data is provided, train directly
            if X_train is not None and y_train is not None:
                self.logger.info(f"Training {model_name} on {dataset_name}", module="MLExperimentPipeline")
                trainer.train(X_train, y_train)

            # Save the model and its parameter configuration
            if hasattr(trainer, 'save_model') and callable(trainer.save_model):
                try:
                    model_path = trainer.save_model(model_name, dataset_name)
                    self.logger.info(f"Model saved to {model_path}", module="MLExperimentPipeline")
                except Exception as e:
                    self.logger.error(f"Error saving model {model_name}: {str(e)}", module="MLExperimentPipeline")

            # Get prediction results
            prediction = trainer.predict(X_test)
            X_full = np.concatenate([X_train, X_test], axis=0)
            Y_full = np.concatenate([y_train, y_test], axis=0)  
            full_prediction = trainer.predict(X_full)

            print(full_prediction.shape,Y_full.shape)

            analysis_path = self.config.get('output', {}).get('analysis_result_path', 'res/analysis/')
            final_path = os.path.join(analysis_path, dataset_name, model_name)
            os.makedirs(final_path, exist_ok=True)

            full = {"True":Y_full.squeeze(),"Prediction":full_prediction.squeeze()}
            full_prediction_df = pd.DataFrame().from_dict(full)
            full_prediction_df.to_csv(os.path.join(final_path, 'full_dataset_prediction.csv'), index=False)

            # Save predictions and true values
            predictions[model_name] = {
                'prediction': prediction,
                'true_values': y_test[:len(prediction)] # type: ignore
            }

        return predictions

    def load_and_process_data(self, dataset_name, data_processor):
        """
        Loads and preprocesses the data
        
        Args:
            dataset_name: Dataset name
            data_processor: Data processor instance
            
        Returns:
            pd.DataFrame: The preprocessed DataFrame
        """
        # Get dataset configuration
        dataset_config = next((d for d in self.config["data"]["datasets"] if d["name"] == dataset_name), None)
        if not dataset_config:
            raise ValueError(f"Dataset config not found for {dataset_name}")

        # Load data
        df = pd.read_csv(dataset_config["path"])

        df_processed = data_processor.process(df) # Then pass it to the process method

        # Ensure output_path is defined in all cases
        output_config = self.config.get("output", {})
        data_result_path = output_config.get("data_result_path", "res/data/") # Use default path as fallback
        output_path = f"{data_result_path}{dataset_name}_processed.csv"

        try:
            df_processed.to_csv(output_path, index=False)
        except Exception as e:
            self.logger.error(f"Error saving processed data for {dataset_name}: {str(e)}", module="MLExperimentPipeline")

        return df_processed

    def prepare_features_and_split(self, df_processed, dataset_name):
        """
        Prepares features and performs time window splitting

        Args:
            df_processed: The preprocessed DataFrame
            dataset_name: Dataset name

        Returns:
            tuple: A tuple containing feature columns, target column, feature data, target data, window size, and test set proportion
        """
        # Get dataset configuration
        dataset_config = next((d for d in self.config["data"]["datasets"] if d["name"] == dataset_name), None)
        if not dataset_config:
            raise ValueError(f"Dataset config not found for {dataset_name}")

        # Extract target column and ignore columns
        target_col = dataset_config.get("target_column") # Re-verify target_column
        ignore_cols = dataset_config.get("ignore_columns", [])

        # All columns that are not ignore_cols and not target_col are features
        ignore_cols = ignore_cols or []
        features = [c for c in df_processed.columns if c not in ignore_cols and c != target_col]

        # Use original feature columns and target column directly
        X = df_processed[features + [target_col]].copy()
        y = X[target_col]

        X = X.loc[:, sorted(X.columns)]
        features = [c for c in X.columns]
        self.logger.info(f"Using features: {features}")

        # Use TimeWindowSplitter to automatically construct windows
        window = dataset_config.get("window", 7)
        test_size = dataset_config.get("test_size", 0.2)

        return features, target_col, X, y, window, test_size

    def analyze_results(self, all_predictions):
        """
        Analyzes prediction results
        
        Args:
            all_predictions: All predictions for all datasets and models
        """
        for analyzer in self.analyzers:
            # Analyze results for each dataset
            for dataset_name, predictions in all_predictions.items():
                self.logger.debug(f"Analyzing results for dataset {dataset_name}", module="MLExperimentPipeline")
                if not predictions:
                    raise ValueError(f"No predictions found for dataset {dataset_name}")

                for model_name, result in predictions.items():
                    prediction = result['prediction']
                    true_values = result['true_values']

                    self.logger.info(f"Analyzing results for {model_name} on {dataset_name}", module="MLExperimentPipeline")
                    result_key = f"{model_name}_{dataset_name}"

                    # Pass different parameters based on analyzer type
                    analyzer_class_name = analyzer.__class__.__name__
                    
                    try:
                        if analyzer_class_name == 'ShapAnalyzer':
                            # SHAP analyzer requires specific parameters
                            shap_model_name = getattr(analyzer, 'model_name', None)
                            if shap_model_name and shap_model_name != model_name:
                                self.logger.info(f"Skipping SHAP analysis for {model_name} on {dataset_name}: "
                                                 f"configured for model {shap_model_name}", module="MLExperimentPipeline")
                                continue
                            
                            if hasattr(analyzer, 'model_name'):
                                if getattr(analyzer, 'model_name', None) == model_name and model_name in self.models:
                                    model = self.models[model_name]
                                    X_test_dataset = self._get_X_test(dataset_name)
                                    X_test_dataset = self._convert_to_array(X_test_dataset)
                                    feature_names = self._feature_names_cache.get(dataset_name)
                                    
                                    analyzer.analyze(
                                        {result_key: prediction},
                                        true_values,
                                        model=model,
                                        X_test=X_test_dataset,
                                        feature_names=feature_names
                                    )
                                else:
                                    self.logger.warning(f"Skipping SHAP analysis for {model_name}: model not found or not configured", module="MLExperimentPipeline")
                            else:
                                self.logger.warning(f"SHAP analyzer missing model_name attribute", module="MLExperimentPipeline")
                        elif analyzer_class_name == 'RecursivePredictorAnalyzer':
                            # Recursive prediction analyzer needs model and test data
                            model = self.models.get(model_name)
                            if model and model_name in self.models:
                                X_test_dataset = self._get_X_test(dataset_name)
                                X_test_dataset = self._convert_to_array(X_test_dataset)
                                
                                analyzer.analyze(
                                    {result_key: prediction},
                                    true_values,
                                    model=model,
                                    X_test=X_test_dataset
                                )
                            else:
                                self.logger.warning(f"Skipping recursive prediction analysis for {model_name}: model not found", module="MLExperimentPipeline")
                                
                        else:
                            # Other analyzers use default parameters
                            analyzer.analyze({result_key: prediction}, true_values)
                            
                    except Exception as e:
                        self.logger.error(f"Error analyzing {model_name} on {dataset_name} with {analyzer_class_name}: {str(e)}", module="MLExperimentPipeline")

                    analyzer.save(dataset_name, model_name)

    def _save_prediction_to_csv(self, dataset_name, model_name, prediction, true_values):
        """
        Saves predictions and true values to a CSV file
        
        Args:
            dataset_name: Dataset name
            model_name: Model name
            prediction: Prediction array
            true_values: True values array
        """
        # Get output path configuration
        output_config = self.config.get("output", {})
        analysis_result_path = output_config.get("analysis_result_path", "res/analysis/") # Use default path as fallback

        # Build the complete save path
        save_dir = f"{analysis_result_path}{dataset_name}/{model_name}/"
        try:
            os.makedirs(save_dir, exist_ok=True) # Create directory (if it does not exist)
        except Exception as e:
            self.logger.error(f"Error creating directory {save_dir}: {str(e)}", module="MLExperimentPipeline")
            return

        # Prepare DataFrame
        try:
            df = pd.DataFrame({
                'Prediction': prediction.flatten(), # Flatten the array to ensure it's one-dimensional
                'TrueValue': true_values.flatten()
            })

            # Save as a CSV file
            file_path = f"{save_dir}predictions.csv"
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Saved predictions for {model_name} on {dataset_name} to {file_path}", module="MLExperimentPipeline")
        except Exception as e:
            self.logger.error(f"Error saving predictions to CSV: {str(e)}", module="MLExperimentPipeline")

    def _convert_to_array(self, data):
        """Uniformly converts input data to a numpy array"""
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
