import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import numpy as np
from src.utils.Logger import Logger
from src.factory.trainer_factory import TrainerFactory
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


@TrainerFactory.register("BaseModelTrainer")
class BaseModelTrainer:
    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4  # Enable L2 regularization
        )
        self.model_save_path = config.get('model_save_path', './models')
        self.logger = Logger

        # Check if CUDA is available and move the model to the corresponding device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Model deployed to device: {'CUDA' if self.device.type == 'cuda' else 'CPU'}", module="BaseModelTrainer")

        # If it's a RAG model, initialize the RAG dataset
        if hasattr(model, 'ragon') and model.ragon:
            rag_dataset_path = config.get('rag_dataset_path')
            if rag_dataset_path:
                self.logger.info("RAG model detected, loading meta dataset...", module="BaseModelTrainer")
                try:
                    model._load_meta_for_rag()  # type: ignore
                    self.logger.info("RAG meta dataset loaded", module="BaseModelTrainer")
                except Exception as e:
                    self.logger.error(f"Failed to load RAG dataset: {str(e)}", module="BaseModelTrainer")
                    raise

    def _prepare_dataloader(self, X, y):
        self.logger.debug(f"Preparing data loader, input data shape: X={X.shape}, y={y.shape}", module="BaseModelTrainer")

        # Ensure input is a numpy array
        X = np.array(X)
        y = np.array(y)

        # Convert input data to Tensor and ensure y is a float type
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def train(self, X_train, y_train):
        self.logger.info(f"Starting model training, training data shape: X={X_train.shape}, y={y_train.shape}", module="BaseModelTrainer")
        self.model.train()

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        dataloader = self._prepare_dataloader(X_train, y_train)

        # Setting up rich progress bar
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True
        )

        # Automatically determine if it's a RAG model
        is_rag_model = hasattr(self.model, 'freeze_rag') and callable(self.model.freeze_rag) and \
                       hasattr(self.model, 'unfreeze_rag') and callable(self.model.unfreeze_rag)

        total_epochs = self.config.get('epochs', 10)
        if is_rag_model:
            self.logger.info("RAG model detected, enabling freeze/unfreeze training strategy", module="BaseModelTrainer")
            freeze_epochs = int(total_epochs * 0.8)
            unfreeze_epochs = total_epochs - freeze_epochs

            self._train_with_rag_freeze(dataloader, freeze_epochs, unfreeze_epochs, progress)
        else:
            self.logger.info("RAG model not detected, using standard training process", module="BaseModelTrainer")
            self._train_simple(dataloader, total_epochs, progress)

        self.logger.info("Model training completed", module="BaseModelTrainer")


    # ----------------------------
    # Private method: simple training process (not involving RAG)
    # ----------------------------
    def _train_simple(self, dataloader, total_epochs, progress):
        task = progress.add_task("[red]Training...", total=total_epochs)

        with progress:
            for epoch in range(total_epochs):
                total_loss = 0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Handle multi-step prediction output, adjusting model output to match the target label's shape
                    if outputs.dim() > labels.dim():
                        outputs = outputs.squeeze(-1)  # Remove the last dimension to match the shape
                        
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                progress.update(task, advance=1, description=f"[red]Epoch {epoch + 1}/{total_epochs} - Loss: {avg_loss:.4f}")
                self.logger.debug(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}", module="BaseModelTrainer")

    # ----------------------------
    # Private method: training process with RAG freeze/unfreeze
    # ----------------------------
    def _train_with_rag_freeze(self, dataloader, freeze_epochs, unfreeze_epochs, progress):
        # Stage 1: Freezing RAG and training the backbone network
        self.logger.info(f"Stage 1 training: Freezing RAG module for {freeze_epochs} epochs", module="BaseModelTrainer")
        self.freeze_rag()
        task_freeze = progress.add_task(f"[blue]Freezing RAG ({freeze_epochs} epochs)...", total=freeze_epochs)
        
        with progress:
            for epoch in range(freeze_epochs):
                total_loss = 0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Handle multi-step prediction output
                    if outputs.dim() > labels.dim():
                        outputs = outputs.squeeze(-1)  # Remove the last dimension to match the shape
                        
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                progress.update(task_freeze, advance=1, description=f"[blue]Epoch {epoch + 1}/{freeze_epochs} - Loss: {avg_loss:.4f}")
                self.logger.debug(f"RAG Freeze Epoch {epoch + 1}, Loss: {total_loss:.4f}", module="BaseModelTrainer")

        # Stage 2: Unfreezing RAG and joint training
        self.logger.info(f"Stage 2 training: Unfreezing RAG module and joint training for {unfreeze_epochs} epochs", module="BaseModelTrainer")
        self.unfreeze_rag()
        task_unfreeze = progress.add_task(f"[green]Unfreezing RAG & Joint Training ({unfreeze_epochs} epochs)...", total=unfreeze_epochs)

        with progress:
            for epoch in range(unfreeze_epochs):
                total_loss = 0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Handle multi-step prediction output
                    if outputs.dim() > labels.dim():
                        outputs = outputs.squeeze(-1)  # Remove the last dimension to match the shape
                        
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                progress.update(task_unfreeze, advance=1, description=f"[green]Epoch {epoch + 1}/{unfreeze_epochs} - Loss: {avg_loss:.4f}")
                self.logger.debug(f"RAG Unfreeze Epoch {epoch + 1}, Loss: {total_loss:.4f}", module="BaseModelTrainer")


    def evaluate(self, X_test, y_test):
        self.logger.info(f"Starting model evaluation, testing data shape: X={X_test.shape}, y={y_test.shape}", module="BaseModelTrainer")
        self.model.eval()

        with torch.no_grad():
            inputs, labels = next(iter(self._prepare_dataloader(X_test, y_test)))
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            
            # Handle multi-step prediction output
            if outputs.dim() > labels.dim():
                outputs = outputs.squeeze(-1)  # Remove the last dimension to match the shape

            preds = outputs.cpu().numpy()
            true = labels.cpu().numpy()

            rmse = np.sqrt(mean_squared_error(true, preds))
            r2 = r2_score(true, preds)

            self.logger.info(f"Model evaluation completed, RMSE: {rmse:.4f}, R²: {r2:.4f}", module="BaseModelTrainer")
            return {"rmse": rmse, "r2": r2}

    def predict(self, X):
        """Use the trained model to make predictions"""
        self.logger.info(f"Starting prediction, input data shape: X={X.shape}", module="BaseModelTrainer")
        self.model.eval()

        X = np.array(X)
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            all_preds = self.model(X_tensor)
            
            # Handle multi-step prediction output
            if all_preds.dim() > 2 and all_preds.shape[-1] == 1:
                all_preds = all_preds.squeeze(-1)  # Remove the last dimension

        self.logger.info("Prediction completed", module="BaseModelTrainer")
        return all_preds.cpu().numpy()

    def freeze_rag(self):
        """Freeze RAG related modules"""
        if hasattr(self.model, 'freeze_rag'):
            self.model.freeze_rag()  # type: ignore
            self.logger.info("RAG module frozen", module="BaseModelTrainer")

    def unfreeze_rag(self):
        """Unfreeze RAG module"""
        if hasattr(self.model, 'unfreeze_rag'):
            self.model.unfreeze_rag()  # type: ignore
            self.logger.info("RAG module unfrozen", module="BaseModelTrainer")

    def save_model(self, model_name: str, dataset_name: str):
        """
        Save the model and its configuration parameters to the specified path
        """
        os.makedirs(self.model_save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{dataset_name}_{timestamp}.pt"
        model_path = os.path.join(self.model_save_path, model_filename)

        # Extract RAG-related configuration (if it exists)
        rag_config = {}
        if hasattr(self.model, 'ragon'):
            rag_config.update({
                'ragon': self.model.ragon,
                'rag_dataset_path': self.model.rag_dataset_path,
                'window_size': self.model.window_size,
                'top_k': self.model.top_k if hasattr(self.model, 'top_k') else 4,
            })

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_class': type(self.model).__name__,
            'input_size': self.config.get('input_size'),
            'output_size': self.config.get('output_size'),
            # Save RAG configuration
            **rag_config
        }, model_path)

        self.logger.info(f"Model saved to: {model_path}", module="BaseModelTrainer")
        return model_path

    @classmethod
    def load_model(cls, model_path: str, model_class=None):
        """
        Load the model from the specified path
        """
        checkpoint = torch.load(model_path)

        if model_class is None:
            raise ValueError("Must provide the model class type or ensure the model class is parsable")

        # Construct RAG parameters
        model_kwargs = {
            'input_size': checkpoint['input_size'],
            'output_size': checkpoint['output_size']
        }

        # Add RAG parameters (if they exist)
        if 'ragon' in checkpoint:
            model_kwargs.update({
                'ragon': checkpoint['ragon'],
                'rag_dataset_path': checkpoint['rag_dataset_path'],
                'window_size': checkpoint.get('window_size', 7),
                'top_k': checkpoint.get('top_k', 4)
            })

        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])

        # If it's a RAG model and a dataset path is provided, load the meta data
        if hasattr(model, 'ragon') and model.ragon and checkpoint.get('rag_dataset_path'):
            try:
                model._load_meta_for_rag()
            except Exception as e:
                print(f"Failed to load RAG data: {str(e)}")
                raise

        config = checkpoint['config'].copy()
        config['model_path'] = model_path

        return cls(model, config)

    def get_model_load_path(self, model_name: str, dataset_name: str) -> str:
        """
        Generate the model load path based on the model name and dataset name
        """
        pattern = f"{model_name}_{dataset_name}_*.pt"
        matching_files = [f for f in os.listdir(self.model_save_path) if f.startswith(pattern.split('_')[0])]

        if not matching_files:
            raise FileNotFoundError(f"No model files found for {model_name} on {dataset_name} in {self.model_save_path}")

        full_paths = [os.path.join(self.model_save_path, f) for f in matching_files]
        return max(full_paths, key=os.path.getctime)
