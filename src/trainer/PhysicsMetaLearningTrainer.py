from matplotlib import pyplot as plt
import numpy as np
import torch
import pandas as pd
from src.factory.trainer_factory import TrainerFactory
from src.trainer.base_model_trainer import BaseModelTrainer
from torch import nn
from typing import Any, Tuple
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from src.data.MiniMaxStandardProcessor import MiniMaxStandardProcessor
from src.utils.time_window_split import TimeWindowSplitter

# Import MAML implementation from learn2learn
from learn2learn.algorithms import MAML

@TrainerFactory.register("PhysicsMetaLearningTrainer")
class PhysicsMetaLearningTrainer(BaseModelTrainer):
    def __init__(self, model: nn.Module, config):
        super().__init__(model, config)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Physics constraint parameters
        self.physics_weight = config.get('physics_weight', 0.75)

        # Meta-learning parameters
        self.meta_learning_rate = config.get('meta_learning_rate', 0.01)
        self.inner_update_steps = config.get('inner_update_steps', 5)
        
        # Wrap model with learn2learn's MAML
        self.maml = MAML(model.to(self.device), lr=self.meta_learning_rate)
        self.meta_optimizer = torch.optim.Adam(self.maml.parameters(), lr=self.meta_learning_rate, weight_decay=1e-4)

        # Time window & normalizer
        self.window_size = config.get('window_size', 7)
        self.scaler = MinMaxScaler()
        self.criterion = nn.MSELoss()

        # Dataset paths
        self.meta_dataset_path = config.get('meta_dataset_path')
        self.physics_dataset_path = config.get('physics_dataset_path')
        self.target_col = config.get('target_column', 'target')

        # Initialize meta_model_path
        self.meta_model_path = config.get('meta_model_path', './meta_models')
        os.makedirs(self.meta_model_path, exist_ok=True)

        self.fine_tune_lr = config.get('fine_tune_lr', 0.001)
        self.fine_tune_steps = self.config.get('fine_tune_steps', 5)

        # Forecast horizon
        self.forecast_horizon = config.get('forecast_horizon', 1)

        # Load or save meta-model
        self._load_or_save_meta_model()
        self.fine_tuned_model = None

        torch.backends.cudnn.benchmark = True

        if hasattr(self.maml.module, 'freeze_rag'):
            self.maml.module.freeze_rag() # type: ignore # ✅ Correct operation

        if not self.loaded_existing_model:
            # Pre-train with meta-learning on multiple sites
            meta_dir = self.config.get('meta_dataset_dir')
            physics_dir = self.config.get('physics_dataset_dir')

            if not meta_dir or not physics_dir:
                raise ValueError("Missing meta_dataset_dir or physics_dataset_dir configuration.")

            self.meta_learning_train(meta_dir, physics_dir)
            self._save_meta_model(f'meta_model_initial.pt')
            self._load_meta_model(os.path.join(self.meta_model_path, 'meta_model_initial.pt'))

    def predict(self, X):
        """
        Use the fine-tuned maml.module for prediction.
        """
        self.logger.info(f"Starting prediction, input data shape: X={X.shape}", module="PhysicsMetaLearningTrainer")
        if self.fine_tuned_model is None:
            raise RuntimeError("Please call train() to fine-tune the model before predicting.")
        model = self.fine_tuned_model.eval()

        # Ensure input is a numpy array
        X = np.array(X)

        # Move data to device
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            all_preds = model(X_tensor)

        self.logger.info("Prediction complete", module="PhysicsMetaLearningTrainer")

        # Move results back to CPU and convert to numpy array
        return all_preds.cpu().numpy()

    def _load_or_save_meta_model(self):
        """Loads an existing meta-model or saves an initial one."""
        meta_model_files = [f for f in os.listdir(self.meta_model_path) if f.startswith('meta_model_')]
        self.loaded_existing_model = False

        if meta_model_files:
            latest_file = max(
                [os.path.join(self.meta_model_path, f) for f in meta_model_files],
                key=os.path.getctime
            )
            self._load_meta_model(latest_file)
            self.loaded_existing_model = True
        else:
            self.logger.info("No existing meta-model found, preparing to pre-train with physics constraints and meta-datasets...", module="PhysicsMetaLearningTrainer")

    def _load_meta_model(self, file_path: str):
        """Loads the meta-model."""
        self.logger.info(f"Loading meta-model from: {file_path}", module="PhysicsMetaLearningTrainer")
        checkpoint = torch.load(file_path, map_location=self.device)
        loaded_hash = hash(tuple(p.data.cpu().numpy().tobytes() for p in checkpoint['model_state_dict'].values()))
        self.logger.debug(f"[Load] Loaded model hash: {loaded_hash}")
        self.maml.module.load_state_dict(checkpoint['model_state_dict'])
        self.config.update(checkpoint['config'])

    def _save_meta_model(self, filename: str):
        """Saves the meta-model."""
        file_path = os.path.join(self.meta_model_path, filename)
        torch.save({
            'model_state_dict': self.maml.module.state_dict(),
            'config': self.config,
            'model_class': type(self.maml.module).__name__
        }, file_path)
        self.logger.info(f"Meta-model saved to: {file_path}", module="PhysicsMetaLearningTrainer")
        current_hash = self._get_model_hash(self.maml.module)
        self.logger.debug(f"[Save] Saved model hash: {current_hash}", module="PhysicsMetaLearningTrainer")

    def _apply_time_window(self, data: torch.Tensor) -> torch.Tensor:
        windowed_data = []
        for i in range(len(data) - self.window_size + 1):
            windowed_data.append(data[i:i + self.window_size])
        return torch.stack(windowed_data)

    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        normalized_data = self.scaler.fit_transform(data.numpy())
        return torch.tensor(normalized_data, dtype=torch.float32)

    def _prepare_task_data_batch(self, batch):
        train_inputs, train_labels, test_inputs, test_labels = batch

        if train_inputs.dim() == 2:
            train_inputs = train_inputs.unsqueeze(1)
        if test_inputs.dim() == 2:
            test_inputs = test_inputs.unsqueeze(1)

        # Ensure label dimensions are correct
        if train_labels.dim() == 2:
            train_labels = train_labels.unsqueeze(1)
        if test_labels.dim() == 2:
            test_labels = test_labels.unsqueeze(1)

        return (
            train_inputs.to(self.device),
            train_labels.to(self.device),
            test_inputs.to(self.device),
            test_labels.to(self.device)
        )

    def meta_learning_train(self, meta_dir, physics_dir):
        """
        Meta-learning main training loop: Load multiple sites -> Build MetaBatch -> MAML train update.
        """
        tasks = self.load_all_sites_tasks(meta_dir, physics_dir)

        criterion = self.criterion
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True
        )

        total_epochs = self.config.get('meta_epochs', 10)
        with progress:
            task_epoch = progress.add_task("[red]Meta Epoch Progress", total=total_epochs)

            for epoch in range(total_epochs):
                total_loss = 0.0
                phy_loss = 0.0
                task_loss = 0.0
                task_batch = progress.add_task(f"[green]Epoch {epoch+1} - Training", total=len(tasks))

                self.maml.train()
                for site_idx, (X_support, y_support, X_query, y_query) in enumerate(tasks):

                    X_support, y_support, X_query, y_query = self._prepare_task_data_batch((X_support, y_support, X_query, y_query))
                    learner = self.maml.clone()

                    # Inner-loop: Fast adaptation, use primary task target (first column)
                    for _ in range(self.inner_update_steps):
                        out = learner(X_support)
                        # Ensure output and target dimensions match
                        # Model output: [batch_size, forecast_horizon]
                        # Target: [batch_size, forecast_horizon, 2] (last dim: 0=main task, 1=physics constraint)
                        if out.dim() == 2 and y_support.dim() == 3 and y_support.shape[2] == 2:
                            # Use primary task target for inner-loop training
                            meta_target = y_support[:, :, 0]
                            loss = criterion(out, meta_target)
                        else:
                            # Other cases, try to adapt
                            meta_target = y_support[:, :, 0] if y_support.dim() == 3 else y_support[:, 0]
                            if out.dim() != meta_target.dim():
                                if out.dim() > meta_target.dim():
                                    meta_target = meta_target.unsqueeze(-1)
                                else:
                                    out = out.squeeze(-1) if out.dim() > meta_target.dim() else out
                            loss = criterion(out, meta_target)
                            
                        learner.adapt(loss, allow_unused=True, first_order=True, allow_nograd=True)

                    # Outer-loop: Query set loss used to update meta-parameters
                    pred_query = learner(X_query)
                    
                    # Primary task loss (using primary task target)
                    if pred_query.dim() == 2 and y_query.dim() == 3 and y_query.shape[2] == 2:
                        meta_target = y_query[:, :, 0]
                        physics_target = y_query[:, :, 1]
                        
                        query_loss = criterion(pred_query, meta_target)
                        physics_loss = criterion(pred_query, physics_target)
                    else:
                        # Other cases, try to adapt
                        meta_target = y_query[:, :, 0] if y_query.dim() == 3 else y_query[:, 0]
                        physics_target = y_query[:, :, 1] if y_query.dim() == 3 and y_query.shape[2] > 1 else meta_target
                        
                        # Adjust dimensions to match prediction output
                        if pred_query.dim() != meta_target.dim():
                            if pred_query.dim() > meta_target.dim():
                                meta_target = meta_target.unsqueeze(-1)
                                physics_target = physics_target.unsqueeze(-1) if physics_target.dim() < pred_query.dim() else physics_target
                            else:
                                pred_query = pred_query.squeeze(-1)
                        
                        query_loss = criterion(pred_query, meta_target)
                        physics_loss = criterion(pred_query, physics_target)

                    total_loss_value = (1 - self.physics_weight) * query_loss + self.physics_weight * physics_loss
                    phy_loss += self.physics_weight * physics_loss.item()
                    task_loss += (1 - self.physics_weight) * query_loss.item()

                    self.meta_optimizer.zero_grad()
                    total_loss_value.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.maml.parameters(), max_norm=1.0)
                    self.meta_optimizer.step()

                    total_loss += total_loss_value.item()
                    progress.update(task_batch, advance=1)

                avg_loss = total_loss / len(tasks)
                avg_phy_loss = phy_loss / len(tasks)
                avg_task_loss = task_loss / len(tasks)
                progress.update(task_epoch, advance=1,
                                 description=f"[red]Epoch {epoch+1}/{total_epochs} | Avg Loss: {avg_loss:.4f} | Task Loss: {avg_task_loss:.4f} | Phy Loss: {avg_phy_loss:.4f}")
                self.logger.debug(f"Meta-learning Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

        # Save final meta-model
        self._save_meta_model("meta_model_final.pt")


    def tensor_hash(self, tensor: torch.Tensor) -> str:
        """
        Generates a hash value for a tensor.

        Parameters:
            tensor (torch.Tensor): The tensor to be hashed.

        Returns:
            int: The hash value.
        """
        # Convert tensor data to an immutable tuple, combined with device info to differentiate tensors on different devices
        data = tensor.cpu().numpy()
        data_tuple = tuple(tensor.cpu().numpy().flatten().tolist())
        device = tensor.device
        statistics = {
            "count": int(data.size),
            "mean": float(np.mean(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "std": float(np.std(data)),
            "var": float(np.var(data)),
            "hash": hash((data_tuple, device))
        }
        return statistics.__str__()
    def _get_model_hash(self, model):
        return hash(tuple(p.data.cpu().numpy().tobytes() for p in model.parameters()))

    def load_all_sites_tasks(self, meta_data_dir: str, physics_data_dir: str):
        """
        Loads tasks (support/query) for all sites, returning a list of meta-learning tasks.

        Parameters:
            meta_data_dir (str): Directory containing CSV files with meta-features.
            physics_data_dir (str): Directory containing CSV files with physics constraint targets.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
                Each task is a quadruple: (train_inputs, train_labels_with_physics, test_inputs, test_labels_with_physics)
        """
        # Get list of file paths
        meta_file_paths = sorted([os.path.join(meta_data_dir, f) for f in os.listdir(meta_data_dir) if f.endswith('.csv')])
        physics_file_paths = sorted([os.path.join(physics_data_dir, f) for f in os.listdir(physics_data_dir) if f.endswith('.csv')])

        assert len(meta_file_paths) == len(physics_file_paths), "The number of meta and physics files must be consistent."

        all_tasks = []

        for meta_file_path, physics_file_path in zip(meta_file_paths, physics_file_paths):
            site_name = os.path.splitext(os.path.basename(meta_file_path))[0]
            self.logger.info(f"Processing site: {site_name}", module="PhysicsMetaLearningTrainer")

            # Load data
            meta_features_df = pd.read_csv(meta_file_path)
            physics_targets_df = pd.read_csv(physics_file_path)

            # Sort by column names (ensure consistent feature order)
            meta_features_df = meta_features_df.loc[:, sorted(meta_features_df.columns)]
            physics_targets_df = physics_targets_df.loc[:, sorted(physics_targets_df.columns)]

            # Normalizers
            feature_scaler = MiniMaxStandardProcessor()
            target_scaler = MiniMaxStandardProcessor()

            # Normalize meta features and physics targets separately
            scaled_meta_features = feature_scaler.process(meta_features_df)
            scaled_physics_target = target_scaler.process(physics_targets_df[[self.target_col]])

            # Construct target data, ensuring clear column order and naming
            combined_target_df = pd.DataFrame({
                'meta_target': scaled_meta_features[self.target_col],
                'physics_target': scaled_physics_target[self.target_col]
            })

            time_splitter = TimeWindowSplitter.from_config(self.config)
            window_size = time_splitter.window_size

            X_train_windowed, y_train_windowed, X_test_windowed, y_test_windowed = time_splitter.split(
                X=scaled_meta_features,
                y=combined_target_df,
                window=window_size,
                test_size=0.2
            )

            # Convert to Tensor
            train_features_tensor = torch.tensor(X_train_windowed, dtype=torch.float32)
            train_combined_targets = torch.tensor(y_train_windowed, dtype=torch.float32)
            test_features_tensor = torch.tensor(X_test_windowed, dtype=torch.float32)
            test_combined_targets = torch.tensor(y_test_windowed, dtype=torch.float32)

            # Validate target data structure
            if train_combined_targets.dim() == 3 and train_combined_targets.shape[-1] == 2:
                self.logger.debug(f"Training target data shape: {train_combined_targets.shape} (format: [batch, horizon, 2])",
                                  module="PhysicsMetaLearningTrainer")
            else:
                self.logger.warning(f"Training target data shape is abnormal: {train_combined_targets.shape}",
                                  module="PhysicsMetaLearningTrainer")

            # Build task
            all_tasks.append((
                train_features_tensor,
                train_combined_targets,
                test_features_tensor,
                test_combined_targets
            ))

        return all_tasks
    def construct_meta_batch(self, tasks, batch_size=4):
        indices = np.random.choice(len(tasks), size=batch_size, replace=False)
        batch_tasks = [tasks[i] for i in indices]
        return batch_tasks

    def plot_train_results(self, X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, title: str = "Train Results"):
        """
        Plots a comparison of true and predicted values during the training phase.
        
        :param X: Input features (time series)
        :param y_true: True target values (torch.Tensor)
        :param y_pred: Model predicted values (torch.Tensor)
        :param title: Plot title
        """
        # Convert to numpy for plotting
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy() # type: ignore
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy() # type: ignore
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy() # type: ignore # Add detach() to fix the error

        plt.figure(figsize=(12, 4))
        plt.plot(y_true, label='True Value', color='blue')
        plt.plot(y_pred, label='Predicted Value', color='red', linestyle='--')
        
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True)

        # Save plot to log path or show
        if self.config.get('save_plot_path'):
            os.makedirs(self.config['save_plot_path'], exist_ok=True)
            plot_path = os.path.join(self.config['save_plot_path'], f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path)
            self.logger.info(f"Plot saved to: {plot_path}", module="PhysicsMetaLearningTrainer")
        else:
            plt.show()

        plt.close()

    def save_model(self, model_name: str, dataset_name: str) -> str:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"models/{model_name}_{dataset_name}_{timestamp}.pt"
        torch.save(self.maml.module.state_dict(), file_path)
        return file_path

    def _process_batch(self, batch):
        physics_inputs, physics_targets, meta_tasks = batch
        return (
            physics_inputs.to(self.device),
            physics_targets.to(self.device),
            meta_tasks
        )

    def train(self, X_train, y_train):
        """
        Fine-tunes the meta-model using target task data, supporting batch training with DataLoader.
        
        :param X_train: Input features (np.ndarray or torch.Tensor)
        :param y_train: Target labels (np.ndarray or torch.Tensor)
        """
        self.maml.module.train()
        current_hash = self._get_model_hash(self.maml.module)
        self.logger.debug(f"[Train] Model hash: {current_hash}", module="PhysicsMetaLearningTrainer")
        criterion = self.criterion

        # Unify inputs as Tensors and prepare DataLoader
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        self.logger.debug(f"[Train] Training data shape: {X_train.shape}")
        self.logger.debug(f"[Train] Training target shape: {y_train.shape}")

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        self.logger.debug(f"Fine-tuning {self.fine_tune_steps}")
        batch_size = self.config.get('batch_size', 16)

        # Use the unified DataLoader provided by BaseModelTrainer
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create differentiable learner
        learner = self.maml.clone()

        if hasattr(learner.module, 'unfreeze_rag'):
            learner.module.unfreeze_rag() # type: ignore # ✅ Safe operation (no need to preserve historical gradients during fine-tuning)

        # Set up rich progress bar
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True
        )

        with progress:
            task_epoch = progress.add_task("[green]Fine-tuning Epoch", total=self.fine_tune_steps)

            for step in range(self.fine_tune_steps):
                total_loss = 0.0

                task_batch = progress.add_task(
                    f"[yellow]Epoch {step + 1}: Processing Batches",
                    total=len(dataloader)
                )

                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward & Loss
                    outputs = learner(inputs)
                    
                    loss = criterion(outputs, labels)

                    # Update parameters (supports parameters not involved in computation)
                    learner.adapt(loss, allow_unused=True, first_order=True)
                    total_loss += loss.item()
                    progress.update(task_batch, advance=1)
                
                avg_loss = total_loss / len(dataloader)
                progress.update(task_epoch, advance=1,
                                 description=f"[green]Step {step+1}/{self.fine_tune_steps} - Avg Loss: {avg_loss:.4f}")
                self.logger.debug(f"Fine-tuning Step {step + 1}, Avg Loss: {avg_loss:.4f}")

        # Ensure prediction dimensions are correct for plotting
        final_predictions = learner(X_train)
        self.plot_train_results(X_train, y_train, final_predictions, title="Post-Training Predictions")
        self.fine_tuned_model = learner.module
