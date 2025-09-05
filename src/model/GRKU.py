import os
import faiss
import numpy as np
from typing import Optional, Tuple
import torch
import pandas as pd
from torch import nn
from src.factory.model_factory import ModelFactory
from src.model.GRKULayer import GRKULayer
from src.data.MiniMaxStandardProcessor import MiniMaxStandardProcessor


@ModelFactory.register("grku_time_series_model")
class GRKU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=1,
        num_layers=1,
        seq_length=None,
        forecast_steps=1,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
        ragon=False,
        rag_dataset_path: Optional[str] = None,
        window_size=None,
        top_k=4,
    ):
        super(GRKU, self).__init__()
        self.ragon = ragon
        self.rag_dataset_path = rag_dataset_path
        # Use seq_length if window_size is not specified
        self.window_size = window_size if window_size is not None else seq_length
        if self.window_size is None:
            raise ValueError("window_size or seq_length must be specified")
            
        self.top_k = top_k
        self.forecast_steps = forecast_steps
        self.output_dim = output_dim

        # Initialize GRKU layer and output layer
        self.GRKULayer = GRKULayer(
            input_dim, hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        # Modify the output layer to support multi-step forecasting
        self.fc_out = nn.Linear(hidden_dim, forecast_steps * output_dim)

        # New: RAG fusion layer (supports multi-step forecasting)
        # Note: Adjusted input dimension from top_k to top_k * forecast_steps
        self.rag_fusion_layer = nn.Linear(top_k * forecast_steps, forecast_steps * output_dim)
        nn.init.zeros_(self.rag_fusion_layer.weight)  # Initialize weights to 0
        nn.init.zeros_(self.rag_fusion_layer.bias)    # Initialize bias to 0

        # Modify the gate network to support multi-step forecasting
        self.gate_network = nn.Sequential(
            nn.Linear(self.window_size, 32),
            nn.Tanh(),
            nn.Linear(32, forecast_steps * output_dim),
            nn.Sigmoid()
        )

        # If RAG is enabled, load and process the meta dataset for retrieval
        self.meta_sequences = None
        self.meta_labels = None  # Store true runoff for future multi-step timesteps
        if self.ragon and self.rag_dataset_path:
            self._load_meta_for_rag()

        # Flag to indicate whether the RAG module is frozen
        self.rag_frozen = False

    def freeze_rag(self):
        """Freezes RAG-related modules"""
        if self.ragon and hasattr(self, 'meta_sequences'):
            for name, param in self.named_parameters():
                if 'rag_' in name:
                    param.requires_grad = False
            self.rag_frozen = True
        else:
            print("RAG not enabled or not loaded")

    def unfreeze_rag(self):
        """Unfreezes RAG modules"""
        if self.ragon and hasattr(self, 'meta_sequences'):
            for name, param in self.named_parameters():
                if 'rag_' in name:
                    param.requires_grad = True
            self.rag_frozen = False

    def is_rag_frozen(self):
        return self.rag_frozen

    def _load_meta_for_rag(self):
        """Loads and processes the meta dataset (supports merging multiple CSVs) for RAG retrieval"""
        if not os.path.isdir(self.rag_dataset_path):  # type: ignore
            raise ValueError(f"The provided path is not a valid directory: {self.rag_dataset_path}")

        if not self.rag_dataset_path:
            raise ValueError("rag_dataset_path is not set")
        csv_files = [os.path.join(self.rag_dataset_path, f) for f in os.listdir(self.rag_dataset_path) if f.endswith('.csv')]
        
        # Read and merge all CSV files
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)

        meta_df = pd.concat(dfs, ignore_index=True)

        # Standardize processing
        meta_df = MiniMaxStandardProcessor().process(meta_df)
        print(meta_df.head())

        if 'runoffs' not in meta_df.columns:
            raise KeyError("Missing 'runoffs' column in meta_dataset_path")

        runoff_data = torch.tensor(meta_df['runoffs'].values, dtype=torch.float32)

        # Construct historical window sequence [T, window_size]
        self.meta_sequences = self._apply_time_window(runoff_data)

        # Construct corresponding multi-step runoff values for the future [T, forecast_steps]
        self.meta_labels = self._create_multi_step_labels(runoff_data, self.forecast_steps)
        
        min_len = min(len(self.meta_sequences), len(self.meta_labels))  
        self.meta_sequences = self.meta_sequences[:min_len]
        self.meta_labels = self.meta_labels[:min_len]
        self._build_faiss_index()

    def _build_faiss_index(self):
        """Builds FAISS index for accelerated retrieval"""
        if self.meta_sequences is None:
            raise ValueError("meta_sequences not yet initialized")

        data = self.meta_sequences.numpy().astype(np.float32)
        dimension = data.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(data)  # type: ignore

    def _apply_time_window(self, data: torch.Tensor) -> torch.Tensor:
        windowed_data = []
        for i in range(len(data) - self.window_size + 1):
            windowed_data.append(data[i:i + self.window_size])
        return torch.stack(windowed_data)
    
    def _create_multi_step_labels(self, data: torch.Tensor, steps: int) -> torch.Tensor:
        """Creates multi-step forecasting labels"""
        multi_step_labels = []
        for i in range(len(data) - self.window_size - steps + 1):
            multi_step_labels.append(data[i + self.window_size : i + self.window_size + steps])
        return torch.stack(multi_step_labels)

    def retrieve_similar_samples(self, query_vector, top_k=4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uses FAISS to quickly find the top_k most similar samples to the query_vector
        Returns (indices, labels)
        """
        query = query_vector.unsqueeze(0).cpu().numpy().astype(np.float32)
        distances, indices = self.index.search(query, top_k)  # type: ignore

        labels = self.meta_labels[torch.from_numpy(indices[0])]  # type: ignore # [K, forecast_steps]
        return torch.tensor(indices), labels.unsqueeze(0)  # [1, K, forecast_steps]
    
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        rag_references = None
        rag_contribution = None
        
        if self.ragon and hasattr(self, 'meta_sequences') and self.meta_sequences is not None:
            # Ensure the correct window size is used
            if x.size(1) != self.window_size:
                # If the input sequence length doesn't match the window size, use the last window_size timesteps
                query_features = x[:, -self.window_size:, 3].detach().cpu().numpy().astype(np.float32)  # [B, window_size]
            else:
                # If it matches, use all timesteps directly
                query_features = x[:, :, 3].detach().cpu().numpy().astype(np.float32)  # [B, window_size]
            
            # Retrieve top_k samples
            _, indices = self.index.search(query_features, self.top_k)  # [B, top_k]

            # Get corresponding multi-step labels
            rag_labels = self.meta_labels[indices]  # [B, top_k, forecast_steps]
            rag_references = rag_labels.detach().clone().float().requires_grad_(False).to(x.device)

            # Compute gate weights (using the corresponding window)
            if x.size(1) != self.window_size:
                gate_input = x[:, -self.window_size:, 3]  # [B, window_size]
            else:
                gate_input = x[:, :, 3]  # [B, window_size]
                
            gate_weights = self.gate_network(gate_input)  # [B, forecast_steps * output_dim]
            gate_weights = gate_weights.view(batch_size, self.forecast_steps, self.output_dim)  # [B, forecast_steps, output_dim]

            # RAG contribution calculation (supports multi-step output)
            # Note: Adjust input dimension to match the new rag_fusion_layer
            rag_features = rag_references.view(batch_size, -1)  # [B, top_k * forecast_steps]
            rag_contribution = self.rag_fusion_layer(rag_features)  # [B, forecast_steps * output_dim]
            rag_contribution = rag_contribution.view(batch_size, self.forecast_steps, self.output_dim)  # [B, forecast_steps, output_dim]
            
            # Apply gate weights
            rag_contribution = (gate_weights * rag_contribution).view(batch_size, -1)  # [B, forecast_steps * output_dim]

        # GRKU layer processing
        grku_out, h_n = self.GRKULayer(x)
        
        # Main output (supports multi-step)
        out = self.fc_out(h_n)  # [B, forecast_steps * output_dim]

        # If there is RAG output, perform fusion
        if rag_contribution is not None:
            out = out + rag_contribution  # Additive fusion

        # Reshape the output to accommodate multi-step forecasting
        out = out.view(batch_size, self.forecast_steps, self.output_dim)  # [B, forecast_steps, output_dim]
        
        return out.view(batch_size, -1)  # [B, forecast_steps * output_dim]
