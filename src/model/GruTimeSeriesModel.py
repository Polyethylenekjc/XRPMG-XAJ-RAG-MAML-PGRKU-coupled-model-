import torch
import torch.nn as nn
from ..factory.model_factory import ModelFactory

@ModelFactory.register("gru_time_series_model")
class GruTimeSeriesModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=1, seq_length=7, forecast_steps=3):
        super(GruTimeSeriesModel, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for output prediction
        self.fc = nn.Linear(hidden_dim, forecast_steps)

    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_length, hidden_dim]
        # Take the hidden state of the last timestep as the decoding input
        last_hidden_state = gru_out[:, -1, :]  # [batch_size, hidden_dim]
        # Output multi-step prediction
        out = self.fc(last_hidden_state)  # [batch_size, forecast_steps]
        return out

    @classmethod
    def from_config(cls, config):
        valid_params = {
            'input_dim': config.get('input_dim', 32),
            'hidden_dim': config.get('hidden_dim', 64),
            'num_layers': config.get('num_layers', 1),
            'seq_length': config.get('seq_length', 7),
            'forecast_steps': config.get('forecast_steps', 1)
        }
        print(f"Initializing GruTimeSeriesModel with:\n{valid_params}")
        return cls(**valid_params)
