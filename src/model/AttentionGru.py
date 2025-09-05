import torch
import torch.nn as nn
from ..factory.model_factory import ModelFactory


@ModelFactory.register("attention_gru")
class AttentionGRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        forecast_steps: int = 1,
        dropout: float = 0.1
    ):
        super(AttentionGRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism: compute attention weights for each time step
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Output layer (fully connected)
        self.fc_out = nn.Linear(hidden_dim, forecast_steps)

    def attention_forward(self, gru_outputs):
        # gru_outputs: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(gru_outputs)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * gru_outputs, dim=1)  # (batch, hidden_dim)
        return context_vector, attention_weights

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        gru_out, hidden = self.gru(x)  # gru_out: (batch, seq_len, hidden_dim)

        # Apply attention mechanism
        context_vector, attn_weights = self.attention_forward(gru_out)  # (batch, hidden_dim)

        # Output prediction
        output = self.fc_out(context_vector)  # (batch, forecast_steps)

        return output

    @classmethod
    def from_config(cls, config):
        input_dim = config.get('input_dim', 10)
        hidden_dim = config.get('hidden_dim', 64)
        num_layers = config.get('num_layers', 1)
        forecast_steps = config.get('forecast_steps', 1)
        dropout = config.get('dropout', 0.1)

        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_steps=forecast_steps,
            dropout=dropout
        )
