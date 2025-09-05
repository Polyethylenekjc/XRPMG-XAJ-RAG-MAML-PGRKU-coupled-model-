import torch
import torch.nn as nn
from ..factory.model_factory import ModelFactory

@ModelFactory.register("vmdi_lstm_ed_model")
class VMDILSTMED(nn.Module):
    """
    VMDI-LSTM-ED model implementation (simplified version of y_p_star).

    This version assumes that y_p_star (lag-1 observation) can be directly obtained from the input features x.
    The user needs to specify which feature index in the configuration corresponds to the historical target variable (e.g., 'runoffs').
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        seq_length,
        hidden_dim,
        num_layers,
        forecast_steps=1,
        target_feature_index=None,  # New parameter
        dropout=0.0  # Optional: Add Dropout
    ):
        """
        Initializes the model.

        Args:
            input_dim (int): The dimension of the original input features.
            output_dim (int): Output dimension.
            seq_length (int): The length of the input sequence.
            hidden_dim (int): The dimension of the LSTM hidden layer.
            num_layers (int): The number of LSTM layers.
            forecast_steps (int): The number of forecast steps.
            target_feature_index (int, optional): The index in the input features that represents the historical target variable.
                                             If None, data injection (DI) is not performed.
            dropout (float): Dropout probability.
        """
        super(VMDILSTMED, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.target_feature_index = target_feature_index
        self.dropout = dropout

        # DI module: Linear transformation + ReLU
        # The input is the original feature x and the extracted y_p_star (1 extra feature)
        if self.target_feature_index is not None:
            self.di_input_dim = self.input_dim + 1
            self.di_layer = nn.Sequential(
                nn.Linear(self.di_input_dim, self.input_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)  # Optional
            )
        else:
            # If target_feature_index is not specified, the DI layer is an identity transform (i.e., no DI is performed)
            self.di_layer = nn.Identity()

        # LSTM layer
        # The input dimension is the dimension after DI processing (theoretically should be consistent with input_dim if DI outputs input_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout if num_layers > 1 else 0.0  # Only applied to non-last layers if num_layers > 1
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, forecast_steps * output_dim),
            nn.Linear(forecast_steps * output_dim, forecast_steps)
            # Can be added if the output requires a specific activation (e.g., ReLU to ensure non-negative runoff)
            # nn.ReLU() 
        )

    def forward(self, x):
        """
        Model forward pass.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Model output, shape (batch_size, forecast_steps * output_dim).
        """
        batch_size = x.size(0)

        x_transformed = x  # Initialize

        # --- Data Injection (DI) Module ---
        if self.target_feature_index is not None:
            # Extract the historical target variable sequence from input x as an approximation of y_p_star
            # The shape of x is (B, L, D), and we extract a specific feature of (B, L)
            # .unsqueeze(-1) changes it to (B, L, 1) for concatenation
            y_p_star_approx = x[:, :, self.target_feature_index].unsqueeze(-1)  # (B, L, 1)

            # Concatenate the original input x with the extracted y_p_star approximation
            di_input = torch.cat((x, y_p_star_approx), dim=-1)  # (B, L, D+1)

            # Apply the DI transformation
            x_transformed = self.di_layer(di_input)  # (B, L, D)
            # Note: This assumes the output dimension of self.di_layer is the same as the last dimension of the input x (D)

        # --- LSTM Processing ---
        lstm_out, _ = self.lstm(x_transformed)  # lstm_out: (B, L, H)

        # Take the output of the last timestep for prediction (a simplification of the Encoder-Decoder architecture)
        final_lstm_output = lstm_out[:, -1, :]  # (B, H)

        # --- Output Layer ---
        output = self.output_layer(final_lstm_output)  # (B, forecast_steps * output_dim)

        return output

    @classmethod
    def from_config(cls, config):
        """
        Create a model instance from a configuration dictionary.
        """
        params = config.get('params', {})
        
        # Basic parameters
        input_dim = params.get('input_dim', 10)
        output_dim = params.get('output_dim', 1)
        seq_length = params.get('seq_length', 15)
        hidden_dim = params.get('hidden_dim', 64)
        num_layers = params.get('num_layers', 2)
        forecast_steps = params.get('forecast_steps', 1)
        dropout = params.get('dropout', 0.0)  # Read Dropout from the configuration

        # Key parameter: Specify the index of the historical target variable in the input features
        # For features ['d2m', 'e', 'pev', 'runoffs', 'skt', 'slhf', 'src', 'ssro', 'str', 'tp']
        # 'runoffs' is the 3rd one (0-indexed)
        target_feature_index = params.get('target_feature_index', None)
        # It is strongly recommended to explicitly set this value in the configuration instead of relying on the default value of None

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_steps=forecast_steps,
            target_feature_index=target_feature_index,
            dropout=dropout
        )

# --- Corresponding Configuration File Example ---
# Please add this configuration block to the appropriate location in your configuration file
"""
vmdi_lstm_ed_model:
  enabled: true
  params:
    forecast_steps: 1         # Number of forecast steps
    hidden_dim: 64            # LSTM hidden layer dimension
    input_dim: 10             # Input feature dimension (matches the features list)
    num_layers: 2             # Number of LSTM layers
    seq_length: 15            # Input sequence length
    target_feature_index: 3   # The index of 'runoffs' in the feature list (0-indexed)
    dropout: 0.2              # Dropout probability (optional)
    output_dim: 1             # Output dimension (usually 1, for predicting a single runoff value)
  trainer:
    batch_size: 32
    epochs: 20
    learning_rate: 0.001
    type: BaseModelTrainer  # or your custom trainer type
  type: vmdi_lstm_ed_model  # Corresponds to the name registered in ModelFactory
"""

# --- Usage Instructions ---
# 1. Ensure your data preprocessing logic includes 'runoffs' (or other historical target variables) as a feature in the input tensor x.
# 2. In the configuration file, set `target_feature_index` to the correct index of 'runoffs' in the feature list (which is 3 in this case).
# 3. Your training and inference code can remain unchanged; call `model(x)` just as you would any other model.
#    The model will automatically extract 'runoffs' from x and perform data injection.
# 4. If `target_feature_index` is set to None or omitted in the configuration, the model degenerates into a standard LSTM without DI.
