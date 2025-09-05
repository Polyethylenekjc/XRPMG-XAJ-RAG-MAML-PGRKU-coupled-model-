import torch
import torch.nn as nn
from ..factory.model_factory import ModelFactory

@ModelFactory.register("simple_pytorch_model")
class SimplePyTorchModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, forecast_steps=1):
        """
        Initializes the model.

        Args:
            input_dim (int): The number of features at each timestep.
            output_dim (int): The output dimension for each forecast step.
            seq_length (int): The length of the input sequence.
            forecast_steps (int): The number of steps to forecast.
        """
        super(SimplePyTorchModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.forecast_steps = forecast_steps
        
        # Calculate the flattened input dimension
        flattened_input_dim = seq_length * input_dim
        
        # Calculate the output dimension
        final_output_dim = forecast_steps * output_dim

        # Define the neural network layers
        # First, flatten the input, then pass it through one or more fully connected layers
        self.model = nn.Sequential(
            # Flattening operation will be performed in the forward pass
            nn.Linear(flattened_input_dim, 128), # Hidden layer size can be adjusted
            nn.ReLU(),
            nn.Linear(128, 64), # Hidden layers can be added or removed
            nn.ReLU(),
            nn.Linear(64, final_output_dim),
            # Note: Decide if an activation function is needed based on the task.
            # Softmax might be needed for classification tasks, while it's usually not necessary for regression tasks (e.g., streamflow prediction).
            # nn.Softmax(dim=1) # Example: if softmax is needed
        )

    def forward(self, x):
        """
        Model forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, forecast_steps * output_dim).
        """
        # Get the batch size
        batch_size = x.size(0)
        
        # Flatten the input tensor from (batch_size, seq_length, input_dim)
        # to (batch_size, seq_length * input_dim)
        x_flattened = x.view(batch_size, -1)
        
        # Pass through the defined neural network model
        output = self.model(x_flattened)
        
        return output

    @classmethod
    def from_config(cls, config):
        """
        Create a model instance from a configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            SimplePyTorchModel: The model instance.
        """
        params = config.get('params', {})
        input_dim = params.get('input_dim', 10)
        output_dim = params.get('output_dim', 1) # Output dimension for prediction tasks is usually 1
        seq_length = params.get('seq_length', 15) # New parameter
        forecast_steps = params.get('forecast_steps', 1) # New parameter
        
        return cls(
            input_dim=input_dim, 
            output_dim=output_dim,
            seq_length=seq_length,
            forecast_steps=forecast_steps
        )
