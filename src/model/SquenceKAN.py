import torch
import torch.nn as nn
from ..factory.model_factory import ModelFactory
from .KAN import KAN

@ModelFactory.register("KAN")
class SequenceKAN(nn.Module):
    """
    Wrapper for KAN model to support sequence input and prediction output
    Input: [batch_size, seq_length, feature_size]
    Output: [batch_size, forecast_horizon]
    """
    
    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        forecast_horizon: int,
        kan_layers_hidden: list,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1]
    ):
        """
        Initializes the SequenceKAN model
        
        Args:
            seq_length: input sequence length
            feature_size: feature dimension
            forecast_horizon: forecast horizon
            kan_layers_hidden: KAN hidden layer structure
            Other parameters: parameters passed to the KAN model
        """
        super(SequenceKAN, self).__init__()
        
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.forecast_horizon = forecast_horizon
        
        # Calculate the flattened input size
        input_size = seq_length * feature_size
        
        # Ensure the input size of the first KAN layer matches
        if kan_layers_hidden[0] != input_size:
            raise ValueError(f"The input size of the first KAN layer ({kan_layers_hidden[0]}) must equal seq_length*feature_size({input_size})")
        
        # Create the KAN model
        self.kan = KAN(
            layers_hidden=kan_layers_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        
        # If the KAN output dimension does not match the forecast_horizon, add a linear layer to adjust
        # Get the output size of the last KAN layer
        kan_output_size = kan_layers_hidden[-1]
        if kan_output_size != forecast_horizon:
            self.output_projection = nn.Linear(kan_output_size, forecast_horizon)
        else:
            self.output_projection = None

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
            x: input tensor, with shape [batch_size, seq_length, feature_size]
            
        Returns:
            Output tensor, with shape [batch_size, forecast_horizon]
        """
        batch_size = x.size(0)
        
        # Check input dimensions
        if x.dim() != 3 or x.size(1) != self.seq_length or x.size(2) != self.feature_size:
            raise ValueError(
                f"Input tensor shape should be [batch_size, {self.seq_length}, {self.feature_size}], "
                f"but got {list(x.shape)}"
            )
        
        # Flatten the input: [batch_size, seq_length, feature_size] -> [batch_size, seq_length * feature_size]
        x = x.view(batch_size, -1)
        
        # Pass through the KAN
        x = self.kan(x)
        
        # Adjust output dimension
        if self.output_projection is not None:
            x = self.output_projection(x)
            
        return x

    @classmethod
    def from_config(cls, config):
        """
        Create a model instance from a configuration
        
        Args:
            config: configuration dictionary
            
        Returns:
            SequenceKAN instance
        """
        valid_params = {
            'seq_length': config.get('seq_length'),
            'feature_size': config.get('feature_size'),
            'forecast_horizon': config.get('forecast_horizon'),
            'kan_layers_hidden': config.get('kan_layers_hidden'),
            'grid_size': config.get('grid_size', 5),
            'spline_order': config.get('spline_order', 3),
            'scale_noise': config.get('scale_noise', 0.1),
            'scale_base': config.get('scale_base', 1.0),
            'scale_spline': config.get('scale_spline', 1.0),
            'base_activation': config.get('base_activation', torch.nn.SiLU),
            'grid_eps': config.get('grid_eps', 0.02),
            'grid_range': config.get('grid_range', [-1, 1])
        }
        
        print(f"Initializing SequenceKAN with:\n{valid_params}")
        return cls(**valid_params)
