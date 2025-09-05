import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..factory.model_factory import ModelFactory

class SimplifiedAttention(nn.Module):
    """
    A simplified attention module to enhance LSTM's feature representation. This implements a simple feature-dimension-based attention to simulate focusing on important features.
    """
    def __init__(self, hidden_dim):
        super(SimplifiedAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # Attention weights vector
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        """
        Args:
            lstm_out (Tensor): LSTM output, shape (batch_size, seq_length, hidden_dim)

        Returns:
            Tensor: Weighted sequence representation, shape (batch_size, hidden_dim)
        """
        # Compute attention scores: (batch_size, seq_length, 1)
        attn_scores = self.attention_weights(lstm_out)
        
        # Apply softmax to get attention weights: (batch_size, seq_length, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum: (batch_size, hidden_dim)
        weighted = torch.sum(attn_weights * lstm_out, dim=1)
        
        return weighted

class TemporalAttention(nn.Module):
    """
    Temporal attention module for the decoder. It computes attention based on the encoder output and a query vector to generate the final representation.
    """
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(TemporalAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attn_dim = attn_dim
        
        # Linear transformations
        self.W_e = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.W_d = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs (Tensor): Encoder output, shape (batch_size, seq_len, encoder_dim)
            decoder_hidden (Tensor): Decoder hidden state, shape (batch_size, decoder_dim)

        Returns:
            Tensor: Attention-weighted context vector, shape (batch_size, encoder_dim)
        """
        # (batch_size, seq_len, attn_dim)
        energy = self.W_e(encoder_outputs) 
        # (batch_size, 1, attn_dim)
        decoder_energy = self.W_d(decoder_hidden).unsqueeze(1)
        # (batch_size, seq_len, attn_dim)
        energy = torch.tanh(energy + decoder_energy)
        # (batch_size, seq_len, 1)
        attention = self.v(energy)
        # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention, dim=1)
        
        # (batch_size, encoder_dim)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context_vector

@ModelFactory.register("eklt_model")
class EKLTModel(nn.Module):
    """
    EKLT water level forecasting model. It combines LSTM, Transformer, and attention mechanisms. Assumes the input data has been pre-processed with EMD (if needed).
    """

    def __init__(self, input_dim, output_dim, seq_length, lstm_hidden_dim, lstm_num_layers,
                 transformer_d_model, transformer_nhead, transformer_num_layers, transformer_dim_feedforward,
                 forecast_steps=1, dropout=0.1):
        """
        Initializes the EKLT model.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Output dimension (typically the number of water level values).
            seq_length (int): Length of the input sequence.
            lstm_hidden_dim (int): LSTM hidden layer dimension.
            lstm_num_layers (int): Number of LSTM layers.
            transformer_d_model (int): The model dimension of the Transformer (must equal lstm_hidden_dim).
            transformer_nhead (int): Number of attention heads in the Transformer.
            transformer_num_layers (int): Number of Transformer encoder layers.
            transformer_dim_feedforward (int): Dimension of the Transformer feedforward network.
            forecast_steps (int): Number of forecast steps.
            dropout (float): Dropout probability.
        """
        super(EKLTModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.forecast_steps = forecast_steps
        self.transformer_d_model = transformer_d_model

        # 1. LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_num_layers, batch_first=True, dropout=dropout if lstm_num_layers > 1 else 0)

        # 2. Attention module after LSTM
        self.lstm_attention = SimplifiedAttention(lstm_hidden_dim)

        # 3. Transformer encoder
        # Ensure dimensions match
        if lstm_hidden_dim != transformer_d_model:
            raise ValueError("LSTM hidden dim must equal Transformer d_model")
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model, 
            nhead=transformer_nhead, 
            dim_feedforward=transformer_dim_feedforward, 
            dropout=dropout,
            batch_first=True # PyTorch >= 1.9
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # 4. Decoder part - temporal attention
        self.decoder_dim = transformer_d_model # Simplified, set to decoder_dim
        self.temporal_attention = TemporalAttention(
            encoder_dim=transformer_d_model, 
            decoder_dim=self.decoder_dim, 
            attn_dim=self.decoder_dim # Simplified, set to decoder_dim
        )
        
        # Decoder query vectors (learnable)
        # Maintain a query vector for each forecast step
        self.decoder_queries = nn.Parameter(torch.randn(forecast_steps, self.decoder_dim))

        # 5. Output layer
        self.output_layer = nn.Linear(transformer_d_model, output_dim)

    def forward(self, x):
        """
        Model forward pass.

        Args:
            x (Tensor): Input sequence, shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: Forecast results, shape (batch_size, forecast_steps * output_dim).
        """
        batch_size = x.size(0)

        # Stage 1 & 2: LSTM + Attention
        lstm_out, _ = self.lstm(x)  # (B, L, H)
        # Apply attention after LSTM to get an aggregated feature vector (B, H)
        # Note: If you want to retain sequence information for the Transformer, you can skip this step or modify it
        # According to the description, LSTM+Attention is for preliminary extraction, then passed to the Transformer
        # So here we keep lstm_out for the Transformer, but attn_lstm_out can also be used as supplementary information
        # For simplicity, we pass lstm_out directly to the Transformer
        # attn_lstm_out = self.lstm_attention(lstm_out) # (B, H) - optional use

        # Stage 3: Transformer Encoder
        # Transformer expects input (B, L, D_model)
        transformer_in = lstm_out 
        transformer_out = self.transformer_encoder(transformer_in)  # (B, L, D_model)

        # Stage 4 & 5: Decoder with Temporal Attention and Output
        outputs = []
        # Use a fixed, learnable hidden state as a representative for the decoder's initial state
        # Simplified here, using the global average pooling of transformer_out as a reference for the decoder's "hidden state"
        decoder_hidden_ref = torch.mean(transformer_out, dim=1) # (B, D_model)
        
        for i in range(self.forecast_steps):
            # Get the query vector for step i
            query = self.decoder_queries[i].unsqueeze(0).expand(batch_size, -1)  # (B, D_model)
            
            # Compute temporal attention context vector
            context = self.temporal_attention(transformer_out, query)  # (B, D_model)
            
            # Generate single-step output
            output = self.output_layer(context)  # (B, output_dim)
            outputs.append(output)

        # Concatenate the results of all forecast steps
        final_output = torch.cat(outputs, dim=1)  # (B, forecast_steps * output_dim)
        
        return final_output

    @classmethod
    def from_config(cls, config):
        """
        Creates a model instance from a configuration dictionary.
        """
        params = config.get('params', {})
        
        input_dim = params.get('input_dim', 1)
        output_dim = params.get('output_dim', 1)
        seq_length = params.get('seq_length', 10)
        forecast_steps = params.get('forecast_steps', 1)
        dropout = params.get('dropout', 0.1)

        # LSTM parameters
        lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        lstm_num_layers = params.get('lstm_num_layers', 2)

        # Transformer parameters
        transformer_d_model = params.get('transformer_d_model', 64)
        transformer_nhead = params.get('transformer_nhead', 8)
        transformer_num_layers = params.get('transformer_num_layers', 2)
        transformer_dim_feedforward = params.get('transformer_dim_feedforward', 128)
        
        # Validate Transformer parameters
        if transformer_d_model % transformer_nhead != 0:
            raise ValueError("transformer_d_model must be divisible by transformer_nhead")

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_length=seq_length,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            forecast_steps=forecast_steps,
            dropout=dropout
        )
