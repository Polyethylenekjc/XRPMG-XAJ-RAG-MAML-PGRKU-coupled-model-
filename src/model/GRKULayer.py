import torch
import torch.nn as nn
from .KAN import KAN


class GRKUCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, grid_eps=0.02, grid_range=[-1, 1]):
        super(GRKUCell, self).__init__()
        self.hidden_size = hidden_size

        # Input projection layer
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Reset gate using LayerNorm + Tanh
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )

        # Update gate using LayerNorm + Sigmoid
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )

        # Candidate hidden state using multi-layer KAN
        self.candidate_hidden = KAN(
            layers_hidden=[hidden_size + hidden_size, 64, hidden_size],
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

        # Candidate state normalization
        self.candidate_norm = nn.LayerNorm(hidden_size)

        # Non-linear activation function
        self.activation = nn.SiLU()  # or nn.GELU()
        self.residual_proj = nn.Linear(hidden_size + hidden_size, hidden_size) if (hidden_size + hidden_size != hidden_size) else None

    def forward(self, x, h_prev):
        x_proj = self.input_proj(x)
        combined = torch.cat((x_proj, h_prev), dim=1)

        reset = self.reset_gate(combined)
        update = self.update_gate(combined)

        h_candidate = self.candidate_hidden(combined)
        
        # Add residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(combined)
        else:
            residual = combined  # can use combined directly if dimensions are consistent

        h_candidate = self.candidate_norm(h_candidate + residual)
        
        h_candidate = self.candidate_norm(h_candidate)
        h_candidate = self.activation(h_candidate)

        h_t = (1 - update) * h_prev + update * h_candidate
        return h_t


class GRKULayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.1,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1]
    ):
        super().__init__()
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList()

        # Diversified parameter settings (different for each layer)
        grid_sizes = [grid_size + i for i in range(num_layers)]
        scales_spline = [scale_spline * (1 + i * 0.2) for i in range(num_layers)]

        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.gru_cells.append(
                GRKUCell(
                    input_dim, hidden_size,
                    grid_size=grid_sizes[i],
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scales_spline[i],
                    grid_eps=grid_eps,
                    grid_range=grid_range
                )
            )

        self.dropout_layer = nn.Dropout(dropout)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.gru_cells[0].hidden_size).to(x.device) # type: ignore

        for t in range(seq_len):
            h_t_list = []
            x_t = x[:, t]

            for layer_idx, (cell, norm) in enumerate(zip(self.gru_cells, self.norm_layers)):
                if layer_idx == 0:
                    h_new = cell(x_t, h_t)
                else:
                    h_new = cell(h_t_list[-1], h_t)

                # Residual connection + normalization
                h_new = norm(h_new + h_t)
                h_new = self.dropout_layer(h_new)
                h_t = h_new
                h_t_list.append(h_t)

        output = torch.stack(h_t_list, dim=1)  # [batch_size, seq_len, hidden_size]
        return output, h_t
