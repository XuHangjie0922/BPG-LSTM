from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch_geometric.nn import GATConv


class STHNN(nn.Module):
    """Time-preserving reconstruction; unresolved paper tensor axes are disclosed."""
    def __init__(self, adjacency, window):
        super().__init__()
        edges = torch.as_tensor(adjacency).nonzero().T.contiguous()
        self.register_buffer('edges', edges.long())
        self.window = window
        self.gat1 = GATConv(1, 64, heads=3, concat=False, dropout=.2)
        self.gat2 = GATConv(64, 64, heads=3, concat=False, dropout=.2)
        self.raw_projection = nn.Linear(14, 64)
        self.projection = nn.Linear(64, 16)
        layer = nn.TransformerEncoderLayer(16, 4, dim_feedforward=2048,
                                           dropout=.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2,
                                                 enable_nested_tensor=False)
        self.head = nn.Linear(16, 1)
        # Positive bias prevents an entirely inactive output ReLU at initialization.
        nn.init.constant_(self.head.bias, 1.)
        position = torch.arange(window).float()[:, None]
        rate = torch.exp(torch.arange(0, 16, 2).float() * (-math.log(10000.) / 16))
        pe = torch.zeros(window, 16)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * rate), torch.cos(position * rate)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        batch, steps, nodes = x.shape
        if steps != self.window or nodes != 14:
            raise ValueError('Expected [batch, window,14]')
        graphs = batch * steps
        offset = torch.arange(graphs, device=x.device) * nodes
        edges = (self.edges[:, None, :] + offset[None, :, None]).reshape(2, -1)
        h = x.reshape(-1, 1)
        h1 = torch.relu(self.gat1(h, edges))
        h2 = torch.relu(self.gat2(h1, edges))
        r1 = torch.relu(h1.reshape(batch, steps, nodes, 64).mean(2))
        r2 = torch.relu(h2.reshape(batch, steps, nodes, 64).mean(2))
        spatial = self.raw_projection(x) + r1 + r2
        tokens = self.projection(spatial) + self.positional_encoding[None]
        # No token or attention edge connects different batch members/engines.
        encoded = self.transformer(tokens)
        return torch.relu(self.head(encoded[:, -1])).squeeze(-1)
