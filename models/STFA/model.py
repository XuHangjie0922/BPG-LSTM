"""Equation-driven local STFA reconstruction, not verified author code."""
import torch
from torch import nn

EDGES = [(1,2),(1,12),(1,4),(1,9),(1,5),(1,3),(2,4),(2,7),(2,8),(2,13),
         (3,14),(3,13),(3,10),(3,6),(4,7),(4,8),(5,9),(5,11),(6,10),(7,8),(8,13),(9,11)]


class STFA(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.kernel = nn.Parameter(torch.empty(10, 1, 5))
        self.attention = nn.Parameter(torch.empty(10, 10))
        self.gat_bias = nn.Parameter(torch.zeros(5))
        self.v = nn.Linear(70, 1, bias=False)
        self.lstm = nn.LSTM(70 + window, 64, batch_first=True)
        # A single effective LSTM bias matches the published parameter count.
        self.lstm.bias_hh_l0.requires_grad_(False)
        self.head = nn.Linear(64, 1)
        adjacency = torch.zeros(14, 14, dtype=torch.bool)
        for i, j in EDGES:
            adjacency[i-1, j-1] = adjacency[j-1, i-1] = True
        self.register_buffer('adjacency', adjacency)
        for k in range(10):
            nn.init.xavier_uniform_(self.kernel[k])
            nn.init.xavier_uniform_(self.attention[k:k+1])
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        with torch.no_grad():
            self.lstm.bias_ih_l0[64:128].fill_(1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def spatial(self, x):
        # x: batch,time,14; each sensor is a scalar node at each time step.
        wh = x[:, :, None, :, None] * self.kernel[None, None, :, :, :]
        left = (wh * self.attention[None, None, :, None, :5]).sum(-1)
        right = (wh * self.attention[None, None, :, None, 5:]).sum(-1)
        logits = nn.functional.leaky_relu(left.unsqueeze(-1) + right.unsqueeze(-2), .2)
        weights = logits.masked_fill(~self.adjacency, -torch.inf).softmax(-1)
        weights = nn.functional.dropout(weights, .2, self.training)
        y = (weights @ wh).mean(2) + self.gat_bias
        return torch.relu(y).flatten(2), weights

    def forward(self, x, diagnostic=False):
        p, spatial_weights = self.spatial(x)
        beta = self.v(torch.tanh(p)).squeeze(-1).softmax(dim=1)
        d = torch.cat([p, beta[:, None, :].expand(-1, self.window, -1)], dim=-1)
        h, _ = self.lstm(d)
        prediction = self.head(h[:, -1]).squeeze(-1)
        return (prediction, beta, spatial_weights) if diagnostic else prediction
