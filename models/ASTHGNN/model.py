from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
"""Paper-based ASTHGNN reconstruction, not an author implementation.

Axes are B,T,N,C. Hypergraph learning is independent at each time step;
temporal convolutions run along T, never across the sensor-node dimension.
Unspecified choices are locked in single_seed_protocol.json.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Hypergraph(nn.Module):
    def __init__(self, channels, temperature):
        super().__init__()
        self.node_attention = nn.Linear(channels, channels, bias=False)
        self.edge_attention = nn.Linear(channels, channels, bias=False)
        self.transform = nn.Linear(channels, channels, bias=False)
        self.temperature = temperature

    def forward(self, x, centers):
        if centers.ndim == 2:
            centers = centers[None, None].expand(x.shape[0], x.shape[1], -1, -1)
        scores = F.leaky_relu(self.node_attention(x) @
                              self.edge_attention(centers).transpose(-1, -2), .2)
        log_probability = F.log_softmax(scores, dim=-1)
        h = F.gumbel_softmax(log_probability, tau=self.temperature, hard=False, dim=-1) \
            if self.training else F.softmax(log_probability / self.temperature, dim=-1)
        dv = h.sum(-1).clamp_min(1e-8).rsqrt()
        de = h.sum(-2).clamp_min(1e-8).reciprocal()
        # Equation (19), factored through hyperedges; no NxN dense Laplacian.
        edge_features = h.transpose(-1, -2) @ (x * dv[..., None])
        y = self.transform((h @ (edge_features * de[..., None])) * dv[..., None])
        # Equation (20) uses 1 / |V|, not hyperedge-degree normalization.
        new_centers = h.transpose(-1, -2) @ y / x.shape[-2]
        return y, new_centers, h


class GatedTemporal(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.filter = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation))

    def forward(self, x):
        z = F.pad(x.permute(0, 3, 2, 1), (self.dilation, 0))
        return (torch.tanh(self.filter(z)) * torch.sigmoid(self.gate(z))).permute(0, 3, 2, 1)


class ASTHGNN(nn.Module):
    def __init__(self, channels=32, hyperedges=3, blocks=3, temporal_per_block=2, temperature=1.):
        super().__init__()
        self.embedding = nn.Linear(1, channels)
        self.centers = nn.Parameter(torch.empty(hyperedges, channels))
        nn.init.xavier_uniform_(self.centers)
        self.spatial = nn.ModuleList([Hypergraph(channels, temperature) for _ in range(blocks)])
        self.temporal = nn.ModuleList([
            nn.Sequential(*[GatedTemporal(channels, 2**(b*temporal_per_block+i))
                            for i in range(temporal_per_block)]) for b in range(blocks)])
        self.channel_projection = nn.Conv1d(channels, 1, 1)
        self.regression = nn.Linear(15, 1)
        self.receptive_field = 2**(blocks*temporal_per_block)

    def forward(self, x, audit=False):
        assert x.ndim == 3 and x.shape[-1] == 15
        z = self.embedding(x[..., None])
        skip, centers, assignments = z, self.centers, []
        for spatial, temporal in zip(self.spatial, self.temporal):
            spatial_z, centers, h = spatial(z, centers)
            z = z + temporal(spatial_z)
            skip = skip + z
            assignments.append(h)
        projected = self.channel_projection(skip[:, -1].transpose(1, 2)).squeeze(1)
        prediction = torch.sigmoid(self.regression(projected)).squeeze(-1) * 125.
        return (prediction, assignments) if audit else prediction
