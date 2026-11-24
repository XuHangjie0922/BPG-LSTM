"""Paper-era SAGPooling compatibility; final softmax is selected by StudyModel."""
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import subgraph


class PaperEraSAGPooling(nn.Module):
    """PyG2.0.4 SAGPooling score/filter algorithm, using current PyG primitives.

    Source: pyg-team/pytorch_geometric/2.0.4/torch_geometric/nn/pool/sag_pool.py.
    Only ratio=.5, min_score=None, multiplier=1 are needed here.
    GCNConv follows the PI-STHNN Eq27; exact author GNN choice is unconfirmed.
    """
    def __init__(self, channels):
        super().__init__()
        self.gnn = GCNConv(channels, 1)

    def forward(self, x, edges, weights, batch):
        scores = torch.tanh(self.gnn(x, edges).view(-1))
        permutation = topk(scores, .5, batch)
        new_edges, new_weights = subgraph(permutation, edges, weights,
                                          relabel_nodes=True, num_nodes=len(x))
        return (x[permutation] * scores[permutation, None], new_edges,
                new_weights, batch[permutation], permutation, scores[permutation])


class PISTHNN(nn.Module):
    def __init__(self, edge_index, edge_weight, window=30, fusion='softmax_original'):
        super().__init__()
        if fusion not in ['softmax_original', 'sigmoid_he']:
            raise ValueError(fusion)
        self.fusion = fusion
        self.window = window
        self.register_buffer('edge_index', edge_index.long())
        self.register_buffer('edge_weight', edge_weight.float())
        self.lstm = nn.LSTM(14, 256, 3, batch_first=True)
        self.pattern = nn.Conv1d(window-1, 256, 1)
        self.pattern_query = nn.Linear(256, 256, bias=False)
        self.gats = nn.ModuleList([
            GATConv(window, 128, heads=5, concat=False),
            GATConv(128, 256, heads=3, concat=False),
            GATConv(256, 128, heads=1, concat=False)])
        self.pools = nn.ModuleList([PaperEraSAGPooling(c) for c in [128, 256, 128]])
        self.fusion_query = nn.Parameter(torch.empty(1024))
        if fusion == 'sigmoid_he':
            nn.init.kaiming_uniform_(self.fusion_query.view(1, -1), nonlinearity='relu')
        else:
            nn.init.uniform_(self.fusion_query, -.05, .05)
        self.head = nn.Linear(1024, 1)

    def forward(self, x, return_details=False):
        b, w, n = x.shape
        if (w, n) != (self.window, 14):
            raise ValueError(x.shape)
        hidden, _ = self.lstm(x)
        h = hidden[:, -1]
        patterns = self.pattern(hidden[:, :-1]).transpose(1, 2)
        alpha = torch.sigmoid((patterns*self.pattern_query(h)[:, None]).sum(-1))
        v = (patterns*alpha[..., None]).sum(1)
        nodes = x.transpose(1, 2).reshape(b*n, w)
        batch = torch.arange(b, device=x.device).repeat_interleave(n)
        offsets = (torch.arange(b, device=x.device)*n)[None, :, None]
        edges = (self.edge_index[:, None, :]+offsets).reshape(2, -1)
        weights = self.edge_weight.repeat(b)
        representations = [v, h]
        for gat, pool in zip(self.gats, self.pools):
            nodes = torch.relu(gat(nodes, edges))
            nodes, edges, weights, batch, _, _ = pool(nodes, edges, weights, batch)
            representations.append(torch.sigmoid(global_mean_pool(nodes, batch, size=b)))
        features = torch.cat(representations, -1)
        logits = features*self.fusion_query
        # Final StudyModel selects softmax per Eq30. The constructor retains
        # sigmoid_he only to reproduce the adopted Kaiming query initialization.
        attention = torch.sigmoid(logits) if self.fusion == 'sigmoid_he' else torch.softmax(logits, -1)
        prediction = self.head(features*attention).squeeze(-1)
        if return_details:
            return prediction, dict(attention=attention, features=features)
        return prediction
