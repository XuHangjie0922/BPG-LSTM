import types
import torch
from torch.nn import functional as F
from model import ASTHGNN
import attention_core as training
def spatial_forward(self, x, centers):
    if centers.ndim == 2:
        centers = centers[None, None].expand(x.shape[0], x.shape[1], -1, -1)
    # Only change versus original: sigmoid rather than LeakyReLU before softmax.
    scores = torch.sigmoid(self.node_attention(x) @ self.edge_attention(centers).transpose(-1, -2))
    log_probability = F.log_softmax(scores, dim=-1)
    h = F.gumbel_softmax(log_probability, tau=self.temperature, hard=False, dim=-1) if self.training else F.softmax(log_probability/self.temperature, dim=-1)
    dv = h.sum(-1).clamp_min(1e-8).rsqrt()
    de = h.sum(-2).clamp_min(1e-8).reciprocal()
    edge_features = h.transpose(-1, -2) @ (x*dv[..., None])
    y = self.transform((h @ (edge_features*de[..., None]))*dv[..., None])
    return y, h.transpose(-1, -2) @ y / x.shape[-2], h

def model_with_sigmoid():
    model = ASTHGNN().cuda()
    for layer in model.spatial:
        layer.forward = types.MethodType(spatial_forward, layer)
    return model
