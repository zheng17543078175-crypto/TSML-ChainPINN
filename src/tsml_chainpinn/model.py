"""Equations (11)-(14). Pointwise attention preserves coordinate derivatives.

Each point is one token, as in the archived implementation. Consequently the
softmax has length one: this is feature exchange, not spatial attention.
The paper does not specify a multi-point tokenization protocol.
"""
import math
import torch
from torch import nn


class FourierEmbedding(nn.Module):
    def __init__(self, bands=64):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.linspace(0, 8, bands))
        self.out_dim = 4 * 2 * bands

    def forward(self, x):
        phase = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        return torch.cat((phase.cos().flatten(1), phase.sin().flatten(1)), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width, beta=2.0):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.Softplus(beta=beta)

    def forward(self, x):
        # Eq. (12): activation outside the skip addition as well as inside.
        return self.act(x + self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, query, context):
        q, kv = query.unsqueeze(1), context.unsqueeze(1)
        # Explicit weight path supports higher-order AD (no fused SDPA kernel).
        update, _ = self.mha(q, kv, kv, need_weights=True)
        return self.norm(q + update).squeeze(1)


class Branch(nn.Module):
    def __init__(self, in_dim, width, depth, heads, out_dim, beta):
        super().__init__()
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, beta) for _ in range(depth)])
        self.self_attention = Attention(width, heads)
        self.output = nn.Linear(width, out_dim)

    def features(self, encoded):
        h = self.blocks(self.input(encoded))
        return self.self_attention(h, h)


class ChainPINN(nn.Module):
    def __init__(self, width=512, depth=10, bands=64, heads=8, softplus_beta=2.0):
        super().__init__()
        self.embedding = FourierEmbedding(bands)
        self.flow = Branch(self.embedding.out_dim, width, depth, heads, 4, softplus_beta)
        self.phase = Branch(self.embedding.out_dim, width, depth, heads, 1, softplus_beta)
        self.phase_to_flow = nn.Linear(width, width)
        self.flow_to_phase = nn.Linear(width, width)
        self.flow_cross = Attention(width, heads)
        self.phase_cross = Attention(width, heads)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)

    def set_stage(self, stage, warmup=False):
        for name, p in self.named_parameters():
            if stage == 1:
                p.requires_grad_(name.startswith("flow."))
            elif stage == 2 or warmup:
                p.requires_grad_(name.startswith("phase."))
            else:
                p.requires_grad_(True)

    def forward(self, x):
        e = self.embedding(x)
        f, a = self.flow.features(e), self.phase.features(e)
        # Both exchanges use the pre-coupling features, not a sequential update.
        fc = self.flow_cross(f, self.phase_to_flow(a))
        ac = self.phase_cross(a, self.flow_to_phase(f))
        logits = self.phase.output(ac)
        return {"flow": self.flow.output(fc),  # [p,u,v,w], normalized
                "alpha_logits": logits, "alpha": logits.sigmoid()}
