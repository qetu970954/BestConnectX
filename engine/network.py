"""Residual policy/value models with optional pooling and board attention."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import ARCHITECTURES
from .game import batch_features
from .runtime import DEFAULT_RULES


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class BoardAttention(nn.Module):
    """One pre-norm Transformer block with four heads and learned 2D relative bias."""
    def __init__(self, channels, size):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(),
                                         nn.Linear(2 * channels, channels))
        self.relative_bias = nn.Parameter(torch.zeros(4, (2 * size - 1) ** 2))
        rows, columns = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        coords = torch.stack((rows.flatten(), columns.flatten()), dim=1)
        offsets = coords[:, None] - coords[None, :] + size - 1
        self.register_buffer('relative_index', offsets[..., 0] * (2 * size - 1) + offsets[..., 1],
                             persistent=False)

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.norm1(tokens)).reshape(batch, height * width, 3, 4, channels // 4) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_bias[:, self.relative_index]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        tokens = tokens + self.projection(attended.transpose(1, 2).reshape(batch, height * width, channels))
        tokens = tokens + self.feedforward(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class Network(nn.Module):
    def __init__(self, channels=64, blocks=6, size=DEFAULT_RULES.height, architecture='residual'):
        super().__init__()
        if any(type(value) is not int for value in (channels, blocks, size)) or not (4 <= channels <= 256 and 0 <= blocks <= 32 and 2 <= size <= 25):
            raise ValueError('Model needs 4..256 channels, 0..32 blocks, and board size 2..25.')
        if architecture not in ARCHITECTURES:
            raise ValueError('Architecture must be residual, pooled, or attention.')
        if architecture == 'attention' and channels % 4:
            raise ValueError('Attention needs a channel count divisible by four.')
        self.config = {'channels': channels, 'blocks': blocks, 'size': size}
        if architecture != 'residual':
            self.config['architecture'] = architecture
        self._native_dirty = True
        self.trunk = nn.Sequential(nn.Conv2d(8, channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU(),
                                   *(Residual(channels) for _ in range(blocks)))
        self.attention = BoardAttention(channels, size) if architecture == 'attention' else None
        self.policy = nn.Conv2d(channels, 1, 1)
        pooled = architecture != 'residual'
        self.value = nn.Sequential(nn.Conv2d(channels, 16 if pooled else 2, 1), nn.ReLU(),
                                   nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()) if pooled else nn.Flatten(),
                                   nn.Linear(16 if pooled else 2 * size * size, 64), nn.ReLU(),
                                   nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        x = self.trunk(x)
        if self.attention is not None:
            x = self.attention(x)
        return self.policy(x).flatten(1), self.value(x).squeeze(1)

    @torch.inference_mode()
    def evaluate(self, games):
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(batch_features(games)).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            policy, value = self(x)
        return policy.float().cpu().numpy(), value.float().cpu().numpy()


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a Blackwell-compatible PyTorch build or select CPU.")
    torch.set_num_threads(4)
    return torch.device(name)


def augment(features, policy, rng):
    """Uniformly sample the eight square-board rotations/reflections, with matching policy targets."""
    size = features.shape[-1]
    if features.shape[-2] != size:
        raise ValueError("Eight-way augmentation requires a square board.")
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    features = np.rot90(features, k, axes=(-2, -1))
    policy = np.rot90(policy.reshape(-1, size, size), k, axes=(-2, -1))
    if flip:
        features, policy = features[..., ::-1], policy[..., ::-1]
    return features.copy(), policy.reshape(-1, size * size).copy()
