"""Small residual policy/value network, initialized locally from scratch."""
import numpy as np
import torch
from torch import nn
from .game import SIZE


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class Network(nn.Module):
    def __init__(self, channels=64, blocks=4, size=SIZE):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks, "size": size}
        self.trunk = nn.Sequential(nn.Conv2d(8, channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU(),
                                   *(Residual(channels) for _ in range(blocks)))
        self.policy = nn.Conv2d(channels, 1, 1)
        self.value = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(2 * size * size, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x).flatten(1), self.value(x).squeeze(1)

    @torch.inference_mode()
    def evaluate(self, games):
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(np.stack([g.features() for g in games])).to(device)
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
