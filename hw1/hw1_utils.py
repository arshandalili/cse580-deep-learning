"""Helpers for HW1. You do not need to modify this file."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cifar10(n_train, n_test, root="data", seed=0):
    """Random subsets of CIFAR-10 as normalized float tensors (N,3,32,32) and int64 labels."""
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    g = torch.Generator().manual_seed(seed)
    out = []
    for train, n in [(True, n_train), (False, n_test)]:
        ds = datasets.CIFAR10(root, train=train, download=True)
        X = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255)
        y = torch.tensor(ds.targets)
        idx = torch.randperm(len(y), generator=g)[:n]
        out += [((X[idx] - mean) / std).contiguous(), y[idx].contiguous()]
    return out


def random_crop_flip(img, pad=4):
    """Standard CIFAR augmentation applied to one (3,32,32) image."""
    img = F.pad(img, (pad, pad, pad, pad))
    i, j = torch.randint(0, 2 * pad + 1, (2,)).tolist()
    img = img[:, i:i + 32, j:j + 32]
    return img.flip(-1) if torch.rand(()) < 0.5 else img


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj))


def load_json(path):
    return json.loads(Path(path).read_text())


def show_images(X, title="", n=8):
    """Undo normalization and display the first n images of a batch."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    fig, axes = plt.subplots(1, n, figsize=(1.3 * n, 1.6))
    for k, ax in enumerate(axes):
        ax.imshow((X[k].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.axis("off")
    fig.suptitle(title, y=1.05)
    plt.show()


def plot_curves(histories, key, ylabel=None, ax=None, logy=False):
    """histories: {label: history dict}. Plots history[key] against epoch."""
    ax = ax or plt.gca()
    for label, h in histories.items():
        ax.plot(h[key], label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel or key)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def print_table(headers, rows):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
