"""
honet/data_factory.py
---------------------
Factory functions for producing DataLoaders for each task in the
HONet demonstration and benchmarking scripts.

Supported tasks
---------------
get_task_data()   – multi-modal demo tasks:
    'IMAGE_MNIST'        : MNIST hand-written digits (image, 1-channel, 28x28)
    'TABULAR_CLUSTERS'   : synthetic 2-D Gaussian mixture (tabular, flat)
    'SEQUENTIAL_WAVES'   : synthetic sinusoidal time-series (sequential)

get_split_cifar10_tasks()  – rigorous benchmark:
    Splits CIFAR-10 into N tasks (2 classes each), returning train & test
    loaders per task together with metadata.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# get_task_data  (multi-modal demo)
# ---------------------------------------------------------------------------

def get_task_data(task_name: str, batch_size: int = 128):
    """
    Return (DataLoader, metadata_dict) for a given named task.

    Metadata keys
    -------------
    name      : human-readable task identifier
    type      : one of 'image', 'tabular', 'sequential'
    channels  : (image only) number of input channels
    size      : (image only) spatial resolution
    input_dim : (tabular / sequential) feature dimension per time step
    seq_len   : (sequential only) number of time steps
    """

    if task_name == 'IMAGE_MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        meta = {
            'name': 'IMAGE_MNIST',
            'type': 'image',
            'channels': 1,
            'size': 28,
        }
        return loader, meta

    elif task_name == 'TABULAR_CLUSTERS':
        np.random.seed(0)
        n_samples = 10_000
        centres = [(-3, -3), (3, -3), (-3, 3), (3, 3)]
        data = []
        for cx, cy in centres:
            pts = np.random.randn(n_samples // len(centres), 2) * 0.5
            pts[:, 0] += cx
            pts[:, 1] += cy
            data.append(pts)
        X = np.vstack(data).astype(np.float32)
        # Normalise to [-1, 1]
        X = (X - X.min()) / (X.max() - X.min()) * 2 - 1
        labels = np.zeros(n_samples, dtype=np.int64)
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(labels)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        meta = {
            'name': 'TABULAR_CLUSTERS',
            'type': 'tabular',
            'input_dim': 2,
        }
        return loader, meta

    elif task_name == 'SEQUENTIAL_WAVES':
        np.random.seed(1)
        n_samples = 5_000
        seq_len = 32
        freqs = np.random.uniform(0.5, 3.0, size=n_samples)
        t = np.linspace(0, 2 * np.pi, seq_len)
        # (n_samples, seq_len, 1)
        X = np.sin(freqs[:, None] * t[None, :]).astype(np.float32)[..., None]
        labels = np.zeros(n_samples, dtype=np.int64)
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(labels)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        meta = {
            'name': 'SEQUENTIAL_WAVES',
            'type': 'sequential',
            'input_dim': 1,
            'seq_len': seq_len,
        }
        return loader, meta

    else:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            "Choose from: 'IMAGE_MNIST', 'TABULAR_CLUSTERS', 'SEQUENTIAL_WAVES'."
        )


# ---------------------------------------------------------------------------
# get_split_cifar10_tasks  (rigorous benchmark)
# ---------------------------------------------------------------------------

def get_split_cifar10_tasks(num_tasks: int = 5, batch_size: int = 128):
    """
    Split CIFAR-10 into `num_tasks` disjoint class-pair tasks.

    Parameters
    ----------
    num_tasks  : number of tasks (must be ≤ 5 for standard CIFAR-10)
    batch_size : mini-batch size

    Returns
    -------
    list of dicts, each with keys:
        'train'  : DataLoader for training split
        'test'   : DataLoader for test split
        'meta'   : metadata dict (name, type, channels, size, classes)
    """
    if num_tasks > 5:
        raise ValueError("CIFAR-10 has 10 classes; maximum 5 two-class tasks.")

    cifar_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_full = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_full = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    tasks = []
    for task_idx in range(num_tasks):
        class_a = task_idx * 2
        class_b = task_idx * 2 + 1
        classes = (class_a, class_b)

        # --- Training split ---
        train_indices = [
            i for i, (_, label) in enumerate(train_full)
            if label in classes
        ]
        train_subset = _subset_to_loader(train_full, train_indices, batch_size, shuffle=True)

        # --- Test split ---
        test_indices = [
            i for i, (_, label) in enumerate(test_full)
            if label in classes
        ]
        test_subset = _subset_to_loader(test_full, test_indices, batch_size, shuffle=False)

        meta = {
            'name': f"CIFAR10_{cifar_classes[class_a]}_{cifar_classes[class_b]}",
            'type': 'image',
            'channels': 3,
            'size': 32,
            'classes': classes,
        }
        tasks.append({'train': train_subset, 'test': test_subset, 'meta': meta})

    return tasks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subset_to_loader(dataset, indices: list, batch_size: int, shuffle: bool) -> DataLoader:
    """Slice a torchvision dataset by index list and wrap in a DataLoader."""
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
