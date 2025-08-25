import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize
from sklearn.datasets import make_blobs
import math
# Add this import at the top of the file


def get_task_data(task_name, batch_size):
    """Factory to produce data loaders and metadata for different tasks."""
    print(f"  Loading data for task: '{task_name}'...")
    if task_name == 'IMAGE_MNIST':
        dataset = MNIST('./data', train=True, download=True, transform=ToTensor())
        meta = {'name': 'MNIST_Image', 'type': 'image', 'channels': 1, 'size': 28}
    elif task_name == 'TABULAR_CLUSTERS':
        X, y = make_blobs(n_samples=10000, centers=4, n_features=10, random_state=42)
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        meta = {'name': 'Tabular_Clusters', 'type': 'tabular', 'input_dim': 10}
    elif task_name == 'SEQUENTIAL_WAVES':
       # honet/data_factory.py, line 21 (and a few lines above for clarity)
        N, L, F = 10000, 30, 1
        t = torch.linspace(0, 4 * math.pi, L).unsqueeze(0) # Shape [1, L]
        
        # Generate N unique frequencies and phases
        frequencies = torch.rand(N, 1) * 3 + 1.0 # Shape [N, 1]
        phases = torch.rand(N, 1) * math.pi     # Shape [N, 1]
        
        # Broadcasting [N, 1] * [1, L] results in [N, L]
        X = torch.sin(frequencies * t + phases) + torch.randn(N, L) * 0.1
        X = X.unsqueeze(-1) # Add feature dimension -> [N, L, 1]
        dataset = TensorDataset(X, torch.zeros(N))
        meta = {'name': 'Sequential_Waves', 'type': 'sequential', 'input_dim': F, 'seq_len': L}
    else:
        raise ValueError(f"Unknown task: {task_name}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, meta
def get_split_cifar10_tasks(num_tasks, batch_size):
    """
    Creates a list of data loaders for the Split CIFAR-10 benchmark.
    """
    print(f"  Loading and splitting CIFAR-10 into {num_tasks} tasks...")
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    tasks = []
    classes_per_task = 10 // num_tasks
    
    for i in range(num_tasks):
        start_class = i * classes_per_task
        end_class = (i + 1) * classes_per_task
        
        # Filter train data
        train_indices = [idx for idx, target in enumerate(train_dataset.targets) if start_class <= target < end_class]
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # Filter test data
        test_indices = [idx for idx, target in enumerate(test_dataset.targets) if start_class <= target < end_class]
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        meta = {'name': f'CIFAR10_T{i+1}_C{start_class}-{end_class-1}', 'type': 'image', 'channels': 3, 'size': 32}
        tasks.append({'train': train_loader, 'test': test_loader, 'meta': meta})
        
    return tasks