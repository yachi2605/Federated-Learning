import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_cifar100():
    """Get CIFAR-100 datasets with standard transforms"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset



def create_non_iid_split(dataset, num_clients: int, alpha: float = 0.5):
    """Create non-IID data distribution using Dirichlet distribution for CIFAR-100"""
    num_classes = 100  # CIFAR-100 has 100 classes
    client_data_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    
    # Get indices for each class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Distribute data using Dirichlet distribution
    for class_id in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        class_count = len(class_indices[class_id])
        allocations = (proportions * class_count).astype(int)
        
        # Ensure all samples are allocated
        allocations[-1] = class_count - np.sum(allocations[:-1])
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + allocations[client_id]
            client_data_indices[client_id].extend(
                class_indices[class_id][start_idx:end_idx]
            )
            start_idx = end_idx
    
    return client_data_indices


def plot_data_distribution(client_data_indices, dataset):
    """Data distribution visualization with stacked bar chart"""
    num_clients = len(client_data_indices)
    num_classes = 100
    
    # Calculate distribution matrix
    distribution = np.zeros((num_clients, num_classes))
    client_data_sizes = []
    
    for client_id, indices in client_data_indices.items():
        client_data_sizes.append(len(indices))
        for idx in indices:
            _, label = dataset[idx]
            distribution[client_id][label] += 1
    
    # 1. Stacked Bar Chart - Classes per Client
    plt.figure(figsize=(16, 8))
    
    # Sort clients by total data size for better visualization
    client_order = np.argsort(client_data_sizes)[::-1]
    sorted_distribution = distribution[client_order]
    
    # Create stacked bar chart
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for class_id in range(num_classes):
        class_data = sorted_distribution[:, class_id]
        plt.bar(range(num_clients), class_data, bottom=bottom, 
                color=colors[class_id % len(colors)], alpha=0.8, 
                label=f'Class {class_id}' if class_id < 20 else "")
        bottom += class_data
    
    plt.xlabel('Clients (Sorted by Data Size)')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Across Clients (Stacked Bar Chart)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(0, num_clients, max(1, num_clients//10)))
    
    plt.tight_layout()
    plt.savefig('data_distribution_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()

   
    return distribution