from .federated_learning import FederatedLearning
from .data_utils import get_cifar100, create_non_iid_split, plot_data_distribution
from .config import Config
import matplotlib.pyplot as plt
import pandas as pd
import json

def run():
    config = Config()
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_dataset, test_dataset = get_cifar100()
    
    # Create non-IID data distribution
    print("Creating non-IID data distribution...")
    client_data_indices = create_non_iid_split(
        train_dataset, config.NUM_CLIENTS, config.NON_IID_ALPHA
    )
    
    # Plot data distribution
    plot_data_distribution(client_data_indices, train_dataset)
    
    # Initialize and run federated learning
    fl = FederatedLearning(config)
    fl.run_federated_learning(train_dataset, test_dataset, client_data_indices)

    # Collect results from the trainer instance
    results = pd.DataFrame(fl.round_metrics)

    # Generate outputs
    generate_outputs(results, client_data_indices, train_dataset, config)

def generate_outputs(results, client_data_indices, train_dataset, config):
    """Generate outputs and visualizations"""
    
    if results is None:
        print("No federated learning results available; skipping output generation.")
        return

    # Convert results to DataFrame
    if isinstance(results, list):
        results_df = pd.DataFrame(results)
    elif isinstance(results, pd.DataFrame):
        results_df = results
    else:
        results_df = pd.DataFrame(results)

    if results_df is None or results_df.empty:
        print("Federated learning produced no metrics to plot or save.")
        return

    # Backward compatibility for legacy column names
    column_aliases = {}
    if 'avg_train_loss' in results_df.columns and 'train_loss' not in results_df.columns:
        column_aliases['avg_train_loss'] = 'train_loss'
    if 'avg_train_acc' in results_df.columns and 'train_acc' not in results_df.columns:
        column_aliases['avg_train_acc'] = 'train_acc'
    if column_aliases:
        results_df = results_df.rename(columns=column_aliases)

    
    # Save data files
    save_data_files(results_df, client_data_indices, train_dataset, config)

def save_data_files(results_df, client_data_indices, train_dataset, config):
    """Save data files for analysis"""
    
    # Save client data statistics
    client_stats = []
    for client_id, indices in client_data_indices.items():
        class_counts = {}
        for idx in indices:
            _, label = train_dataset[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        client_stats.append({
            'client_id': client_id,
            'data_size': len(indices),
            'num_classes': len(class_counts)
        })
    
    pd.DataFrame(client_stats).to_csv('client_data_stats.csv', index=False)
    print("Saved client data statistics to 'client_data_stats.csv'")
    
    # Save training configuration
    config_dict = {key: value for key, value in vars(config).items() if not key.startswith('_')}
    with open('training_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print("Saved training configuration to 'training_config.json'")
    
    # Save final results summary
    if 'test_accuracy' in results_df.columns:
        final_round = results_df.iloc[-1]
        summary = {
            'final_test_accuracy': float(final_round['test_accuracy']),
            'final_test_loss': float(final_round['test_loss']),
            'total_rounds': len(results_df),
            'model': config.MODEL_NAME
        }
        
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("Saved training summary to 'training_summary.json'")
