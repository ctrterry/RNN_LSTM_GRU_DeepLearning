import os
import sys
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import our modules
from local_code.stage_4_code.classification.data_loader import TextDataLoader
from local_code.stage_4_code.classification.method_RNN import TextRNN, TextClassifier
from local_code.stage_4_code.classification.evaluation_accuracy import calculate_metrics
from local_code.stage_4_code.classification.evaluation_plot import plot_training_loss, plot_metrics

def run_experiment(config, train_loader, test_loader):
    """
    Run a single experiment with given configuration.
    
    Args:
        config (dict): Configuration parameters
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        
    Returns:
        dict: Results of the experiment
    """
    # Initialize model with config parameters
    model = TextRNN(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Initialize classifier with config parameters
    classifier = TextClassifier(model)
    
    # Set optimizer parameters
    classifier.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    training_losses, test_losses, epoch_metrics = classifier.train(
        train_loader, 
        test_loader, 
        num_epochs=config['num_epochs']
    )
    
    # Get final metrics
    final_metrics = epoch_metrics[-1]
    
    # Create results dictionary
    results = {
        'config': config,
        'final_metrics': final_metrics,
        'training_losses': training_losses,
        'test_losses': test_losses,
        'epoch_metrics': epoch_metrics
    }
    
    return results

def main():
    # Set paths
    data_dir = "data/stage_4_data/text_classification"
    result_dir = "result/stage_4_result/classification/ablation_studies"
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = TextDataLoader(data_dir)
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()
    
    # Convert to PyTorch tensors
    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.LongTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    
    # Define hyperparameter combinations to test
    configs = [
        # Base configuration. Acc: Accuracy:  0.7279, Precision: 0.7822, Recall:    0.6318, F1 Score:  0.6990
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 100,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # Test different embedding dimensions
        # { # 
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 150, # Trying with the larger embedding. From 100 to 200
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # { # 
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 250, # Trying with the larger embedding. From 100 to 200
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # }
        # { # This result is the best result
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200, # Trying with the larger embedding. From 100 to 200
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # }
        # # Test different hidden dimensions
        # { # This result is reall bad
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 100,
        #     'hidden_dim': 256, # Trying with the larger hidden_dim. 128 to 256
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # }
        # # Test different dropout rates
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 100,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.3,  # Trying with the smalle lr from 0.5 to 0.3.
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # # Test different learning rates
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 100,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.0005,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # # Test with weight decay
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 100,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0.0001,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # }

        # Right now, we found that the embadding_dim = 200 is the best
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 256,  # Increased from 128
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 128,
        #     'num_layers': 3,  # Increased from 2
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.3,  # Lower dropout
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.6,  # Higher dropout
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.0005,  # Lower learning rate
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0.0001,  # Small weight decay
        #     'num_epochs': 10,
        #     'batch_size': 64
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 128,
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0.01,  # Higher weight decay
        #     'num_epochs': 10,
        #     'batch_size': 64
        # }, 
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 256,  # Increased from 128
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 128 # Increasing Batch size by 2 
        # },
        # {
        #     'vocab_size': data_loader.get_vocab_size(),
        #     'embedding_dim': 200,
        #     'hidden_dim': 256,  # Increased from 128
        #     'num_layers': 2,
        #     'dropout': 0.5,
        #     'learning_rate': 0.001,
        #     'weight_decay': 0,
        #     'num_epochs': 10,
        #     'batch_size': 32 # decreasing Batch size / 2 
        # }
        { # After trying all posiver hyperparamter as different values. I was found that this is best. 
            'vocab_size': data_loader.get_vocab_size(),
            'embedding_dim': 200,
            'hidden_dim': 128, 
            'num_layers': 3,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'weight_decay': 0,
            'num_epochs': 20,
            'batch_size': 64
        }

    ]
    
    # Store all results
    all_results = []
    
    # Run experiments
    for i, config in enumerate(configs):
        print(f"\nRunning experiment {i+1}/{len(configs)}")
        print("Configuration:", config)
        
        # Create data loaders with current batch size
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        # Run experiment
        results = run_experiment(config, train_loader, test_loader)
        all_results.append(results)
        
        # Save individual experiment results
        experiment_dir = os.path.join(result_dir, f"experiment_{i+1}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save configuration and metrics
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(results['final_metrics'], f, indent=4)
        
        # Plot training curves
        plot_training_loss(
            results['training_losses'],
            results['test_losses'],
            save_path=os.path.join(experiment_dir, 'training_curves.png')
        )
        
        # Plot final metrics
        plot_metrics(
            results['final_metrics'],
            save_path=os.path.join(experiment_dir, 'final_metrics.png')
        )
    
    # Create summary DataFrame
    summary_data = []
    for i, result in enumerate(all_results):
        summary_data.append({
            'Experiment': i+1,
            'Embedding Dim': result['config']['embedding_dim'],
            'Hidden Dim': result['config']['hidden_dim'],
            'Num Layers': result['config']['num_layers'],
            'Dropout': result['config']['dropout'],
            'Learning Rate': result['config']['learning_rate'],
            'Weight Decay': result['config']['weight_decay'],
            'Accuracy': result['final_metrics']['accuracy'],
            'Precision': result['final_metrics']['precision'],
            'Recall': result['final_metrics']['recall'],
            'F1 Score': result['final_metrics']['f1']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv(os.path.join(result_dir, 'experiment_summary.csv'), index=False)
    print("\nExperiment Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary to JSON
    with open(os.path.join(result_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)

if __name__ == "__main__":
    main() 