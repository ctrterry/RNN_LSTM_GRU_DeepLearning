import matplotlib.pyplot as plt
import os

def plot_training_loss(train_losses, test_losses, save_path=None):
    """
    Plot the training and test loss curves.
    
    Args:
        train_losses (list): List of training losses for each epoch
        test_losses (list): List of test losses for each epoch
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(test_losses, 'r-', label='Test Loss')
    plt.title('Training and Test Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def plot_metrics(metrics, save_path=None):
    """
    Plot the evaluation metrics as a bar chart.
    
    Args:
        metrics (dict): Dictionary containing the metrics
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract metrics and their values
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Create bar chart
    plt.bar(metric_names, metric_values)
    plt.title('Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Metrics are between 0 and 1
    
    # Add value labels on top of bars
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close() 