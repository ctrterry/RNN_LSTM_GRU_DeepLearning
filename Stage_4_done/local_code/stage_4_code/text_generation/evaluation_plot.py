import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch

def plot_training_progress(train_losses, test_losses, save_path):
    """
    Plot training and test losses over epochs.
    
    Args:
        train_losses (list): List of training losses per epoch
        test_losses (list): List of test losses per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    
    plt.title('Training and Test Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%m%d")
    save_path = save_path.replace('.png', f'_{timestamp}.png')
    
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics, save_path):
    """
    Plot accuracy, precision, recall, and F1 score over epochs.
    
    Args:
        metrics (list): List of dictionaries containing metrics per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(metrics) + 1)
    
    # Extract metrics
    accuracy = [m['accuracy'] for m in metrics]
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    
    plt.plot(epochs, accuracy, 'b-', label='Accuracy')
    plt.plot(epochs, precision, 'r-', label='Precision')
    plt.plot(epochs, recall, 'g-', label='Recall')
    plt.plot(epochs, f1, 'y-', label='F1 Score')
    
    plt.title('Model Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%m%d")
    save_path = save_path.replace('.png', f'_{timestamp}.png')
    
    plt.savefig(save_path)
    plt.close()

def calculate_perplexity(model, data_loader, device):
    """
    Calculate perplexity of the model on the given data.
    
    Args:
        model: The language model
        data_loader: DataLoader containing the evaluation data
        device: Device to run the model on
        
    Returns:
        float: Perplexity score
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            hidden = model.init_hidden(data.size(1), device)
            output, _ = model(data, hidden)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)),
                target.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += target.numel()
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()

def calculate_metrics(predictions, targets, model=None, data_loader=None, device=None):
    """
    Calculate accuracy, precision, recall, F1 score, and perplexity.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth targets
        model: The language model (optional)
        data_loader: DataLoader for perplexity calculation (optional)
        device: Device to run the model on (optional)
        
    Returns:
        dict: Dictionary containing metrics
    """
    # Convert to numpy arrays
    preds = predictions.cpu().numpy()
    targs = targets.cpu().numpy()
    
    # Calculate metrics
    accuracy = np.mean(preds == targs)
    
    # Calculate precision, recall, and F1 for each class
    precision = []
    recall = []
    f1 = []
    
    for class_idx in range(preds.max() + 1):
        true_positives = np.sum((preds == class_idx) & (targs == class_idx))
        false_positives = np.sum((preds == class_idx) & (targs != class_idx))
        false_negatives = np.sum((preds != class_idx) & (targs == class_idx))
        
        # Calculate precision
        if true_positives + false_positives > 0:
            p = true_positives / (true_positives + false_positives)
        else:
            p = 0
        precision.append(p)
        
        # Calculate recall
        if true_positives + false_negatives > 0:
            r = true_positives / (true_positives + false_negatives)
        else:
            r = 0
        recall.append(r)
        
        # Calculate F1
        if p + r > 0:
            f = 2 * (p * r) / (p + r)
        else:
            f = 0
        f1.append(f)
    
    # Average metrics across classes
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    metrics = {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }
    
    # Calculate perplexity if model and data_loader are provided
    if model is not None and data_loader is not None and device is not None:
        perplexity = calculate_perplexity(model, data_loader, device)
        metrics['perplexity'] = perplexity
    
    return metrics

def plot_perplexity(train_perplexities, test_perplexities, save_path):
    """
    Plot training and test perplexities over epochs.
    
    Args:
        train_perplexities (list): List of training perplexities per epoch
        test_perplexities (list): List of test perplexities per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_perplexities) + 1)
    
    plt.plot(epochs, train_perplexities, 'b-', label='Training Perplexity')
    plt.plot(epochs, test_perplexities, 'r-', label='Test Perplexity')
    
    plt.title('Training and Test Perplexity Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%m%d")
    save_path = save_path.replace('.png', f'_{timestamp}.png')
    
    plt.savefig(save_path)
    plt.close() 