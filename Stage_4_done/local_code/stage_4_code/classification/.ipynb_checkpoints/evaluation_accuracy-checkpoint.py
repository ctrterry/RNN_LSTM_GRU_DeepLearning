import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print the evaluation metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary containing the metrics
    """
    print("\nEvaluation Metrics:")
    print("-" * 20)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("-" * 20)

def save_metrics(metrics, filepath):
    """
    Save the evaluation metrics to a file.
    
    Args:
        metrics (dict): Dictionary containing the metrics
        filepath (str): Path to save the metrics
    """
    with open(filepath, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write("-" * 20 + "\n") 