import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import our modules
from local_code.stage_4_code.classification.data_loader import TextDataLoader
from local_code.stage_4_code.classification.method_RNN import TextRNN, TextClassifier
from local_code.stage_4_code.classification.evaluation_accuracy import calculate_metrics, print_metrics, save_metrics
from local_code.stage_4_code.classification.evaluation_plot import plot_training_loss, plot_metrics

def main():
    # Set paths
    data_dir = "data/stage_4_data/text_classification"
    result_dir = "result/stage_4_result/classification"
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
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    vocab_size = data_loader.get_vocab_size()
    model = TextRNN(vocab_size)
    classifier = TextClassifier(model)
    
    # Train model
    num_epochs = 20
    training_losses, test_losses, epoch_metrics = classifier.train(train_loader, test_loader, num_epochs)
    
    # Plot training and test losses
    plot_training_loss(
        training_losses,
        test_losses,
        save_path=os.path.join(result_dir, "training_test_loss.png")
    )
    
    # Save final metrics
    final_metrics = epoch_metrics[-1]  # Get metrics from last epoch
    save_metrics(final_metrics, os.path.join(result_dir, "metrics.txt"))
    
    # Plot final metrics
    plot_metrics(final_metrics, save_path=os.path.join(result_dir, "metrics_plot.png"))
    
    # Save model
    classifier.save_model(os.path.join(result_dir, "model.pth"))

if __name__ == "__main__":
    main() 