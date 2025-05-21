import os
import sys
import torch
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import our modules
from local_code.stage_4_code.text_generation.data_loader import TextDataLoader
from local_code.stage_4_code.text_generation.Method_RNN import Method_RNN
from local_code.stage_4_code.text_generation.model import TextGenerator
from local_code.stage_4_code.text_generation.evaluation_plot import (
    plot_training_progress, plot_metrics, calculate_metrics, plot_perplexity
)

def main():
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data/stage_4_data/text_generation")
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "result/stage_4_result/text_generation/rnn")
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize data loader with 90/10 split
    data_loader = TextDataLoader(data_dir, seq_length=10, batch_size=32, train_ratio=0.80)
    
    # Get vocabulary size
    vocab_size = data_loader.get_vocab_size()
    
    # Initialize model
    model = Method_RNN(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.5
    )
    
    # Initialize trainer
    trainer = TextGenerator(model)
    
    # Training parameters
    num_epochs = 10
    train_losses = []
    test_losses = []
    train_perplexities = []
    test_perplexities = []
    metrics = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(data_loader.train_loader)
        train_losses.append(train_loss)
        
        # Calculate training perplexity
        model.eval()
        with torch.no_grad():
            total_train_loss = 0
            total_train_tokens = 0
            for data, target in data_loader.train_loader:
                data, target = data.to(trainer.device), target.to(trainer.device)
                hidden = model.init_hidden(data.size(1), trainer.device)
                output, _ = model(data, hidden)
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target.view(-1),
                    reduction='sum'
                )
                total_train_loss += loss.item()
                total_train_tokens += target.numel()
            
            avg_train_loss = total_train_loss / total_train_tokens
            train_perplexity = torch.exp(torch.tensor(avg_train_loss))
            train_perplexities.append(train_perplexity.item())
        
        # Evaluate
        test_loss = trainer.evaluate(data_loader.test_loader)
        test_losses.append(test_loss)
        
        # Calculate test metrics and perplexity
        model.eval()
        all_predictions = []
        all_targets = []
        total_test_loss = 0
        total_test_tokens = 0
        
        with torch.no_grad():
            for data, target in data_loader.test_loader:
                data, target = data.to(trainer.device), target.to(trainer.device)
                hidden = model.init_hidden(data.size(1), trainer.device)
                output, _ = model(data, hidden)
                
                # Calculate loss for perplexity
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target.view(-1),
                    reduction='sum'
                )
                total_test_loss += loss.item()
                total_test_tokens += target.numel()
                
                # Get predictions for metrics
                predictions = torch.argmax(output, dim=-1)
                all_predictions.extend(predictions.view(-1).cpu().numpy())
                all_targets.extend(target.view(-1).cpu().numpy())
        
        # Calculate test perplexity
        avg_test_loss = total_test_loss / total_test_tokens
        test_perplexity = torch.exp(torch.tensor(avg_test_loss))
        test_perplexities.append(test_perplexity.item())
        
        # Calculate other metrics
        epoch_metrics = calculate_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        metrics.append(epoch_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity.item():.4f}")
        print(f"Test Perplexity: {test_perplexity.item():.4f}")
        print(f"Accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"Precision: {epoch_metrics['precision']:.4f}")
        print(f"Recall: {epoch_metrics['recall']:.4f}")
        print(f"F1 Score: {epoch_metrics['f1']:.4f}")
        print("-" * 50)
    
    # Save metrics
    timestamp = datetime.now().strftime("%m%d")
    metrics_path = os.path.join(result_dir, f"metrics_{timestamp}.txt")
    with open(metrics_path, 'w') as f:
        f.write("Epoch\tTrain_Perplexity\tTest_Perplexity\tAccuracy\tPrecision\tRecall\tF1\n")
        for epoch, (train_perp, test_perp, metric) in enumerate(zip(train_perplexities, test_perplexities, metrics), 1):
            f.write(f"{epoch}\t{train_perp:.4f}\t{test_perp:.4f}\t{metric['accuracy']:.4f}\t{metric['precision']:.4f}\t{metric['recall']:.4f}\t{metric['f1']:.4f}\n")
    
    # Plot training progress
    plot_training_progress(
        train_losses,
        test_losses,
        os.path.join(result_dir, "training_progress.png")
    )
    
    # Plot metrics
    plot_metrics(
        metrics,
        os.path.join(result_dir, "metrics.png")
    )
    
    # Plot perplexity
    plot_perplexity(
        train_perplexities,
        test_perplexities,
        os.path.join(result_dir, "perplexity.png")
    )
    
    # Generate some text
    start_words = "what did the bartender say"
    generated_text = model.generate(
        start_words,
        data_loader.get_word_to_idx(),
        data_loader.get_idx_to_word()
    )
    
    # Save generated text
    with open(os.path.join(result_dir, "generated_text.txt"), 'w') as f:
        f.write(f"Start words: {start_words}\n")
        f.write(f"Generated text: {generated_text}\n")

if __name__ == "__main__":
    main() 