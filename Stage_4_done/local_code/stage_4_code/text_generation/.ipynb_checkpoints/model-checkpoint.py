import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class BaseTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(BaseTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        return embedded, hidden
    
    def init_hidden(self, batch_size, device):
        # To be implemented by child classes
        raise NotImplementedError
    
    def generate(self, start_words, word_to_idx, idx_to_word, max_length=100):
        # To be implemented by child classes
        raise NotImplementedError

class TextGenerator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            hidden = self.model.init_hidden(data.size(1), self.device)
            output, hidden = self.model(data, hidden)
            
            loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                hidden = self.model.init_hidden(data.size(1), self.device)
                output, hidden = self.model(data, hidden)
                loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
                total_loss += loss.item()
        return total_loss / len(test_loader)
    
    def save_metrics(self, metrics, save_path):
        timestamp = datetime.now().strftime("%m%d")
        with open(save_path, 'w') as f:
            f.write("Epoch\tAccuracy\tPrecision\tRecall\tF1\n")
            for epoch, metric in enumerate(metrics, 1):
                f.write(f"{epoch}\t{metric['accuracy']:.4f}\t{metric['precision']:.4f}\t{metric['recall']:.4f}\t{metric['f1']:.4f}\n")
    
    def save_plot(self, train_losses, test_losses, save_path):
        timestamp = datetime.now().strftime("%m%d")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Time')
        plt.legend()
        plt.grid(True)
        
        # Add timestamp to filename
        save_path = save_path.replace('.png', f'_{timestamp}.png')
        plt.savefig(save_path)
        plt.close() 