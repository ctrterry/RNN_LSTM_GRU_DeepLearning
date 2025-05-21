import torch
import torch.nn as nn
from .model import BaseTextGenerator

class Method_RNN(BaseTextGenerator):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(Method_RNN, self).__init__(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        
    def forward(self, x, hidden=None):
        # x shape: (seq_len, batch_size)
        embedded, _ = super().forward(x)
        
        # RNN forward pass
        output, hidden = self.rnn(embedded, hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to vocabulary size
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def generate(self, start_words, word_to_idx, idx_to_word, max_length=100):
        self.eval()
        words = start_words.split()
        hidden = self.init_hidden(1, next(self.parameters()).device)
        
        # Convert start words to indices
        indices = [word_to_idx[word] for word in words]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = torch.tensor([indices[-1]], device=next(self.parameters()).device).unsqueeze(0)
                
                # Forward pass
                output, hidden = self(x, hidden)
                
                # Get probabilities
                probs = torch.softmax(output[-1], dim=-1)
                
                # Sample next word
                next_word_idx = torch.multinomial(probs, 1).item()
                indices.append(next_word_idx)
                
                # Stop if we predict the end token···
                if next_word_idx == word_to_idx['<END>']:
                    break
        
        # Convert indices back to words
        generated_words = [idx_to_word[idx] for idx in indices]
        return ' '.join(generated_words) 