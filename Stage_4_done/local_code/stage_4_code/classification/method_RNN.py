import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Trying now is overfiting about the dropout = 0.5
# Second trying, setup dropout = 0.9 -> This is really bad performance
# Where is my output dimension?
# can I changed my output dimension to 1 ?

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, num_layers=3, dropout=0.5):
        """
        Initialize the RNN model for text classification.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of hidden state
            num_layers (int): Number of RNN layers
            dropout (float): Dropout rate
        """
        super(TextRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # x: [batch_size, sequence_length]
        embedded = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        
        # LSTM output
        lstm_out, _ = self.lstm(embedded)  # [batch_size, sequence_length, hidden_dim*2]
        
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]
        
        # Apply dropout
        dropped = self.dropout(last_hidden)
        
        # Fully connected layer
        out = self.fc(dropped)  # [batch_size, 1]
        
        # Sigmoid activation
        out = self.sigmoid(out)
        
        return out

class TextClassifier:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the text classifier.
        
        Args:
            model (TextRNN): The RNN model
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.BCELoss()
        
        # Initialize optimizer with initial learning rate
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by half
            patience=2,  # Wait 2 epochs before reducing
            min_lr=1e-6,  # Minimum learning rate
            verbose=False  # Don't print scheduler messages
        )
        
        self.training_losses = []
        self.test_losses = []
        self.epoch_metrics = []
        self.learning_rates = []  # Track learning rates
        self.best_test_loss = float('inf')
        self.patience = 3  # Number of epochs to wait for improvement
        self.patience_counter = 0
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            
        Returns:
            dict: Dictionary containing metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    
    def calculate_epoch_metrics(self, y_true, y_pred):
        """
        Calculate epoch-wise metrics.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            
        Returns:
            dict: Dictionary containing epoch-wise metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y.unsqueeze(1))
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate_epoch(self, test_loader):
        """
        Evaluate the model and return metrics.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            tuple: (test_loss, metrics)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))
                total_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        return total_loss / len(test_loader), metrics
    
    def train(self, train_loader, test_loader, num_epochs=10):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            num_epochs (int): Number of epochs to train
            
        Returns:
            tuple: (training_losses, test_losses, epoch_metrics)
        """
        self.training_losses = []
        self.test_losses = []
        self.epoch_metrics = []
        self.learning_rates = []
        self.best_test_loss = float('inf')
        self.patience_counter = 0

        # self.scheduler = OneCycleLR(
        # self.optimizer,
        #     max_lr=3e-3,                   # peak LR to try
        #     steps_per_epoch=len(train_loader),
        #     epochs=num_epochs,
        #     pct_start=0.3,                 # 30% of cycle to warm up
        #     div_factor=10,                 # start LR = max_lr/div_factor
        #     final_div_factor=1e4,          # end LR = max_lr/final_div_factor
        #     anneal_strategy='cos'          # cosine decay schedule
        # )
        
        # First, train the model and show progress
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_losses.append(train_loss)
            
            # Evaluate
            test_loss, metrics = self.evaluate_epoch(test_loader)
            self.test_losses.append(test_loss)
            self.epoch_metrics.append(metrics)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Print training progress
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}')
            
            # Early stopping check
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.patience_counter = 0
                # Save best model
                self.save_model('best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # After training is complete, print the metrics summary
        print("\n=== Text Classification Epoch-wise Metrics Summary ===")
        print("Epoch  Accuracy  Precision  Recall    F1-score")
        print("-" * 50)
        
        for epoch, metrics in enumerate(self.epoch_metrics):
            print(f"{epoch+1:2d}    {metrics['accuracy']:.4f}    {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}")
            
        return self.training_losses, self.test_losses, self.epoch_metrics
    
    def predict(self, test_loader):
        """
        Make predictions on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            tuple: (predictions, true_labels)
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds = (outputs > 0.5).float()
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch_y.numpy())
                
        return np.array(predictions), np.array(true_labels)
    
    def save_model(self, path):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(path)) 