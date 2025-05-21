import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import re
import os

class TextDataset(Dataset):
    def __init__(self, data_path, seq_length=50):
        self.data_path = data_path
        self.seq_length = seq_length
        self.word_to_idx, self.idx_to_word = self._build_vocabulary()
        self.data = self._load_data()
        
    def _build_vocabulary(self):
        # Read the CSV file
        df = pd.read_csv(self.data_path)
        
        # Combine all jokes into one text
        text = ' '.join(df['Joke'].astype(str))
        
        # Tokenize
        words = text.lower().split()
        
        # Build vocabulary
        word_counts = Counter(words)
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, count in word_counts.items() if count > 1]
        
        # Create mappings
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        return word_to_idx, idx_to_word
    
    def _load_data(self):
        df = pd.read_csv(self.data_path)
        sequences = []
        
        for joke in df['Joke']:
            # Tokenize and convert to indices
            words = joke.lower().split()
            indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            
            # Add start and end tokens
            indices = [self.word_to_idx['<START>']] + indices + [self.word_to_idx['<END>']]
            
            # Create sequences
            for i in range(0, len(indices) - self.seq_length):
                sequences.append(indices[i:i + self.seq_length + 1])
        
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # For batch_first=True, we need to ensure the sequence is properly shaped
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        return x, y

class TextDataLoader:
    def __init__(self, data_dir, seq_length=50, batch_size=64, train_ratio=0.8):
        self.data_path = os.path.join(data_dir, 'data.csv')
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        # Create dataset
        self.dataset = TextDataset(self.data_path, seq_length)
        
        # Split into train and test
        train_size = int(train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def get_vocab_size(self):
        return len(self.dataset.word_to_idx)
    
    def get_word_to_idx(self):
        return self.dataset.word_to_idx
    
    def get_idx_to_word(self):
        return self.dataset.idx_to_word 