import os
import re
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextDataLoader:
    def __init__(self, data_dir, max_vocab_size=10000, max_sequence_length=200):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the data directory
            max_vocab_size (int): Maximum size of vocabulary
            max_sequence_length (int): Maximum length of sequences
        """
        self.data_dir = data_dir
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def clean_text(self, text):
        """
        Clean the text by removing special characters, converting to lowercase,
        and removing stop words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of cleaned words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        
        return words
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from the texts.
        
        Args:
            texts (list): List of text samples
            
        Returns:
            dict: Word to index mapping
        """
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self.clean_text(text)
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)  # -2 for <PAD> and <UNK>
        
        # Create word to index mapping
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
        return self.word_to_idx
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of indices.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Sequence of word indices
        """
        words = self.clean_text(text)
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate sequence
        if len(sequence) < self.max_sequence_length:
            sequence.extend([0] * (self.max_sequence_length - len(sequence)))
        else:
            sequence = sequence[:self.max_sequence_length]
            
        return sequence
    
    def load_data(self):
        """
        Load and preprocess the data.
        
        Returns:
            tuple: (X_train, y_train), (X_test, y_test)
        """
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        # Load training data
        for label in ['pos', 'neg']:
            label_dir = os.path.join(self.data_dir, 'train', label)
            for filename in os.listdir(label_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        X_train.append(text)
                        y_train.append(1 if label == 'pos' else 0)
        
        # Load test data
        for label in ['pos', 'neg']:
            label_dir = os.path.join(self.data_dir, 'test', label)
            for filename in os.listdir(label_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        X_test.append(text)
                        y_test.append(1 if label == 'pos' else 0)
        
        # Build vocabulary from training data
        self.build_vocabulary(X_train)
        
        # Convert texts to sequences
        X_train = np.array([self.text_to_sequence(text) for text in X_train])
        X_test = np.array([self.text_to_sequence(text) for text in X_test])
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        return (X_train, y_train), (X_test, y_test)
    
    def get_vocab_size(self):
        """Get the size of the vocabulary."""
        return len(self.word_to_idx) 