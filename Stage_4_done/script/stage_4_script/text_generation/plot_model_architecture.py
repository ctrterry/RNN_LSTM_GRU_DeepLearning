import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def plot_simple_architecture():
    # Set paths
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "result/stage_4_result/text_generation/model_architecture")
    os.makedirs(result_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot RNN Architecture
    plt.subplot(131)
    plt.title('RNN Architecture', fontsize=12, pad=20)
    
    # Draw RNN structure
    plt.plot([0.2, 0.8], [0.5, 0.5], 'k-', linewidth=2)  # Input to hidden
    plt.plot([0.8, 0.8], [0.5, 0.7], 'k-', linewidth=2)   # Hidden to output
    plt.plot([0.8, 0.2], [0.7, 0.7], 'k-', linewidth=2)   # Output to next input
    
    # Add nodes
    plt.scatter([0.2], [0.5], s=200, c='lightblue', edgecolor='black', label='Input')
    plt.scatter([0.8], [0.5], s=200, c='lightgreen', edgecolor='black', label='Hidden')
    plt.scatter([0.8], [0.7], s=200, c='lightcoral', edgecolor='black', label='Output')
    
    # Add text
    plt.text(0.5, 0.3, 'RNN Cell\n\nEmbedding: 200\nHidden: 256\nLayers: 2', 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    
    # Plot LSTM Architecture
    plt.subplot(132)
    plt.title('LSTM Architecture', fontsize=12, pad=20)
    
    # Draw LSTM structure
    plt.plot([0.2, 0.8], [0.5, 0.5], 'k-', linewidth=2)  # Input to cell
    plt.plot([0.8, 0.8], [0.5, 0.7], 'k-', linewidth=2)   # Cell to output
    plt.plot([0.8, 0.2], [0.7, 0.7], 'k-', linewidth=2)   # Output to next input
    
    # Add nodes
    plt.scatter([0.2], [0.5], s=200, c='lightblue', edgecolor='black', label='Input')
    plt.scatter([0.8], [0.5], s=200, c='lightgreen', edgecolor='black', label='LSTM Cell')
    plt.scatter([0.8], [0.7], s=200, c='lightcoral', edgecolor='black', label='Output')
    
    # Add text
    plt.text(0.5, 0.3, 'LSTM Cell\n\nEmbedding: 200\nHidden: 256\nLayers: 2\n\nForget Gate\nInput Gate\nOutput Gate\nCell State', 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    
    # Plot GRU Architecture
    plt.subplot(133)
    plt.title('GRU Architecture', fontsize=12, pad=20)
    
    # Draw GRU structure
    plt.plot([0.2, 0.8], [0.5, 0.5], 'k-', linewidth=2)  # Input to cell
    plt.plot([0.8, 0.8], [0.5, 0.7], 'k-', linewidth=2)   # Cell to output
    plt.plot([0.8, 0.2], [0.7, 0.7], 'k-', linewidth=2)   # Output to next input
    
    # Add nodes
    plt.scatter([0.2], [0.5], s=200, c='lightblue', edgecolor='black', label='Input')
    plt.scatter([0.8], [0.5], s=200, c='lightgreen', edgecolor='black', label='GRU Cell')
    plt.scatter([0.8], [0.7], s=200, c='lightcoral', edgecolor='black', label='Output')
    
    # Add text
    plt.text(0.5, 0.3, 'GRU Cell\n\nEmbedding: 200\nHidden: 256\nLayers: 2\n\nUpdate Gate\nReset Gate\nHidden State', 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    
    # Add legend
    plt.figlegend(['Input', 'Hidden/Cell', 'Output'], 
                  loc='center', bbox_to_anchor=(0.5, 0.02),
                  ncol=3, fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(os.path.join(result_dir, "simple_model_architectures.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_simple_architecture() 