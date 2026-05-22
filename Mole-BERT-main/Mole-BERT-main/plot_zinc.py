import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_history(csv_file):
    # Load the data
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
    
    df = pd.read_csv(csv_file)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Total Loss
    ax1.plot(df['epoch'], df['total_loss'], color='black', linewidth=2, label='Total Loss')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('HVQVAE Training Convergence')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Decomposition of Losses
    ax2.plot(df['epoch'], df['vq_loss'], label='VQ Commitment Loss', color='blue')
    ax2.plot(df['epoch'], df['atom_reconstruction_loss'], label='Atom Recon Loss', color='red')
    ax2.plot(df['epoch'], df['chiral_reconstruction_loss'], label='Chiral Recon Loss', color='green')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sub-Loss Value')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
    # Save the plot
    plot_name = csv_file.replace('.csv', '.png')
    plt.savefig(plot_name)
    print(f"Plot saved as {plot_name}")
    plt.show()

if __name__ == "__main__":
    # Change this to match your actual history filename
    history_file = '/mnt/hdd/gtoken/data1/mole_bert/Mole-BERT-main/Mole-BERT-main/zinc_gin_500'
    plot_history(history_file)