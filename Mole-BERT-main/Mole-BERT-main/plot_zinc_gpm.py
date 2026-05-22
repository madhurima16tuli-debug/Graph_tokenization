import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_results(csv_path):
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found. Ensure your training script finished at least 10 epochs.")
        return
    
    df = pd.read_csv(csv_path)
    
    # 2. Initialize the Figure
    # We use two subplots: one for Total Loss and one for component breakdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- Subplot 1: Total Convergence ---
    ax1.plot(df['epoch'], df['total_loss'], color='black', linewidth=2, label='Total Loss')
    ax1.set_ylabel('Loss Value')
    ax1.set_title(f'Training Convergence: {os.path.basename(csv_path)}')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Subplot 2: Chemical Component Breakdown ---
    # This helps a PhD student see if the model is failing on Chemistry or Discretization
    if 'vq_loss' in df.columns:
        ax2.plot(df['epoch'], df['vq_loss'], label='VQ Commitment (Discretization)', color='#1f77b4', alpha=0.8)
    
    if 'atom_loss' in df.columns:
        ax2.plot(df['epoch'], df['atom_loss'], label='Atom Reconstruction (Identity)', color='#d62728', alpha=0.8)
    
    # If using the GIN file which had 'chiral_reconstruction_loss'
    if 'chiral_reconstruction_loss' in df.columns:
        ax2.plot(df['epoch'], df['chiral_reconstruction_loss'], label='Chiral Recon', color='#2ca02c', alpha=0.8)

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Sub-Loss Value')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    
    # 3. Save the Plot
    output_png = csv_path.replace('.csv', '.png')
    plt.savefig(output_png, dpi=300)
    print(f"Successfully generated plot: {output_png}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HVQVAE Loss History")
    parser.add_argument("--file", type=str, required=True, help="/mnt/hdd/gtoken/data1/mole_bert/Mole-BERT-main/Mole-BERT-main/zinc_gpm")
    args = parser.parse_args()
    
    plot_training_results(args.file)