import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot Mole-BERT Pretraining Metrics')
    parser.add_argument('--file', type=str, default='pretrain_molebert_history.csv',
                        help='Path to the history CSV file generated during pretraining')
    parser.add_argument('--output_image', type=str, default='pretrain_metrics.png',
                        help='Filename to save the resulting plot')
    args = parser.parse_args()

    try:
        # Load the tracking history
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: The history file '{args.csv_file}' was not found.")
        print("Please ensure your pretraining script has saved progress or provide the correct path via --csv_file.")
        return

    # Create a 1x2 grid of subplots without calling plt.figure() to ensure environment stability
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['loss'], label='Total Loss (Tri + CL + Mask)', color='#1f77b4', linewidth=2)
    plt.title('Pretraining Loss Journey')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['acc'], label='Node Masking Accuracy', color='#2ca02c', linewidth=2)
    plt.title('Token Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Automatically optimize subplot spaces to prevent label overlapping or truncation
    plt.tight_layout()

    # Save the visualization directly to disk
    plt.savefig(args.output_image, dpi=300)
    print(f"Success! Metrics successfully visualized and saved to '{args.output_image}'.")

if __name__ == "__main__":
    main()