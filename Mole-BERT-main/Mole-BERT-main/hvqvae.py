import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Importing from your project's model and loader files
from model import (
    GPMEncoder,
    VectorQuantizer,
    GNNDecoder,
    NUM_NODE_ATTR,
    NUM_NODE_CHIRAL,
)
from loader import MoleculeDataset

# --- Model Definition ---

class HVQVAE(nn.Module):
    def __init__(self, emb_dim=300, num_tokens=512, **kwargs):
        super().__init__()
        # GPM Encoder uses random walks and Transformer blocks
        self.encoder = GPMEncoder(hidden_dim=emb_dim, **kwargs)
        
        # Chemistry-aware Vector Quantizer (C, N, O neighborhoods)
        self.codebook = VectorQuantizer(emb_dim, num_tokens)
        
        # Decoders for reconstruction self-supervision
        self.atom_decoder = GNNDecoder(emb_dim, NUM_NODE_ATTR)
        self.chiral_decoder = GNNDecoder(emb_dim, NUM_NODE_CHIRAL)

    def forward(self, data):
        """Standard forward pass for training the tokenizer."""
        graph_rep, node_rep = self.encoder(data)
        quantized, vq_loss = self.codebook(data.x, node_rep)
        
        pred_atom = self.atom_decoder(quantized, data.edge_index, data.edge_attr)
        pred_chiral = self.chiral_decoder(quantized, data.edge_index, data.edge_attr)
        
        return quantized, graph_rep, vq_loss, pred_atom, pred_chiral

    @torch.no_grad()
    def get_codebook_indices(self, data):
        _, node_rep = self.encoder(data)
        # Passes the GPM features through the nearest-neighbor codebook logic
        return self.codebook.get_codebook_indices(data.x, node_rep)

# --- Training Logic ---

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def main():
    parser = argparse.ArgumentParser(description="Train HVQVAE with GPM on ZINC Subset")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--batch_size", type=int, default=256, help="Input batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension size")
    parser.add_argument("--gpm_num_patterns", type=int, default=10, help="Number of random walk patterns")
    parser.add_argument("--gpm_walk_length", type=int, default=5, help="Length of each random walk")
    parser.add_argument("--output_model_file", type=str, required=True, help="Path to save weights (.pth)")
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent", help="Dataset name")
    #parser.add_argument("--gpm_num_transformer_layers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model
    model = HVQVAE(
        emb_dim=args.emb_dim,
        num_patterns=args.gpm_num_patterns,
        walk_length=args.gpm_walk_length,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 2. Load Dataset & Implement 10% Subset Logic
    print(f"Loading dataset: {args.dataset}...")
    dataset = MoleculeDataset(f"./dataset/{args.dataset}", dataset=args.dataset)
    
    num_molecules = len(dataset)
    indices = list(range(num_molecules))
    random.seed(42)
    random.shuffle(indices)
    
    # Selecting the 10% slice for faster PhD iteration
    subset_size = int(num_molecules * 0.1)
    subset_id = indices[:subset_size]
    dataset_subset = torch.utils.data.Subset(dataset, subset_id)
    
    loader = DataLoader(dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Training on {subset_size} molecules (10% of total).")

    # 3. Setup Logging for plot_zinc.py
    history = []
    log_file = args.output_model_file.replace('.pth', '_history.csv')

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_vq = 0
        epoch_atom = 0
        epoch_chiral = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            _, _, vq_loss, p_atom, p_chiral = model(batch)
            
            # Calculate atom + chiral reconstruction losses (CrossEntropy)
            loss_atom = criterion(p_atom, batch.x[:, 0].long())
            loss_chiral = criterion(p_chiral, batch.x[:, 1].long())
            recon_loss = loss_atom + loss_chiral

            # Total Loss = Reconstruction + VQ Commitment
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_vq += vq_loss.item()
            epoch_atom += loss_atom.item()
            epoch_chiral += loss_chiral.item()
            
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Calculate averages for logging
        avg_loss = epoch_loss / len(loader)
        avg_vq = epoch_vq / len(loader)
        avg_atom = epoch_atom / len(loader)
        avg_chiral = epoch_chiral / len(loader)
        acc= compute_accuracy(p_atom, batch.x[:, 0].long())
        print(f"-> Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Atom Loss: {avg_atom:.4f} | Chiral Loss: {avg_chiral:.4f} | VQ Loss: {avg_vq:.4f} | Accuracy: {acc:.4f}")

        # Store metrics in history
        history.append({
            'epoch': epoch,
            'total_loss': avg_loss,
            'vq_loss': avg_vq,
            'atom_loss': avg_atom,
            'chiral_loss': avg_chiral,
        })

        # Checkpoint: Save weights and logs every 10 epochs
        if epoch % 100 == 0:
            pd.DataFrame(history).to_csv(log_file, index=False)
            torch.save(model.encoder.state_dict(), f"{args.output_model_file}_epoch{epoch}.pth")
            print(f"Checkpointed at epoch {epoch}")

    # 5. Final Exports
    # Save the encoder specifically (used for loading in pretrain/finetune)
    torch.save(model.encoder.state_dict(), args.output_model_file)
    # Save the full CSV for plotting
    pd.DataFrame(history).to_csv(log_file, index=False)
    
    print("-" * 30)
    print(f"SUCCESS: Training Complete.")
    print(f"Encoder weights: {args.output_model_file}")
    print(f"Loss history (CSV): {log_file}")

if __name__ == "__main__":
    main()