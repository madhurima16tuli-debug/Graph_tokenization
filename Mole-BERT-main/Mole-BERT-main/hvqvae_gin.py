import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random
import pandas as pd
import os

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader
from model import GNN, VectorQuantizer, GNNDecoder

# --- Model Definition ---
class HVQVAE_GIN(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, num_tokens=512, drop_ratio=0.1):
        super(HVQVAE_GIN, self).__init__()
        self.encoder = GNN(num_layer=num_layer, emb_dim=emb_dim, JK="last", drop_ratio=drop_ratio, gnn_type="gin")
        self.codebook = VectorQuantizer(emb_dim, num_tokens)
        self.atom_decoder = GNNDecoder(emb_dim, 120)  # num_atom_type
        self.chiral_decoder = GNNDecoder(emb_dim, 4)  # num_chirality_tag

    def forward(self, data):
        node_rep = self.encoder(data.x, data.edge_index, data.edge_attr)
        quantized, vq_loss = self.codebook(data.x, node_rep)
        p_atom = self.atom_decoder(quantized, data.edge_index, data.edge_attr)
        p_chiral = self.chiral_decoder(quantized, data.edge_index, data.edge_attr)
        return quantized, vq_loss, p_atom, p_chiral

    # --- ADDED: Essential for pretrain.py ---
    @torch.no_grad()
    def get_codebook_indices(self, data):
        """
        Generates discrete labels for pre-training.
        Maps continuous GIN node representations to codebook indices.
        """
        # Get node embeddings from GIN
        node_rep = self.encoder(data.x, data.edge_index, data.edge_attr)
        
        # Return indices based on chemistry-aware partitioning (C, N, O, etc.)
        return self.codebook.get_codebook_indices(data.x, node_rep)

# --- Training Logic ---
def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    total_vq = 0
    total_atom = 0
    total_chiral = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        _, vq_loss, p_atom, p_chiral = model(batch)
        
        # Reconstruction Losses
        loss_atom = F.cross_entropy(p_atom, batch.x[:, 0].long())
        loss_chiral = F.cross_entropy(p_chiral, batch.x[:, 1].long())
        recon_loss = loss_atom + loss_chiral

        loss = recon_loss + vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_vq += vq_loss.item()
        total_atom += loss_atom.item()
        total_chiral += loss_chiral.item()
        
    n = len(loader)
    return total_loss/n, total_vq/n, total_atom/n, total_chiral/n

def main():
    parser = argparse.ArgumentParser(description='GIN-based HVQVAE Tokenizer Training')
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent', help='Dataset name')
    parser.add_argument('--output_model_file', type=str, default='gin_vqvae_model.pth', help='Path to save weights')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = MoleculeDataset(f"./dataset/{args.dataset}", dataset=args.dataset)
    num_molecules = len(dataset)
    indices = list(range(num_molecules))
    random.seed(42)
    random.shuffle(indices)
    subset_id = indices[:int(num_molecules*0.1)] 
    dataset = torch.utils.data.Subset(dataset, subset_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = HVQVAE_GIN(emb_dim=args.emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    log_file = args.output_model_file.replace('.pth', '_history.csv')

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        avg_loss, avg_vq, avg_atom, avg_chiral = train(model, device, loader, optimizer)
        print(f"Loss: {avg_loss:.4f} (VQ: {avg_vq:.4f}, Atom: {avg_atom:.4f}, Chiral: {avg_chiral:.4f})")

        history.append({
            'epoch': epoch,
            'total_loss': avg_loss,
            'vq_loss': avg_vq,
            'atom_reconstruction_loss': avg_atom,
            'chiral_reconstruction_loss': avg_chiral
        })

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_model_file}_epoch{epoch}.pth")
            pd.DataFrame(history).to_csv(log_file, index=False)

    torch.save(model.state_dict(), args.output_model_file)
    pd.DataFrame(history).to_csv(log_file, index=False)
    print(f"Training history saved to {log_file}")

if __name__ == "__main__":
    main()