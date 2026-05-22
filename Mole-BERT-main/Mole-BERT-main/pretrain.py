import argparse
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.nn import global_mean_pool

from loader import MoleculeDataset
from dataloader import DataLoaderMaskingPred
from model import GNN 

# Import both architectures
from hvqvae import HVQVAE
from hvqvae_gin import HVQVAE_GIN

# --- Loss Functions ---
triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
criterion = nn.CrossEntropyLoss()

class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x_node = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x_node, batch)
        x = self.projection_head(x)
        return x_node, x

    def loss_cl(self, x1, x2):
        T = 0.2 # Temperature for contrastive stability
        eps = 1e-8
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)
        sim_matrix = torch.mm(x1_norm, x2_norm.t()) / T
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim_matrix)
        pos_sim = torch.exp(torch.sum(x1_norm * x2_norm, dim=-1) / T)
        loss = -torch.log(pos_sim / (exp_sim.sum(dim=1) + eps)).mean()
        # if loss.item() < 0:
        #     print("Warning: Negative loss value encountered in contrastive loss. Check for numerical stability issues.")
        #     breakpoint()
        return loss

    def loss_tri(self, x, x1, x2):
        return triplet_loss(x, x1, x2)

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, epoch, model_list, tokenizer, dataset, optimizer_list, device):
    model, linear_pred_atoms1, _, linear_pred_atoms2, _ = model_list
    optimizer_model, optimizer_atoms1, _, optimizer_atoms2, _ = optimizer_list
    
    model.train()
    linear_pred_atoms1.train()
    linear_pred_atoms2.train()

    loss_accum, acc_node_accum = 0, 0
    # .shuffle() works on the subset as well
    dataset1 = dataset
    dataset2 = copy.deepcopy(dataset1)
    
    loader1 = DataLoaderMaskingPred(dataset1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, mask_rate=args.mask_rate1, mask_edge=args.mask_edge)
    loader2 = DataLoaderMaskingPred(dataset2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, mask_rate=args.mask_rate2, mask_edge=args.mask_edge)

    epoch_iter = tqdm(zip(loader1, loader2), desc="Iteration", total=len(loader1))
    for step, (batch1, batch2) in enumerate(epoch_iter):
        batch1, batch2 = batch1.to(device), batch2.to(device)
        
        node_rep1, graph_rep1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        node_rep2, graph_rep2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        
        loss_cl = model.loss_cl(graph_rep1, graph_rep2)

        with torch.no_grad():
            batch_origin_x = copy.deepcopy(batch1.x)
            batch_origin_x[batch1.masked_atom_indices] = batch1.mask_node_label
            batch_origin_edge = copy.deepcopy(batch1.edge_attr)
            batch_origin_edge[batch1.connected_edge_indices] = batch1.mask_edge_label   
            batch_origin_edge[batch1.connected_edge_indices + 1] = batch1.mask_edge_label 
            batch1.x, batch1.edge_attr = batch_origin_x, batch_origin_edge
            
            atom_ids = tokenizer.get_codebook_indices(batch1)
            labels1, labels2 = atom_ids[batch1.masked_atom_indices], atom_ids[batch2.masked_atom_indices]
            _, graph_rep = model.forward_cl(batch_origin_x, batch1.edge_index, batch_origin_edge, batch1.batch)

        loss_tri = model.loss_tri(graph_rep, graph_rep1, graph_rep2)
        #print("loss tri=",loss_tri)
        #print("loss cl=",loss_cl)

        loss_tricl = loss_cl + 0.1 * loss_tri

        pred_node1, pred_node2 = linear_pred_atoms1(node_rep1[batch1.masked_atom_indices]), linear_pred_atoms2(node_rep2[batch2.masked_atom_indices])
        loss_mask = criterion(pred_node1, labels1) + criterion(pred_node2, labels2)

        acc_node = (compute_accuracy(pred_node1, labels1) + compute_accuracy(pred_node2, labels2)) * 0.5
        acc_node_accum += acc_node

        loss = loss_tricl + loss_mask
        optimizer_model.zero_grad(); optimizer_atoms1.zero_grad(); optimizer_atoms2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Stability fix
        optimizer_model.step(); optimizer_atoms1.step(); optimizer_atoms2.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f} tacc: {acc_node:.4f}")

    return loss_accum/max(step, 1), acc_node_accum/max(step, 1)

def main():
    parser = argparse.ArgumentParser(description='Mole-BERT Stable Multi-Tokenizer Pre-training')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--num_tokens', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent')
    parser.add_argument('--output_model_file', type=str, default='pretrain_molebert')
    parser.add_argument('--tokenizer_type', type=str, default='gin', choices=['gin', 'gpm'])
    parser.add_argument('--tokenizer_model_file', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mask_edge', type=int, default=1)
    parser.add_argument('--mask_rate1', type=float, default=0.15); 
    parser.add_argument('--mask_rate2', type=float, default=0.30)
    parser.add_argument('--num_layer', type=int, default=5); 
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument("--gpm_num_patterns", type=int, default=10);
     parser.add_argument("--gpm_walk_length", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    
    # --- Load and Subset Dataset ---
    full_dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset) 
    
    num_molecules = len(full_dataset)
    indices = list(range(num_molecules))
    random.seed(42) # ENSURES CONSISTENCY
    random.shuffle(indices)
    
    subset_size = int(num_molecules * 0.1)
    subset_id = indices[:subset_size]
    dataset_subset = torch.utils.data.Subset(full_dataset, subset_id)
    print(f"Pre-training on 10% subset: {len(dataset_subset)} molecules")

    # 1. Initialize Main Model
    gnn = GNN(args.num_layer, args.emb_dim, JK="last", drop_ratio=0.1, gnn_type=args.gnn_type)
    model = graphcl(gnn).to(device)

    # 2. Initialize Tokenizer
    if args.tokenizer_type == 'gpm':
        tokenizer = HVQVAE(emb_dim=args.emb_dim, num_tokens=args.num_tokens, num_patterns=args.gpm_num_patterns, walk_length=args.gpm_walk_length).to(device)
    else:
        tokenizer = HVQVAE_GIN(num_layer=args.num_layer, emb_dim=args.emb_dim, num_tokens=args.num_tokens).to(device)

    state_dict = torch.load(args.tokenizer_model_file, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('encoder.') and not k.startswith('codebook.') and not k.startswith('atom_decoder.'):
            new_state_dict[f'encoder.{k}'] = v
        else:
            new_state_dict[k] = v
    
    msg = tokenizer.load_state_dict(new_state_dict, strict=False)
    print(f"Loading Tokenizer weights: {msg}")
    tokenizer.eval()

    # 3. Heads & Optimizers
    linear_pred_atoms1 = torch.nn.Linear(args.emb_dim, args.num_tokens).to(device)
    linear_pred_atoms2 = torch.nn.Linear(args.emb_dim, args.num_tokens).to(device)
    model_list = [model, linear_pred_atoms1, None, linear_pred_atoms2, None]
    
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_atoms1 = optim.Adam(linear_pred_atoms1.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_atoms2 = optim.Adam(linear_pred_atoms2.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_list = [optimizer_model, optimizer_atoms1, None, optimizer_atoms2, None]

    history = []
    for epoch in range(1, args.epochs + 1):
        # PASS dataset_subset HERE
        loss, acc = train(args, epoch, model_list, tokenizer, dataset_subset, optimizer_list, device)
        history.append({'epoch': epoch, 'loss': loss, 'acc': acc})
        if epoch % 10 == 0:
            pd.DataFrame(history).to_csv(f"{args.output_model_file}_history.csv", index=False)
            torch.save(model.gnn.state_dict(), f"{args.output_model_file}_epoch{epoch}.pth")

    torch.save(model.gnn.state_dict(), f"{args.output_model_file}_final.pth")

if __name__ == "__main__":
    main()