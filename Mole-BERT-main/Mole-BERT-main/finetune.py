import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader
from splitters import scaffold_split, random_split, random_scaffold_split

# Ensure GPM_graphpred and GNN_graphpred are both defined in your model.py
from model import GNN_graphpred, GPM_graphpred

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, epoch, model, device, loader, optimizer):
    model.train()
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)

        # Branching logic to handle GPM vs GNN input signatures
        if isinstance(model, GPM_graphpred):
            pred, _ = model(batch)
        else:
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(
            is_valid,
            loss_mat,
            torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
        )

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            if isinstance(model, GPM_graphpred):
                pred, _ = model(batch)
            else:
                pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(
                roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i])
            )

    return sum(roc_list) / len(roc_list) if len(roc_list) > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Finetuning of GNN or HVQVAE-GPM models"
    )
    # Standard Finetuning Args
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--graph_pooling", type=str, default="mean")
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--dataset", type=str, default="sider")
    parser.add_argument(
        "--input_model_file", type=str, default="None", help="Pretrained encoder path"
    )
    parser.add_argument(
        "--model_backbone", type=str, default="gpm", choices=["gin", "gpm"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runseed", type=int, default=0)
    parser.add_argument("--split", type=str, default="scaffold")
    parser.add_argument("--num_workers", type=int, default=4)

    # GPM-specific Arguments (Must match your pre-training settings)
    parser.add_argument("--gpm_num_patterns", type=int, default=10)
    parser.add_argument("--gpm_walk_length", type=int, default=5)
    parser.add_argument("--gpm_p", type=float, default=1.0)
    parser.add_argument("--gpm_q", type=float, default=1.0)
    parser.add_argument("--gpm_pe_encoder", type=str, default="none")
    parser.add_argument("--gpm_pattern_encoder", type=str, default="transformer")
    parser.add_argument("--gpm_pattern_enc_heads", type=int, default=4)
    parser.add_argument("--gpm_pattern_enc_layers", type=int, default=2)
    parser.add_argument("--gpm_use_vq", type=int, default=1)
    parser.add_argument("--gpm_codebook_size", type=int, default=512)
    parser.add_argument("--gpm_vq_heads", type=int, default=4)
    parser.add_argument("--gpm_transformer_layers", type=int, default=3)
    parser.add_argument("--gpm_transformer_heads", type=int, default=4)
    parser.add_argument("--gpm_atom_embed_dim", type=int, default=16)
    parser.add_argument("--gpm_bond_embed_dim", type=int, default=16)

    parser.add_argument("--filename", type=str, default="", help="output filename")
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Dataset Task Mapping
    dataset_tasks = {
        "tox21": 12,
        "hiv": 1,
        "pcba": 128,
        "muv": 17,
        "bace": 1,
        "bbbp": 1,
        "toxcast": 617,
        "sider": 27,
        "clintox": 2,
    }
    num_tasks = dataset_tasks.get(args.dataset, 0)

    # Use your absolute path
    # dataset_root = "/mnt/hdd/gtoken/data1/mole_bert/Mole-BERT-main/Mole-BERT-main/dataset/chem_dataset/dataset"
    # dataset = MoleculeDataset(dataset_root, dataset=args.dataset)

    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)

    # Split selection
    smiles_csv = os.path.join(dataset, "processed", "smiles.csv")
    smiles_list = pd.read_csv(smiles_csv, header=None)[0].tolist()

    if args.split == "scaffold":
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1
        )
    else:
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model Setup
    if args.model_backbone == "gpm":
        model = GPM_graphpred(args, num_tasks)
        if args.input_model_file != "None":
            print(f"Loading GPM weights from {args.input_model_file}")
            model.from_pretrained(args.input_model_file)
    else:
        model = GNN_graphpred(
            args.num_layer,
            args.emb_dim,
            num_tasks,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            graph_pooling=args.graph_pooling,
            gnn_type=args.gnn_type,
        )
        if args.input_model_file != "None":
            model.from_pretrained(args.input_model_file)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # Finetuning Loop
    for epoch in range(1, args.epochs + 1):
        train(args, epoch, model, device, train_loader, optimizer)
        val_auc = eval(args, model, device, val_loader)
        test_auc = eval(args, model, device, test_loader)
        print(f"Epoch {epoch} | val_auc: {val_auc:.4f} | test_auc: {test_auc:.4f}")


if __name__ == "__main__":
    main()

