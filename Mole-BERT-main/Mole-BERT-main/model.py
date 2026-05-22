import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_cluster import random_walk
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import einsum
import copy

# Constants from Mole-BERT and HVQVAE configurations
num_atom_type = 120 
num_chirality_tag = 3
num_bond_type = 6 
num_bond_direction = 3 
NUM_NODE_ATTR = 119
NUM_NODE_CHIRAL = 4
NUM_BOND_ATTR = 4

# ───────────────────────────────────────────────────────────────────────
# GPM & VQ COMPONENTS (Architecture Definitions)
# ───────────────────────────────────────────────────────────────────────

def _l2norm(t):
    return F.normalize(t, p=2, dim=-1)

class VectorQuantizer(nn.Module):
    """Chemistry-aware Vector Quantizer"""
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x, e):
        #This calculates the distance between the incoming 
        #continuous embedding e and every vector in your codebook. It returns the index of the closest one.
        encoding_indices = self.get_codebook_indices(x, e) 
        #This replaces the continuous vector e with the actual vector stored at that index in the codebook.
        quantized = self.quantize(encoding_indices)
        q_latent_loss = F.mse_loss(quantized, e.detach())
        e_latent_loss = F.mse_loss(e, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, loss

    def get_codebook_indices(self, x, e):
        atom_type = x[:, 0]
        index_c = (atom_type == 5)
        index_n = (atom_type == 6)
        index_o = (atom_type == 7)
        index_others = ~(index_c | index_n | index_o)
        encoding_indices = torch.ones(x.size(0), dtype=torch.long, device=x.device)

        def _nearest(e_sub, weight_slice):
            if e_sub.size(0) == 0: return torch.empty(0, dtype=torch.long, device=x.device)
            d = torch.sum(e_sub ** 2, dim=1, keepdim=True) + torch.sum(weight_slice ** 2, dim=1) - 2.0 * torch.matmul(e_sub, weight_slice.t())
            return torch.argmin(d, dim=1)

        w = self.embeddings.weight
        encoding_indices[index_c] = _nearest(e[index_c], w[0:377])
        encoding_indices[index_n] = _nearest(e[index_n], w[378:433]) + 378
        encoding_indices[index_o] = _nearest(e[index_o], w[434:488]) + 434
        encoding_indices[index_others] = _nearest(e[index_others], w[489:511]) + 489
        return encoding_indices

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)

@torch.no_grad()
def _sample_patterns(mol, num_patterns, walk_length):
    """Random-walk sampler for GPM"""
    row, col = mol.edge_index.cpu()
    N = mol.num_nodes
    if mol.edge_index.shape[1] == 0:
        return torch.zeros(num_patterns, N, walk_length + 1, dtype=torch.long), torch.zeros(num_patterns, N, walk_length, dtype=torch.long)
    start = torch.arange(N).repeat(num_patterns)
    walks, edge_ids = random_walk(row, col, start=start, walk_length=walk_length, return_edge_indices=True)
    return walks.view(num_patterns, N, walk_length + 1), edge_ids.view(num_patterns, N, walk_length)

class GPMEncoder(nn.Module):
    """GPM Encoder using random walks and Transformer fusion"""
    def __init__(self, hidden_dim=300, num_patterns=10, walk_length=5, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.walk_length = walk_length
        self.atom_encoder = AtomEncoder(emb_dim=kwargs.get('atom_embed_dim', 16))
        self.bond_encoder = BondEncoder(emb_dim=kwargs.get('bond_embed_dim', 16))
        #self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)   # 300-dim
        #self.bond_encoder = BondEncoder(emb_dim=hidden_dim)   # 300-dim

        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        #num_tf_layers = kwargs.get('num_transformer_layers', 4)
        self.pattern_transformer = nn.TransformerEncoder(layer, num_layers=2)
        #layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=6, batch_first=True,
        #norm_first=True,        # pre-norm: stable gradients
        #dropout=0.1,
        #dim_feedforward=hidden_dim * 4)
        #self.pattern_transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.pre_proj = nn.Linear(16 + 16, hidden_dim)
        #self.pre_proj = nn.Linear(hidden_dim + hidden_dim, hidden_dim)  # 600 → 300
        self.pool = global_mean_pool
        #self.pattern_attn = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        atom_feat = self.atom_encoder(data.x)
        bond_feat = self.bond_encoder(data.edge_attr) if data.edge_attr is not None else torch.zeros(data.edge_index.shape[1], 16, device=data.x.device)
        
        patterns, eids = _sample_patterns(data, self.num_patterns, self.walk_length)
        patterns, eids = patterns.to(data.x.device), eids.to(data.x.device)
        
        h, n, kp1 = patterns.shape
        atom_seq = atom_feat[patterns.view(-1)].view(h * n, kp1, -1)
        bond_seq = torch.zeros(h * n, kp1, 16, device=data.x.device) 
        # replace the bond_seq line with:
        #flat_eids = eids.view(-1)                          # (h*n*walk_length,)
        #valid = flat_eids >= 0                             # -1 = dead-end padding
        #bond_embed = torch.zeros(h * n * self.walk_length, self.hidden_dim, device=data.x.device)
        #bond_embed[valid] = bond_feat[flat_eids[valid]]
        #bond_seq = bond_embed.view(h * n, self.walk_length, self.hidden_dim)
        # pad to walk_length+1 to match atom_seq length
        #bond_seq = F.pad(bond_seq, (0, 0, 0, 1))          # append zero for the terminal node
        
        feat_seq = self.pre_proj(torch.cat([atom_seq, bond_seq], dim=-1))
        #positions = torch.arange(kp1, device=data.x.device).unsqueeze(0)
        #feat_seq = feat_seq + self.pos_embedding(positions)
        #out = self.pattern_transformer(feat_seq)           # (h*n, k+1, d)
        #step_rep = out.mean(1).view(h, n, -1)              # (h, n, d)
        #attn_w = torch.softmax(self.pattern_attn(step_rep), dim=0)   # (h, n, 1)
        #node_rep = (attn_w * step_rep).sum(0) 
        node_rep = self.pattern_transformer(feat_seq).mean(1).view(h, n, -1).mean(0)
        
        return self.pool(node_rep, data.batch), node_rep

class GPM_graphpred(nn.Module):
    """Graph prediction model using GPM backbone for finetuning"""
    def __init__(self, args, num_tasks):
        super().__init__()
        self.gnn = GPMEncoder(
            hidden_dim=args.emb_dim, 
            num_patterns=args.gpm_num_patterns, 
            walk_length=args.gpm_walk_length
        )
        self.graph_pred_linear = nn.Linear(args.emb_dim, num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        graph_rep, _ = self.gnn(data)
        return self.graph_pred_linear(graph_rep), None

# ───────────────────────────────────────────────────────────────────────
# ORIGINAL MOLE-BERT GIN COMPONENTS (Unchanged)
# ───────────────────────────────────────────────────────────────────────

class GINConv(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        super(GINConv, self).__init__(aggr=aggr)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_j, edge_attr):
        return x_j

class GNNDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, JK="last", gnn_type="gin"):
        super().__init__()
        self.conv = GINConv(hidden_dim, out_dim)
    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)

class GNN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        # 1. ADD THESE: Map discrete atom/chirality indices to emb_dim
        # num_atom_type is usually 120; num_chirality_tag is 3
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # 2. Define GNN layers
        self.gnns = torch.nn.ModuleList([GINConv(emb_dim, emb_dim) for _ in range(num_layer)])
        
        # 3. Batch Norm layers (now correctly expecting emb_dim)
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x, edge_index, edge_attr):
    # This must match the embedding layers defined in __init__
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        
        for i in range(self.num_layer):
            x = self.gnns[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        return x

# model.py

class GNN_graphpred(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        # This uses the GNN class defined earlier in model.py
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # Loads the weights we just pre-trained
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, x, edge_index, edge_attr, batch):
        node_representation = self.gnn(x, edge_index, edge_attr)
        return self.graph_pred_linear(self.pool(node_representation, batch)), node_representation