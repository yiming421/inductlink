import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul, SparseTensor
from torch_sparse.matmul import spmm_add
from torch import Tensor
from util import adjoverlap, get_entropy_normed_cond_gaussian_prob
import math
import torchhd
from torch_geometric.utils import negative_sampling

class LinearPredictor(nn.Module):
    def __init__(self, in_channels):
        super(LinearPredictor, self).__init__()
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, x_i, x_j):
        return self.linear(x_i * x_j).squeeze(-1)

class DotPredictor(nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def forward(self, x_i, x_j):
        # print(x_i.size(), x_j.size(), flush=True)
        return torch.sum(x_i * x_j, dim=-1)

class CosinePredictor(nn.Module):
    def __init__(self):
        super(CosinePredictor, self).__init__()

    def forward(self, x_i, x_j):
        x = torch.nn.functional.cosine_similarity(x_i, x_j, dim=-1)
        return x

class ManhattanPredictor(nn.Module):
    def __init__(self):
        super(ManhattanPredictor, self).__init__()

    def forward(self, x_i, x_j):
        x = torch.sum(torch.abs(x_i - x_j), dim=-1)
        return x

class LorentzPredictor(nn.Module):
    def __init__(self):
        super(LorentzPredictor, self).__init__()

    def forward(self, x_i, x_j):
        n = x_i.size(-1)
        x = torch.sum(x_i[:, 0:n//2] * x_j[:, 0:n//2], dim=-1) - torch.sum(x_i[:, n//2:] * x_j[:, n//2:], dim=-1)
        return x

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False, norm=False, scale=False):
        super().__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(h_feats, h_feats))
        # Create a separate LayerNorm for each layer (except perhaps the final one)
        self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(layer - 1)]) if norm else None
        
        for _ in range(layer - 2):
            self.lins.append(nn.Linear(h_feats, h_feats))
        self.lins.append(nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.norm = norm
        self.h_feats = h_feats
        if scale:
            self.scale = nn.LayerNorm(h_feats)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if hasattr(self, 'scale'):
            x = self.scale(x)
        ori = x
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.res:
                x = x + ori
            if self.norm:
                x = self.norms[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()


class NCNPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout,
                 beta=1.0):
                 
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 nn.LayerNorm(hidden_channels),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels))

    def multidomainforward(self,
                           x,
                           adj_t,
                           tar_ei):
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj_t, adj_t, tar_ei)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs.squeeze()

    def forward(self, x, adj_t, tar_ei):
        return self.multidomainforward(x, adj_t, tar_ei)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,norm=False, tailact=False):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if norm:
            self.lins.append(nn.LayerNorm(hidden_channels))
        self.lins.append(nn.ReLU())
        if dropout > 0:
            self.lins.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if norm:
                self.lins.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.ReLU())
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        if tailact:
            self.lins.append(nn.LayerNorm(out_channels))
            self.lins.append(nn.ReLU())
            self.lins.append(nn.Dropout(dropout))

    def forward(self, x):
        x = self.lins(x)
        return x.squeeze()

class inductive_GCN(MessagePassing):
    def __init__(self, layers, alpha=0.5, dropout=0.2, mlp_layers=2, mlp_hidden=256, att_temp=1, entropy=1, agg_type='add_att'):
        super(inductive_GCN, self).__init__(aggr='add')
        self.layers = layers
        self.agg_type = agg_type
        if agg_type == 'add_att':
            self.alpha = alpha
            self.alphas = Parameter(torch.Tensor(alpha ** np.arange(layers + 1)), requires_grad=True)
        elif agg_type == 'mlp_att':
            self.dim = layers * (layers + 1)
            self.mlp = MLP(self.dim, mlp_hidden, self.layers + 1, mlp_layers, dropout)
            self.att_temp = att_temp
            self.entropy = entropy
        elif agg_type == 'transformer_att':
            raise NotImplementedError
        else:
            raise ValueError('Unknown aggregation type')
        
    def compute_dist(self, y_feat):
        bsz, n_channel, n_class = y_feat.shape
        # Conditional gaussian probability
        cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
        for i in range(bsz):
            cond_gaussian_prob[i, :, :] = get_entropy_normed_cond_gaussian_prob(
                y_feat[i, :, :].cpu().numpy(), self.entropy
            )

        # Compute pairwise distances between channels n_channels(n_channels-1)/2 total features
        dist = np.zeros((bsz, self.dim), dtype=np.float32)

        pair_index = 0
        for c in range(n_channel):
            for c_prime in range(n_channel):
                if c != c_prime:  # Diagonal distances are useless
                    dist[:, pair_index] = cond_gaussian_prob[:, c, c_prime]
                    pair_index += 1

        dist = torch.from_numpy(dist).to(y_feat.device)
        return dist

    def forward(self, x, adj_t, edge_weight=None):
        xs = []
        adj_t = gcn_norm(adj_t, edge_weight, adj_t.size(0), dtype=float)
        xs.append(x)
        for i in range(self.layers):
            x = self.propagate(adj_t, x=x, edge_weight=edge_weight, size=None)
            xs.append(x)
        if self.agg_type == 'add_att':
            res = xs[0] * self.alphas[0]
            for i in range(self.layers):
                res = res + xs[i + 1] * self.alphas[i + 1]
        elif self.agg_type == 'mlp_att':
            y_feat = torch.stack(xs, dim=1)
            dist = self.compute_dist(y_feat)
            att = self.mlp(dist)
            att = torch.softmax(att / self.att_temp, dim=-1)
            res = torch.sum(y_feat * att.unsqueeze(2), dim=1)
        return res
    
    def message_and_aggregate(self, adj_t, x, edge_weight):
        return matmul(adj_t, x, reduce='add')
    
class inductive_GCN_feat(nn.Module):
    def __init__(self, layers, input, hidden, dropout=0, relu=False, linear=False, conv='gcn'):
        super(inductive_GCN_feat, self).__init__()
        self.layers = layers
        if conv == 'gcn':
            self.conv1 = GCNConv(input, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif conv == 'sage':
            self.conv1 = SAGEConv(input, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        else:
            raise ValueError('Unknown convolution')
        self.layers = layers
        self.relu = relu
        self.dropout = dropout
        self.linear = linear
        self.norm = nn.LayerNorm(hidden)
        if linear:
            self.lin = nn.Linear(hidden, hidden)

    def forward(self, x, adj_t, reduce_model):
        x = reduce_model(x)
        x = self.conv1(x, adj_t)
        for i in range(self.layers - 1):
            if self.relu:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            if self.linear:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
                x = self.lin(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            x = self.conv2(x, adj_t)

        return x

class inductive_GCN_no_feat(nn.Module):
    def __init__(self, layers, hidden, residual=0, dropout=0, relu=False, linear=False, conv='gcn'):
        super(inductive_GCN_no_feat, self).__init__()
        self.layers = layers
        if conv == 'gcn':
            self.conv1 = GCNConv(hidden, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif conv == 'sage':
            self.conv1 = SAGEConv(hidden, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        else:
            raise ValueError('Unknown convolution')
        self.layers = layers
        self.relu = relu
        self.dropout = dropout
        self.linear = linear
        if linear:
            self.lin = nn.Linear(hidden, hidden)
        self.residual = residual

    def forward(self, x, adj_t):
        ori = x
        x = self.conv1(x, adj_t) + self.residual * ori
        for i in range(self.layers - 1):
            if self.relu:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            if self.linear:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
                x = self.lin(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            x = self.conv2(x, adj_t) + self.residual * ori
        return x
    
MINIMUM_SIGNATURE_DIM = 64
class MPLPNodeLabel(torch.nn.Module):
    def __init__(self, dim: int=1024, signature_sampling="torchhd", minimum_degree_onehot: int=-1, hop: int=2):
        super().__init__()
        self.dim = dim
        self.signature_sampling = signature_sampling
        self.minimum_degree_onehot = minimum_degree_onehot
        self.hop = hop

    def forward(self, edge: Tensor, adj_t: SparseTensor, deg: Tensor):
        x = self.get_random_node_vectors(adj_t)
        D = x.shape[-1]
        degrev = torch.reciprocal(deg)

        x = torch.concat((x, x*torch.rsqrt(deg), x*torch.rsqrt(1+torch.log(deg))), dim=1)
        xs = [x, degrev*(adj_t@x)]
        for i in range(2, self.hop+1):
            xs.append(degrev*(adj_t @ xs[-1]) - xs[-2])
        x = torch.concat(xs, dim=1).unflatten(1, (-1, D))
        ret = x[edge[0]] @ x[edge[1]].transpose(1, 2)
        return (ret + ret.transpose(1, 2)).flatten(1, 2)
    
    def get_random_node_vectors(self, adj_t: SparseTensor) -> Tensor:
        num_nodes = adj_t.size(0)
        device = adj_t.device()
        if self.minimum_degree_onehot > 0:
            degree = adj_t.sum(dim=1)
            nodes_to_one_hot = degree >= self.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            if one_hot_dim + MINIMUM_SIGNATURE_DIM > self.dim:
                raise ValueError(f"There are {int(one_hot_dim)} nodes with degree higher than {self.minimum_degree_onehot}, select a higher threshold to choose fewer nodes as hub")
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            if one_hot_dim>0:
                one_hot_embedding = F.one_hot(torch.arange(0, one_hot_dim)).float().to(device)
                embedding[nodes_to_one_hot,:one_hot_dim] = one_hot_embedding
        else:
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            nodes_to_one_hot = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            one_hot_dim = 0
        rand_dim = self.dim - one_hot_dim

        if self.signature_sampling == "torchhd":
            scale = math.sqrt(1 / rand_dim)
            node_vectors = torchhd.random(num_nodes - one_hot_dim, rand_dim, device=device)
            node_vectors.mul_(scale)  # make them unit vectors
        elif self.signature_sampling == "gaussian":
            node_vectors = F.normalize(torch.nn.init.normal_(torch.empty((num_nodes - one_hot_dim, rand_dim), dtype=torch.float32, device=device)))
        elif self.signature_sampling == "onehot":
            embedding = torch.zeros(num_nodes, num_nodes, device=device)
            node_vectors = F.one_hot(torch.arange(0, num_nodes)).float().to(device)

        embedding[~nodes_to_one_hot, one_hot_dim:] = node_vectors
        return embedding

class NodeFeat(torch.nn.Module):
    def __init__(self, hop: int=2):
        super().__init__()
        self.hop = hop

    def forward(self, x: Tensor, adj_t: SparseTensor, deg: Tensor):
        D = x.shape[-1]
        degrev = torch.reciprocal(deg)

        x = torch.concat((x, x*torch.rsqrt(deg), x*torch.sqrt(deg)), dim=1)
        xs = [x, degrev*(adj_t@x)]
        for i in range(2, self.hop+1):
            xs.append(degrev*(adj_t @ xs[-1]) - xs[-2])
        x = torch.concat(xs, dim=1).unflatten(1, (-1, D)).transpose(1, 2) # N, D, C
        return x

class UniPred(nn.Module):

    def __init__(self, hop: int, hiddim: int) -> None:
        super().__init__()
        hop = hop + 1
        attendim = int((3*hop*hiddim)**0.5)
        attendim = ((attendim+4-1)//4) * 4
        self.nodefeatlin = MLP(3*hop, attendim, attendim, 2, 0.0)
        self.nodefeatlin2 = MLP(attendim, hiddim, hiddim, 2, 0.0)
        self.transenc = nn.MultiheadAttention(attendim, 4, batch_first=True)
        self.hiddim = hiddim
        self.linpred = nn.Linear(hiddim, 1)

    def forward(self, nodefeat: Tensor):
        '''
        linklabel: (#E, hop*3*3*hop)
        nodefeat: (2, #E, D, hop*3)
        '''
        nodefeat = self.nodefeatlin(nodefeat)
        nodefeat = self.transenc.forward(nodefeat[[1, 0]].flatten(0, 1), nodefeat.flatten(0, 1), nodefeat.flatten(0, 1), need_weights=False)[0]
        nodefeat = nodefeat.unflatten(0, (2, -1))
        nodefeat = self.nodefeatlin2((nodefeat[0]*nodefeat[1]).mean(dim=-2))
        nodefeat = self.linpred(nodefeat)
        return nodefeat.squeeze()

class PureGCNConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, adj_t):
        norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
        x = norm * x
        x = spmm_add(adj_t, x) + x
        x = norm * x
        return x

class PureGCN(nn.Module):
    def __init__(self, num_layers=2) -> None:
        super().__init__()
        self.conv = PureGCNConv()
        self.num_layers = num_layers

    def forward(self, x, adj_t):
        for _ in range(self.num_layers):
            x = self.conv(x, adj_t)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureGCN_v1(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden=256, dp=0, norm=False, project='lin') -> None:
        super().__init__()
        
        # Input projection
        if project == 'lin':
            self.lin = nn.Linear(input_dim, hidden)
        elif project == 'mlp':
            self.lin = MLP(input_dim, hidden, hidden, 2, dp, norm)  # Ensure MLP is defined elsewhere
        elif project == 'id':
            self.lin = nn.Identity()
        else:
            raise ValueError('Unknown projection method:', project)
        
        # GCN Convolution Layer
        self.conv = PureGCNConv()
        self.num_layers = num_layers
        self.dp = dp
        self.norm = norm

        # Use separate LayerNorm instances per layer if normalization is enabled
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

    def forward(self, x, adj_t):
        x = self.lin(x)  # Apply input projection
        for i in range(self.num_layers):
            if self.norm:
                x = self.norms[i](x)  # Apply per-layer normalization
            if self.dp > 0:
                x = F.dropout(x, p=self.dp, training=self.training)
            x = self.conv(x, adj_t)  # Apply GCN convolution
        return x

# Combined model that encodes nodes (using a DeepSets-style encoder) and then predicts links in batches.
class DeepSetPredictor(nn.Module):
    def __init__(self, hidden_dim, embed_dim, dropout=0.5, mlp_layers=2, res=False, norm=False, scale=False):
        """
        Args:
            hidden_dim (int): Hidden dimension for the node encoder.
            embed_dim (int): Fixed dimension of node embeddings.
            dropout (float): Dropout probability for the predictor.
            mlp_layers (int): Number of layers in the Hadamard predictor.
            res (bool): Use residual connections in the predictor.
            norm (bool): Apply layer normalization in the predictor.
            scale (bool): Scale the Hadamard product in the predictor.
        """
        super(DeepSetPredictor, self).__init__()
        self.phi = MLP(1, hidden_dim, hidden_dim, 2, dropout, norm, False)
        self.rho = MLP(hidden_dim, embed_dim, 1, mlp_layers, dropout, norm, False)
        self.pool = PMA(dim=hidden_dim, num_heads=4, num_seeds=1)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = x.unsqueeze(-1)
        encoded = self.phi(x)
        aggregated = self.pool(encoded)
        embedding = self.rho(aggregated)
        return embedding.squeeze()

# PMA: Pooling by Multihead Attention
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        """
        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            num_seeds (int): Number of seed vectors (1 for a single global representation).
        """
        super(PMA, self).__init__()
        self.num_seeds = num_seeds
        self.seed = nn.Parameter(torch.Tensor(num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, X):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (batch, set_size, dim).
        Returns:
            torch.Tensor: Pooled output of shape (batch, num_seeds, dim).
        """
        batch_size = X.size(0)
        S = self.seed.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.mha(S, X, X)
        return pooled

class SetTransformerPredictor(nn.Module):
    def __init__(self, hidden_dim, embed_dim, dropout=0.2, mlp_layers=2, num_heads=4, num_layers=1, norm=False, pma=False):
        """
        Args:
            hidden_dim (int): Hidden dimension for the transformer.
            embed_dim (int): Final embedding dimension.
            dropout (float): Dropout probability.
            mlp_layers (int): Number of layers in the final MLP head.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
        """
        super(SetTransformerPredictor, self).__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        # self.transformer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
                                        dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        
        if pma:
            self.pma = PMA(dim=hidden_dim, num_heads=num_heads, num_seeds=1)
        self.rho = MLP(hidden_dim, embed_dim, 1, mlp_layers, dropout, norm, False)
    
    def forward(self, x_i, x_j):
        x = x_i * x_j  
        x = x.unsqueeze(-1) 
        x = self.input_proj(x)     
        # x, _ = self.transformer(x, x, x)    
        x = self.transformer(x)
        if hasattr(self, 'pma'):
            pooled = self.pma(x)
        else:
            pooled = torch.sum(x, dim=1, keepdim=True) 
        out = self.rho(pooled) 
        return out.squeeze(-1)
    
class PFNTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, n_head=1, mlp_layers=2, dropout=0.2, norm=False):
        super(PFNTransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.transformer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_head,
            dropout=dropout,
        )
        self.ffn = MLP(
            in_channels=hidden_dim,
            hidden_channels=4 * hidden_dim,
            out_channels=hidden_dim,
            num_layers=mlp_layers,
            dropout=dropout,
            norm=norm,
            tailact=False)
        self.context_norm1 = nn.LayerNorm(hidden_dim)
        self.context_norm2 = nn.LayerNorm(hidden_dim)
        self.tar_norm1 = nn.LayerNorm(hidden_dim)
        self.tar_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x_context, x_target):
        # x_context = self.context_norm1(x_context)
        x_context_att, _ = self.transformer(x_context, x_context, x_context)
        x_context = x_context_att + x_context
        x_context = self.context_norm1(x_context)

        # x_context = self.context_norm2(x_context)
        x_context_fnn = self.ffn(x_context).unsqueeze(1)
        x_context = x_context_fnn + x_context
        x_context = self.context_norm2(x_context)
        
        # x_target = self.tar_norm1(x_target)
        x_target_att, _ = self.transformer(x_target, x_context, x_context)
        x_target = x_target_att + x_target
        x_target = self.tar_norm1(x_target)

        # x_target = self.tar_norm2(x_target)
        x_target_fnn = self.ffn(x_target).unsqueeze(1)
        x_target = x_target_fnn + x_target
        x_target = self.tar_norm2(x_target)

        return x_context, x_target

class PFNPredictor(nn.Module):
    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2, norm=False, scale=False, 
                 dynamic=False, context_num=128, padding='zeros', premlp=False, 
                 output_target=False, column_att=False):
        super(PFNPredictor, self).__init__()
        # Store original hidden_dim for feature splitting
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # MLP for initial edge prediction (outputs 1 dimension for label prediction)
        if padding == 'mlp':
            self.mlp = MLP(hidden_dim, hidden_dim, 1, 2, dropout, norm, False)
        
        if premlp:
            self.premlp = MLP(hidden_dim, hidden_dim, hidden_dim, 2, dropout, norm, False)
        
        # Feature scaling
        self.scale = nn.LayerNorm(hidden_dim) if scale else None
        
        # Shared transformer components (dimension = hidden_dim + 1 for label concatenation)

        self.transformer_row = nn.ModuleList([
            PFNTransformerLayer(hidden_dim + 1, n_head=nhead, mlp_layers=2, dropout=dropout, norm=norm)
            for _ in range(num_layers)
        ])

        if column_att:
            self.transformer_col = nn.MultiheadAttention(
                embed_dim=1,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        
        # Final prediction head
        self.head = MLP(
            in_channels=hidden_dim + 1,
            hidden_channels=hidden_dim + 1,
            out_channels=1,
            num_layers=mlp_layers,
            dropout=dropout,
            norm=norm,
            tailact=False
        )
        self.dynamic = dynamic
        self.context_num = context_num
        self.padding = padding
        self.output_target = output_target
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data, x, edge):
        if self.dynamic:
            data.context_pos = data.edge_index[:, torch.randperm(data.edge_index.size(1))[:self.context_num // 2]].to(x.device)
            data.context_neg = negative_sampling(data.edge_index, num_nodes=data.num_nodes,
                                                num_neg_samples=self.context_num // 2).to(x.device)

        # Feature normalization
        if self.scale is not None:
            x = self.scale(x)

        # 1. Feature Preparation ---------------------------------------------
        # Edge features (target links)
        x_edge = x[edge[0]] * x[edge[1]]  # [num_edges, hidden_dim]
        
        # Context features (positive and negative examples)
        x_context_pos = x[data.context_pos[0]] * x[data.context_pos[1]]  # [num_pos, hidden_dim]
        x_context_neg = x[data.context_neg[0]] * x[data.context_neg[1]]  # [num_neg, hidden_dim]

        if hasattr(self, 'premlp'):
            x_edge = self.premlp(x_edge)
            x_context_pos = self.premlp(x_context_pos)
            x_context_neg = self.premlp(x_context_neg)
        
        # 2. Label Concatenation ----------------------------------------------
        # Add label indicators to context
        x_context_pos = torch.cat([
            x_context_pos, 
            torch.ones(x_context_pos.size(0), 1, device=x.device)
        ], dim=-1)
        x_context_neg = torch.cat([
            x_context_neg,
            torch.zeros(x_context_neg.size(0), 1, device=x.device)
        ], dim=-1)
        
        # Add predicted labels to target edges
        if self.padding == 'mlp':
            x_edge_label = F.sigmoid(self.mlp(x_edge)).unsqueeze(-1)
        elif self.padding == 'zeros':
            x_edge_label = torch.zeros(x_edge.size(0), 1, device=x.device)  # Zero out labels
        elif self.padding == 'ones':
            x_edge_label = torch.ones(x_edge.size(0), 1, device=x.device)
        else:
            raise ValueError('Unknown padding method:', self.padding)
        
        x_edge = torch.cat([x_edge, x_edge_label], dim=-1)  # [num_edges, hidden_dim+1]
        x = torch.cat([x_context_pos, x_context_neg, x_edge], dim=0)  # [num_context+num_edges, hidden_dim+1]

        if hasattr(self, 'transformer_col'):
            x = x.unsqueeze(-1)
            x_att, _ = self.transformer_col(x, x, x)
            x = x_att + x
            x = x.squeeze(-1)
        
        x_context = x[:(x_context_pos.size(0) + x_context_neg.size(0))]
        x_edge = x[(x_context_pos.size(0) + x_context_neg.size(0)):]

        # 3. Transformer Processing -------------------------------------------
        # Prepare inputs (add batch dimension for transformer)
        x_context = x_context.unsqueeze(1)  # [num_context, 1, hidden_dim+1]
        x_edge = x_edge.unsqueeze(1)        # [num_edges, 1, hidden_dim+1]

        # Context self-attention
        for layer in self.transformer_row:
            x_context, x_edge = layer(x_context, x_edge)

        # 4. Final Prediction -------------------------------------------------
        # Remove batch dimension and predict
        if self.output_target:
            return x_edge.squeeze(1)[-1].squeeze(-1)
        else:
            return self.head(x_edge.squeeze(1)).squeeze(-1)

# TBD
class ProdigyPredictor(nn.Module):
    def __init__(self, hidden_dim, scale):
        if scale:
            self.scale = nn.LayerNorm(hidden_dim)
    def forward(self, data, x, edge):
        if self.scale is not None:
            x = self.scale(x)
        return x[edge[0]] * x[edge[1]]
