import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul, SparseTensor
import torch_sparse
from torch_sparse.matmul import spmm_add
from util import adjoverlap
import torch
from torch import Tensor
from typing import Iterable, Final, Optional, Dict, Tuple
import math
import torchhd

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout, inplace=True))
            self.lins.append(nn.SiLU(inplace=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        if dropout > 0:
            self.lins.append(nn.Dropout(dropout, inplace=True))
        self.lins.append(nn.SiLU(inplace=True))
        
    def forward(self, x):
        x = self.lins(x)
        return x

def my_gcn_norm(adj_t: Tensor) -> Tensor:
    deg = adj_t @ torch.ones((adj_t.shape[0]), device=adj_t.device)
    degnorm = torch.rsqrt_(deg.clamp_min_(1))
    crow_index = adj_t.crow_indices()
    col_index = adj_t.col_indices()
    value = adj_t.values()
    value = value * degnorm[col_index] *torch.repeat_interleave(degnorm, torch.diff(crow_index))
    return torch.sparse_csr_tensor(crow_index, col_index, value, size=adj_t.size())


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

    def forward(self, x: Tensor, edge: Tensor, adj_t: SparseTensor, deg: Tensor):
        D = x.shape[-1]
        degrev = torch.reciprocal(deg)

        x = torch.concat((x, x*torch.rsqrt(deg), x*torch.sqrt(deg)), dim=1)
        xs = [x, degrev*(adj_t@x)]
        for i in range(2, self.hop+1):
            xs.append(degrev*(adj_t @ xs[-1]) - xs[-2])
        x = torch.concat(xs, dim=1).unflatten(1, (-1, D)).transpose(1, 2) # N, D, C
        return x[edge]


class UniPred(nn.Module):

    def __init__(self, hop: int, num_layer: int, hiddim: int) -> None:
        super().__init__()
        hop = hop + 1
        attendim = int((3*hop*hiddim)**0.5)
        attendim = ((attendim+4-1)//4) * 4
        self.nodefeatlin = MLP(3*hop, attendim, attendim, 2, 0.0)
        self.nodefeatlin2 = MLP(attendim, hiddim, hiddim, 2, 0.0)
        self.transenc = nn.MultiheadAttention(attendim, 4, batch_first=True)
        self.hiddim = hiddim
        self.linklabellin = MLP(3*hop*3*hop, hiddim, hiddim, 2, 0.0)
        self.linpred = nn.Linear(hiddim, 1)

    def forward(self, linklabel: Tensor, nodefeat: Tensor):
        '''
        linklabel: (#E, hop*3*3*hop)
        nodefeat: (2, #E, D, hop*3)
        '''
        # print(linklabel.shape, nodefeat.shape)
        nodefeat = self.nodefeatlin(nodefeat)
        nodefeat = self.transenc.forward(nodefeat[[1, 0]].flatten(0, 1), nodefeat.flatten(0, 1), nodefeat.flatten(0, 1), need_weights=False)[0]
        nodefeat = nodefeat.unflatten(0, (2, -1))
        nodefeat = self.nodefeatlin2((nodefeat[0]*nodefeat[1]).mean(dim=-2))
        linklabel = nodefeat#self.linklabellin(linklabel) + nodefeat
        return self.linpred(nodefeat)


class SymSpmm(torch.autograd.Function):
    """
    Random Common Neighbor with feature X
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def setup_context(ctx, inputs, output):
        adj, X = inputs
        ctx.save_for_backward(adj)

    @staticmethod
    def forward(adj: Tensor, X: Tensor)->Tensor:
        '''
        X (n, d)
        adj1 (n, n)
        '''
        tX = X.detach()
        assert not tX.requires_grad
        ret = adj @ tX
        if X.requires_grad:
            ret = ret.requires_grad_(True)
        return ret

    @staticmethod
    def backward(ctx, grad_output)->Tuple[Tensor|None, None, None]:
        if not ctx.needs_input_grad[1]:
            return None, None
        adj = ctx.saved_tensors[0]
        assert not grad_output.requires_grad
        return None, adj.to(grad_output.dtype) @ grad_output


SIGMA2=0.5
SIGMA3=0.25
def nzt_distribution(shape: Iterable[int], dtype: torch.dtype, device: torch.device)->Tensor:
    '''
    p(-0.5)=2/3
    p(1)=1/3
    '''
    L = shape[-1]
    M = L//3
    prob = torch.rand(tuple(shape), dtype=dtype, device=device)
    ret = torch.zeros_like(prob) - M/(L-M)
    border = torch.topk(prob, M, dim=-1,)[0].min(dim=-1, keepdim=True)[0]
    ret[prob>=border] = 1.
    ret = ret.to(dtype)
    return ret

class Bias(nn.Module):
    def __init__(self, hiddim) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hiddim))

    def forward(self, x):
        return x + self.bias

from torch_geometric.nn import MessagePassing
# a vanilla message passing layer 
class PureConv(MessagePassing):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="sum", use_weight: bool=False) -> None:
        super().__init__(aggr, node_dim=0)
        self.aggr = aggr
        if use_weight:
            self.lin1 = nn.Linear(indim, outdim)
            self.lin2 = Bias(outdim)
        else:
            if indim == outdim:
                self.lin1 = nn.Identity()
                self.lin2 = nn.Identity()
            else:
                raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin1(x)
        if isinstance(adj_t, Tensor):
            xshape = x.shape[1:]
            x = x.flatten(1, -1)
            if self.aggr == "mean":
                x = torch.sparse.mm(adj_t, x, reduce="mean")
            elif self.aggr == "max":
                x = torch.sparse.mm(adj_t, x, reduce="amax")
            elif self.aggr == "sum":
                x = SymSpmm.apply(adj_t, x)
            elif self.aggr == "gcn":
                raise NotImplementedError
                norm = torch.rsqrt_((adj_t@torch.ones_like(x[:, 0])).clamp_min_(1)).reshape(-1, 1)
                x = norm * x
                x = norm * SymSpmm.apply(adj_t, x)
            x = x.unflatten(1, xshape)
        elif isinstance(adj_t, Tuple):
            ei, ea = adj_t
            x = self.propagate(ei, size=(x.shape[0], x.shape[0]), x=x, ea=ea)
        return self.lin2(x)
    def message(self, x_j, ea):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        return x_j * ea
    
convdict = {
    "max": lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "sum": lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "mean": lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
}






class Vmean(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, X: torch.Tensor):
        return X - X.mean(dim=-2, keepdim=True)

class Vnorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.EPS = 1e-4

    def forward(self, X):
        return X / (X.std(dim=-2, keepdim=True)+self.EPS)

# Vanilla MPNN composed of several layers.
class EquivNoiseGCN(nn.Module):
    L: Final[int]
    C: Final[int]
    normadj: Final[bool]
    num_layers: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False,
                 L=128,
                 C=3,
                 leakyratio: float=0.01):
        super().__init__()
        self.L = L
        self.C = C

        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            self.xemb.append(nn.Linear(in_channels, hidden_channels))
            self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        self.eemb_x = nn.Embedding(10, hidden_channels)
        self.eemb_Z = nn.Embedding(10, C)
        self.res = res
        self.num_layers = num_layers
        
        
        if num_layers == 1:
            hidden_channels = out_channels
        self.hidden_channels = hidden_channels
        self.normadj = "gcn" in conv_fn
        conv_fn = conv_fn.replace("gcn", "sum")
        ispure = "pure" in conv_fn
        conv_fn = conv_fn.replace("pure", "")
        convfn = convdict[conv_fn]
        self.convs = nn.ModuleList()
        self.convZs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lins2 = nn.ModuleList()
        self.linZs = nn.ModuleList()

        self.linZZk = nn.ModuleList()
        self.lincoefflins = nn.ModuleList()
        self.lincoefflinv = nn.ModuleList()

        self.snorm2 = nn.ModuleList()
        self.vnorm2 = nn.ModuleList()
        self.snorm = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.vnorm = nn.Sequential(Vmean(), Vnorm())#nn.LayerNorm(self.L, elementwise_affine=False)#nn.Sequential(Vmean(), Vnorm()) #nn.InstanceNorm1d(self.C, track_running_stats=False)#
        self.sact = nn.SiLU(inplace=True)
        self.vact = nn.ModuleList()#nn.SiLU(inplace=True)

        #self.linnewz = nn.ModuleList()

        for i in range(num_layers):
            #self.linnewz.append(nn.Conv1d(1, C, 1))
            self.vact.append(nn.Sequential(nn.SiLU(inplace=True), nn.Conv1d(C, C, 1)))
            self.lins.append(nn.Identity() if ispure else nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.SiLU(inplace=True)))
            self.lins2.append(nn.Identity() if ispure else nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.SiLU(inplace=True),nn.Linear(hidden_channels, hidden_channels)))
            self.linZs.append(nn.Sequential(nn.Conv1d(C, C, 1), nn.SiLU(inplace=True)))
                        
            self.linZZk.append(nn.Sequential(nn.Conv1d(C, C, 1), nn.SiLU(inplace=True)))

            self.convs.append(convfn(hidden_channels, hidden_channels))
            self.convZs.append(convfn(hidden_channels, hidden_channels))
            
            self.lincoefflins.append(nn.Linear(hidden_channels, C*C if i>0 else C*1))
            self.lincoefflinv.append(nn.Linear(C*C, hidden_channels))# if i > 0 else 1*1
        self.taillin = nn.Dropout(taildropout)

        coeff_coeff = (self.C+self.hidden_channels) ** -0.5
        self.coeff_coeff = min(coeff_coeff, 1/coeff_coeff)

    def forward(self, x: Tensor, adj_t: Tensor):
        if self.normadj:
            adj_t = my_gcn_norm(adj_t)
        x = self.xemb(x.squeeze())
        
        Z = nzt_distribution((x.shape[0], 1, self.L), dtype=x.dtype, device=x.device) #(N, C, L)
        # Z[:, :] = Z[:, [0]]
        
        N = x.shape[0]
        ZX_coeff = (N**-0.5)*(1/self.C/self.L)

        for i in range(0, self.num_layers):            
            x1 = self.snorm(x)
            Z1 = Z
            if i > 0:
                Z1 = self.vnorm(Z)
                rxz = Z1@((1/self.L)*Z1.transpose(1, 2))
                coeffX = self.lincoefflinv[i](rxz.flatten(1,2))
                x1 = coeffX * x1

            xn = self.lins2[i](x1)

            coeffZ = self.coeff_coeff*self.lincoefflins[i](x1).unflatten(1, (self.C, -1)) # (N, C, C)
            Zn = self.vact[i](coeffZ@Z1) 

            if True: #self.res:
                x = x + xn
                Z = Z + Zn
            else:
                x = xn
                Z = Zn
            
            x1 = self.snorm(x)
            Z1 = self.vnorm(Z)
            x1 = self.lins[i](x1)
            xn = self.convs[i](x1, adj_t)
            Z2 = self.linZs[i](Z1) #+ self.linnewz[0](nzt_distribution((N, 1, self.L), dtype=x.dtype, device=x.device))
            Zn = self.convZs[i](Z2, adj_t)
            if i>0:
                if True:
                    ZX = (ZX_coeff*Z1.permute((1, 2, 0)))@x1 # (C, L, D)
                    ZQ = self.linZZk[i](Z1) #(N, C, L)
                    xn = xn + ZQ.flatten(1, 2)@ZX.flatten(0, 1)

                    ZZ = (ZX_coeff*Z2.permute((0, 3, 2, 1)))@ Z1.permute((0, 3, 1, 2)) # (L, C, C)
                    out = ZQ.permute((2, 0, 1)) @ ZZ #(L, N, C)
                    out = out.permute((1, 2, 0)) #(N, C, L)
                    # print(out.shape, ZQ.shape, ZZ.shape, Zn.shape)
                    Zn = Zn + out
                
            if True: #self.res:
                x = x + xn
                Z = Z + Zn
            else:
                x = xn
                Z = Zn
        return self.taillin(x), self.taillin(Z)



# GAE predictor for ablation study
class LinkPredictor(nn.Module):
    cndeg: Final[int]
    L: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 C=3, 
                 L=128,
                 use_xijlin: bool=True):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.cndeg = cndeg
        zoutdim = 3*hidden_channels if use_xijlin else hidden_channels
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, zoutdim), nn.LayerNorm(zoutdim)) if use_xijlin else nn.Identity()
        self.Zlin = nn.Sequential(
            nn.Linear(C**2, hidden_channels),
            lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, zoutdim),
            nn.Dropout(dropout, inplace=True),
            nn.SiLU(inplace=True), nn.LayerNorm(zoutdim))
        self.Zlin2 = nn.Sequential(nn.Conv1d(C, C, 1))
        self.ZXlin = nn.Sequential(nn.Linear(hidden_channels, zoutdim), nn.LayerNorm(zoutdim))
        self.L = L
        self.C = C
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.lin = nn.Sequential(nn.Linear(zoutdim, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.SiLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels)) if twolayerlin else nn.Linear(zoutdim, out_channels)

    def forward(self, x, Z, adj, tar_ei,):
        N = x.shape[0]
        ZX_coeff = (N**-0.5)*(1/self.C/self.L/10)
        
        xi, xj = x[tar_ei[0]], x[tar_ei[1]]
        xij = self.xijlin(xi * xj)
        
        Zi, Zj = Z[tar_ei[0]], Z[tar_ei[1]]
        rxz = Zi@((1/self.L)*Zj.transpose(1, 2))
        xz = self.Zlin(rxz.flatten(1,2))+self.Zlin(rxz.transpose(1,2).flatten(1,2))

        ZX = (ZX_coeff*self.Zlin2(Z).permute((1, 2, 0)))@x
        xzx = self.ZXlin(torch.tensordot(Zi*Zj, ZX, dims=2))
        
        # print("pred", xij.abs().max().item(), xz.abs().max().item(), xzx.abs().max().item())
        
        rx = xij*xz*xzx
        xs = self.lin(rx)
        return xs

