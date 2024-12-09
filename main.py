import argparse
import torch
from data import load_data, load_ogbl_data, load_other_data, load_all_train_data_no_feat, load_all_train_data_feat
from model import MPLPNodeLabel, NodeFeat, UniPred
import numpy as np
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import time
from ogb.linkproppred import Evaluator
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor
from torch import Tensor
from routine import train, test, test_citation2, train_multiple, test_all
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as linalg
import copy
from torch_geometric.nn import Node2Vec

datasets_all_no_feat = ['Ecoli', 'Yeast', 'NS', 'PB', 'Power', 'USAir', 'Celegans']
datasets_all_feat = ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Computers', 'Photo']

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def linklda(x: Tensor, edge_index: Tensor, dim: int):
    '''
    x (n, d)
    edge_index (2, m)
    dim: return dimension
    '''
    n = x.shape[0]
    m = edge_index.shape[1]

    n_p = m
    n_n = n**2-n_p

    mu = x.mean(dim=0)
    xi, xj = x[edge_index[0]], x[edge_index[1]]
    mu_p_i, mu_p_j = xi.mean(dim=0), xj.mean(dim=0)
    mu_n_i, mu_n_j = (n**2)/n_n*mu-(n_p/n_n)*mu_p_i, (n**2)/n_n*mu-(n_p/n_n)*mu_p_j
    S_b = (2*mu-mu_p_i-mu_p_j)
    S_b = S_b.unsqueeze(-1) * S_b.unsqueeze(0)

    S_w = 4*(n**2/n_n)*(1/n)*x.T@x
    S_w += ((n_n-n_p)/n_n)*(1/m)*((xi+xj).T@(xi+xj))
    S_w -= (mu_p_i+mu_p_i).unsqueeze(-1)*(mu_p_i+mu_p_j).unsqueeze(0)
    S_w -= (mu_n_i+mu_n_i).unsqueeze(-1)*(mu_n_i+mu_n_j).unsqueeze(0)
    

    evals, evecs = torch.lobpcg(S_b, dim, S_w, largest=True)
    return x@evecs

def normalize(x: Tensor):
    x = x - x.mean(dim=0)
    x = x/x.std(dim=0)
    return x

def pca(data, hidden):
    d = cp.ones(data.edge_index.shape[1]).reshape(-1)
    row = cp.array(data.edge_index[0]).reshape(-1)
    col = cp.array(data.edge_index[1]).reshape(-1)
    A = sp.coo_matrix((d, (row, col)), shape=(data.num_nodes, data.num_nodes))
    A = A - A.mean(axis=0)
    st = time.time()
    u, sigma, _ = linalg.svds(A, k=hidden)
    feat = u @ cp.diag(sigma)
    print(time.time() - st, flush=True)
    data.x = torch.tensor(feat, dtype=torch.float32)

def pca_feat(data, hidden):
    A = cp.array(data.x)
    A = A - A.mean(axis=0)
    A = A / A.std(axis=0)

    st = time.time()
    u, sigma, _ = linalg.svds(A, k=hidden)
    print(A.shape, u.shape, flush=True)
    feat = u @ cp.diag(sigma)
    print(time.time() - st, flush=True)
    data.x = torch.tensor(feat, dtype=torch.float32)
    
def eigen(data, hidden):
    d = cp.ones(data.edge_index.shape[1]).reshape(-1)
    row = cp.array(data.edge_index[0]).reshape(-1)
    col = cp.array(data.edge_index[1]).reshape(-1)
    A = sp.coo_matrix((d, (row, col)), shape=(data.num_nodes, data.num_nodes))
    st = time.time()
    eigenvectors = linalg.eigsh(A, k=hidden)
    print(time.time() - st, flush=True)
    data.x = torch.tensor(eigenvectors[1], dtype=torch.float32)

def node2vec(edge_index, num_nodes, hidden, device):
    st = time.time()
    model = Node2Vec(edge_index, embedding_dim=hidden, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    model.train()
    for epoch in range(100):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Node2Vec Epoch: {epoch:03d}, Loss: {total_loss / len(loader):.4f}')
    return model(torch.arange(num_nodes, device=device)).detach()

def parse_args():
    parser = argparse.ArgumentParser(description='Inductlink')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_dataset', type=str, default='ogbl-collab')
    parser.add_argument('--test_dataset', type=str, default='planetoid')
    parser.add_argument('--num_layers', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--test_epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--metric', type=str, default='Hits@50')
    parser.add_argument('--pred', type=str, default='Dot')
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--test_mode', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='no-feat')
    parser.add_argument('--step_lr_decay', action='store_true', default=False)
    parser.add_argument('--maskinput', action='store_true', default=False)
    parser.add_argument('--res', action='store_true', default=False)
    parser.add_argument('--relu', action='store_true', default=False)
    parser.add_argument('--residual', type=float, default=0.0)
    parser.add_argument('--lda', action='store_true', default=False)
    parser.add_argument('--learnable_emb', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--conv', type=str, default='gcn')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--initial_emb', type=str, default='noise')
    parser.add_argument('--loss_fn', type=str, default='bce')
    parser.add_argument('--rand_noise', action='store_true', default=False)
    parser.add_argument('--agg_type', type=str, default='add_att')
    return parser.parse_args()

def init_emb(data, args, device):
    if args.initial_emb == 'eigen':
        eigen(data, args.hidden)
    elif args.initial_emb == 'pca':
        pca(data, args.hidden)
    elif args.initial_emb == 'ones':
        data.x = torch.ones(data.num_nodes, args.hidden)
    elif args.initial_emb == 'noise':
        embedding = nn.Embedding(data.num_nodes, args.hidden)
        torch.nn.init.xavier_uniform_(embedding.weight)
        data.x = embedding.weight
    elif args.initial_emb == 'node2vec':
        data.x = node2vec(data.edge_index, data.num_nodes, args.hidden, device)
    else:
        raise ValueError('Invalid initial embedding type')
    
def unify_dim(data, hidden, device, multiple=False):
    if multiple:
        for d in data:
            if d.x.shape[1] > hidden:
                pca_feat(d, hidden)
            elif d.x.shape[1] < hidden:
                n = hidden // d.x.shape[1] + 1
                d.x = torch.cat([d.x for _ in range(n)], dim=1)
                pca_feat(d, hidden)
            d = d.to(device)
    else:
        if data.x.shape[1] > hidden:
            pca_feat(data, hidden)
        elif data.x.shape[1] < hidden:
            n = hidden // data.x.shape[1] + 1
            data.x = torch.cat([data.x for _ in range(n)], dim=1)
            pca_feat(data, hidden)
        data = data.to(device)

def test_learnable_emb(model, pred, data, split_edge, evaluator, batch_size, device, args):
    embedding = nn.Embedding(data.num_nodes, args.hidden).to(device)
    torch.nn.init.xavier_uniform_(embedding.weight)
    data.x = embedding.weight
    data = data.to(device)

    optimizer = torch.optim.Adam(embedding.parameters(), lr=args.lr)

    for i in range(args.test_epochs):
        loss = train(model, pred, optimizer, data, split_edge, args, True)
        print(f'Test epoch: {i:03d}, Loss: {loss:.4f}')
        results = test(model, pred, data, split_edge, evaluator, batch_size, args)
        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(f'Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
            if key == args.metric:
                print(key, f'Test: {test_hits:.4f}')
        print('---------------------------------')

def main():
    args = parse_args()
    print(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cp.cuda.Device(args.gpu).use()

    if args.train_dataset in ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Computers', 'Photo']:
        data, split_edge = load_data(args.train_dataset)
        data = data.to(device)
    elif args.train_dataset in ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2']:
        data, split_edge = load_ogbl_data(args.train_dataset)
        if args.train_dataset == 'ogbl-ppa' or args.train_dataset == 'ogbl-ddi':
            init_emb(data, args, device)
        data = data.to(device)
    elif args.train_dataset == 'all_no_feat':
        data, split_edge = load_all_train_data_no_feat()
        for d in data:
            init_emb(d, args, device)
            d = d.to(device)
    elif args.train_dataset == 'all_feat':
        data, split_edge = load_all_train_data_feat()
        for d in data:
            d = d.to(device)
    else:
        data, split_edge = load_other_data(args.train_dataset)
        embedding = init_emb(data, args, device)
        data = data.to(device)

    if args.lda:
        data.x = normalize(data.x)
        data.x = linklda(data.x, data.edge_index, 32)
    
    if args.pca:
        pca_feat(data, 96)

    HOP = 3
    linklabeler = MPLPNodeLabel(256, hop=HOP)
    nodefeater = NodeFeat(HOP)
    model = UniPred(HOP, 1, 32).to(device)
    
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    evaluator = Evaluator(name='ogbl-ppa')

    print(f'number of parameters: {sum(p.numel() for p in params)}')

    best_results = None
    best_epoch = 0
    best_model = None
    best_pred = None

    for epoch in range(1, 1 + args.epochs):
        if args.rand_noise:
            if args.train_dataset == 'all_no_feat':
                for d in data:
                    embedding = nn.Embedding(d.num_nodes, args.hidden)
                    torch.nn.init.xavier_uniform_(embedding.weight)
                    d.x = embedding.weight
                    d = d.to(device)
            else:
                embedding = nn.Embedding(data.num_nodes, args.hidden)
                torch.nn.init.xavier_uniform_(embedding.weight)
                data.x = embedding.weight
                data = data.to(device)
        st = time.time()
        if args.train_dataset == 'all_no_feat' or args.train_dataset == 'all_feat':
            loss = train_multiple((linklabeler, nodefeater), model, optimizer, data, split_edge, args, args.learnable_emb)
        else:
            loss = train((linklabeler, nodefeater), model, optimizer, data, split_edge, args, args.learnable_emb)
        if args.step_lr_decay and epoch % 100 == 0:
            adjustlr(optimizer, epoch / args.epochs, args.lr)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        print(f'Train Time: {time.time() - st:.4f}')
        st = time.time()
        if args.train_dataset == 'all_no_feat' or args.train_dataset == 'all_feat':
            results_list = test_all((linklabeler, nodefeater), model, data, split_edge, evaluator, args.batch_size, args)
            results = results_list[0]
        else:
            results = test((linklabeler, nodefeater), model, data, split_edge, evaluator, args.batch_size, args)
        print(f'Test Time: {time.time() - st:.4f}')
        if best_results is None:
            best_results = {key: [0, 0, 0] for key in results}
        avg_results = {key: [1, 1, 1] for key in results}
        if args.train_dataset == 'all_no_feat' or args.train_dataset == 'all_feat':
            for i in range(len(results_list)):
                if args.train_dataset == 'all_no_feat':
                    print(f'Dataset: {datasets_all_no_feat[i]}')
                else:
                    print(f'Dataset: {datasets_all_feat[i]}')

                for key, result in results_list[i].items():
                    train_hits, valid_hits, test_hits = result
                    avg_results[key][0] *= train_hits
                    avg_results[key][1] *= valid_hits
                    avg_results[key][2] *= test_hits
                    print(f'Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
                    if key == args.metric:
                        print(key, f'Test: {test_hits:.4f}')
                print('---------------------------------')
            for key in avg_results:
                avg_results[key] = [avg_results[key][0]**(1/len(results_list)), avg_results[key][1]**(1/len(results_list)), avg_results[key][2]**(1/len(results_list))]
                print(f'Average {key}: Train: {avg_results[key][0]:.4f}, Valid: {avg_results[key][1]:.4f}, Test: {avg_results[key][2]:.4f}') # consider perf biased issue
                if key == args.metric:
                    print(key, f'Average Test: {avg_results[key][2]:.4f}')
                if avg_results[key][1] >= best_results[key][1]:
                    best_results[key] = list(avg_results[key])
                    if key == args.metric:
                        best_epoch = epoch
                        best_model = model.state_dict()
                        print(f'Best_epoch: {best_epoch}')
            if epoch - best_epoch > 200:
                break
        else:
            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                if valid_hits >= best_results[key][1]:
                    best_results[key] = list(result)
                    if key == args.metric:
                        best_epoch = epoch
                        best_model = copy.deepcopy(model.state_dict())
                        print(f'Best_epoch: {best_epoch}')
                print(f'Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
                if key == args.metric:
                    print(key, f'Test: {test_hits:.4f}')
            if epoch - best_epoch > 200:
                break
        print('---------------------------------')

    print(f'Final results: val {best_results[args.metric][1]:.4f}, test {best_results[args.metric][2]:.4f}')

    if args.test_mode:
        model.load_state_dict(best_model)
        if args.test_dataset == 'planetoid':
            for dataset in ['Cora', 'CiteSeer', 'PubMed']:
                args.dataset = dataset
                data, split_edge = load_data(args.dataset)
                data = data.to(device)
                if args.model == 'feat':
                    unify_dim(data, args.hidden, device, False)
                results = test(model, pred, data, split_edge, evaluator, args.batch_size, args)
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(f'{dataset} {key}: Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
                print('---------------------------------')
        elif args.test_dataset in ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2']:
            data, split_edge = load_ogbl_data(args.test_dataset)
            if args.test_dataset == 'ogbl-ppa' or args.test_dataset == 'ogbl-ddi':
                init_emb(data, args, device)
                if args.test_epochs > 0:
                    test_learnable_emb(model, pred, data, split_edge, evaluator, args.batch_size, device, args)
            if args.model == 'feat':
                unify_dim(data, args.hidden, device, False)

            data = data.to(device)
            if args.test_dataset == 'ogbl-citation2':
                results = test_citation2(model, pred, data, split_edge, evaluator, args.batch_size)
            else:
                results = test(model, pred, data, split_edge, evaluator, args.batch_size, args)
            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(f'{args.test_dataset} {key}: Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
            print('---------------------------------')
        elif args.test_dataset in ['Cora', 'CiteSeer', 'PubMed']:
            data, split_edge = load_data(args.test_dataset)
            if args.model == 'feat':
                unify_dim(data, args.hidden, device, False)

            data = data.to(device)
            results = test(model, pred, data, split_edge, evaluator, args.batch_size, args)
            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(f'{args.test_dataset} {key}: Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
            print('---------------------------------')
        else:
            data, split_edge = load_other_data(args.test_dataset)
            init_emb(data, args, device)
            if args.test_epochs > 0:
                test_learnable_emb(model, pred, data, split_edge, evaluator, args.batch_size, device, args)

            data = data.to(device)
            print(data.x.max(), data.x.min(), data.x.mean(), data.x.std(), flush=True)
            results = test(model, pred, data, split_edge, evaluator, args.batch_size, args)
            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(f'{args.test_dataset} {key}: Train: {train_hits:.4f}, Valid: {valid_hits:.4f}, Test: {test_hits:.4f}')
            print('---------------------------------')

if __name__ == '__main__':
    main()