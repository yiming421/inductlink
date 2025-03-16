from model import PureGCN_v1, Hadamard_MLPPredictor, DotPredictor, NoisePredictor, PureGCN
from data import load_data, load_other_data, load_ogbl_data
import argparse
import torch
from ogb.linkproppred import Evaluator
from torch_geometric.utils import negative_sampling
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_sparse import SparseTensor
import wandb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--metric', type=str, default='hits@100')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--test_batch_size', type=int, default=131072)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--hidden", default=256, type=int)
    parser.add_argument("--dp", default=0.2, type=float)
    parser.add_argument("--num_neg", default=3, type=int)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--maskinput', type=bool, default=False)
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--model', type=str, default='PureGCN_v1')
    parser.add_argument('--predictor', type=str, default='Hadamard')
    parser.add_argument('--sweep', type=bool, default=False)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_res', type=bool, default=False)
    parser.add_argument('--emb', type=bool, default=False)
    parser.add_argument('--reduce', type=str, default='lin')

    return parser.parse_args()

def train(model, data, train_pos_edge, optimizer, pred, batch_size, maskinput, num_neg):
    st = time.time()

    model.train()
    pred.train()

    train_pos_edge = train_pos_edge.t()

    dataloader = DataLoader(range(train_pos_edge.size(0)), batch_size, shuffle=True)
    total_loss = 0
    if maskinput:
        adjmask = torch.ones(train_pos_edge.size(0), dtype=torch.bool, device=data.x.device)
    st_sample = time.time()
    train_neg_edge = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=train_pos_edge.size(0) * num_neg).t()
    print(f"Sample time: {time.time()-st_sample}", flush=True)
    train_neg_edge = train_neg_edge.view(-1, num_neg, 2)

    for perm in dataloader:
        if maskinput:
            adjmask[perm] = False
            edge = train_pos_edge[adjmask].t()
            # edge = add_self_loops(edge, num_nodes=data.num_nodes)[0]
            adj = SparseTensor.from_edge_index(edge,sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(train_pos_edge.device)
            adj = adj.to_symmetric().coalesce()
            adjmask[perm] = True
        else:
            adj = data.adj_t

        # model forward
        # print(f"Memory1: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True)
        h = model(data.x, adj)
        # print(f"Memory2: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True)

        pos_edge = train_pos_edge[perm]

        neg_edge = train_neg_edge[perm]
        neg_edge = torch.reshape(neg_edge, (-1, 2))
        neg_edge = neg_edge.to(data.x.device)

        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        # print(f"Memory3: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True)
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])
        # print(f"Memory4: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True)

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    en = time.time()
    print(f"Train time: {en-st}", flush=True)

    return total_loss / len(dataloader)

@torch.no_grad()
def test(model, predictor, data, pos_test_edge, neg_test_edge, pos_valid_edge, neg_valid_edge, pos_train_edge, evaluator, batch_size):
    st = time.time()
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    # predict
    # break into mini-batches for large edge sets
    pos_valid_loader = DataLoader(range(pos_valid_edge.size(1)), batch_size, shuffle=False)
    neg_valid_loader = DataLoader(range(neg_valid_edge.size(1)), batch_size, shuffle=False)


    valid_pos_score = []
    for idx in pos_valid_loader:
        e = pos_valid_edge[:, idx]
        out = predictor(h[e[0]], h[e[1]])
        valid_pos_score.append(out)
    valid_pos_score = torch.cat(valid_pos_score, dim=0)

    valid_neg_score = []
    for idx in neg_valid_loader:
        e = neg_valid_edge[:, idx]
        out = predictor(h[e[0]], h[e[1]])
        valid_neg_score.append(out)
    valid_neg_score = torch.cat(valid_neg_score, dim=0)

    # calculate valid metric
    valid_results = {}
    for k in [20, 50, 100]:
        evaluator.K = k
        valid_results[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': valid_pos_score,
            'y_pred_neg': valid_neg_score,
        })[f'hits@{k}']

    train_pos_score = []
    for idx in pos_valid_loader:
        e = pos_train_edge[:, idx]
        out = predictor(h[e[0]], h[e[1]])
        train_pos_score.append(out)
    train_pos_score = torch.cat(train_pos_score, dim=0)

    train_results = {}
    for k in [20, 50, 100]:
        evaluator.K = k
        train_results[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': train_pos_score,
            'y_pred_neg': valid_neg_score,
        })[f'hits@{k}']

    test_results = {}
    pos_test_loader = DataLoader(range(pos_test_edge.size(1)), batch_size, shuffle=False)
    neg_test_loader = DataLoader(range(neg_test_edge.size(1)), batch_size, shuffle=False)

    if 'full_adj_t' in data:
        h = model(data.x, data.full_adj_t)

    test_pos_score = []
    for idx in pos_test_loader:
        e = pos_test_edge[:, idx]
        out = predictor(h[e[0]], h[e[1]])
        test_pos_score.append(out)
    test_pos_score = torch.cat(test_pos_score, dim=0)

    test_neg_score = []
    for idx in neg_test_loader:
        e = neg_test_edge[:, idx]
        out = predictor(h[e[0]], h[e[1]])
        test_neg_score.append(out)
    test_neg_score = torch.cat(test_neg_score, dim=0)

    for k in [20, 50, 100]:
        evaluator.K = k
        test_results[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': test_pos_score,
            'y_pred_neg': test_neg_score,
        })[f'hits@{k}']

    print(f"Test time: {time.time()-st}", flush=True)
    return train_results, valid_results, test_results


def run(args):
    if args.sweep:
        wandb.init(project='inductlink')
        config = wandb.config
        for key in config.keys():
            setattr(args, key, config[key])
    else:
        wandb.init(project='inductlink', config=args)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.dataset.startswith('ogbl-'):
        data, split_edge = load_ogbl_data(args.dataset)
    elif args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Computers', 'Photo']:
        data, split_edge = load_data(args.dataset)
    else:
        data, split_edge = load_other_data(args.dataset)

    if args.emb:
        emb = nn.Embedding(data.num_nodes, args.hidden)
        nn.init.orthogonal_(emb.weight)
        data.x = emb.weight.detach()

    data.x = data.x.to(device)
    data.adj_t = data.adj_t.to(device)
    if 'full_adj_t' in data:
        data.full_adj_t = data.full_adj_t.to(device)

    input_dim = data.x.size(1)

    if args.model == 'PureGCN_v1':
        model = PureGCN_v1(input_dim, args.num_layers, args.hidden, args.dp, args.norm, args.reduce)
    elif args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    else:
        raise NotImplementedError
    
    '''if input_dim <= args.hidden:
        num_repeats = args.hidden // input_dim + 1
        xs = [data.x]
        h = data.x
        for _ in range(num_repeats):
            h = model(h, data.adj_t).detach()
            xs.append(h)
        data.x = torch.cat(xs, dim=1).to(device)

    U, S, V = torch.pca_lowrank(data.x, q=args.hidden)
    data.x = torch.mm(U, torch.diag(S))'''
    
    if args.predictor == 'Hadamard':
        predictor = Hadamard_MLPPredictor(args.hidden, args.dp, args.mlp_layers, args.mlp_res, args.norm, args.scale)
    elif args.predictor == 'Dot':
        predictor = DotPredictor()
    elif args.predictor == 'Noise':
        predictor = NoisePredictor(args.hidden, args.dp, args.mlp_layers, args.mlp_res, args.norm, args.scale)
    else:
        raise NotImplementedError
    model = model.to(device)
    predictor = predictor.to(device)
    evaluator = Evaluator('ogbl-ppa')

    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t().to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device).t().to(device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device).t().to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device).t().to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device).t().to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

    st_all = time.time()
    best_valid = 0
    final_test = 0

    for epoch in range(args.epochs):
        st = time.time()
        loss = train(model, data, pos_train_edge, optimizer, predictor, args.batch_size, args.maskinput, args.num_neg)
        train_metric, valid_metric, test_metric = test(model, predictor, data, pos_test_edge, neg_test_edge, pos_valid_edge, neg_valid_edge, pos_train_edge, evaluator, args.test_batch_size)
        wandb.log({'train_loss': loss, 'train_metric': train_metric[args.metric], 'valid_metric': valid_metric[args.metric], 'test_metric': test_metric[args.metric]})
        for k, v in train_metric.items():
            print(f"Epoch {epoch} {k} (Train): {v:.4f}")
        for k, v in valid_metric.items():
            print(f"Epoch {epoch} {k} (Valid): {v:.4f}")
        for k, v in test_metric.items():
            print(f"Epoch {epoch} {k} (Test): {v:.4f}")
        print(f"Epoch {epoch} Loss: {loss}", flush=True)
        en = time.time()
        print(f"Epoch time: {en-st}", flush=True)
        if valid_metric[args.metric] > best_valid:
            best_valid = valid_metric[args.metric]
            best_epoch = epoch
            final_test = test_metric[args.metric]

        if epoch - best_epoch >= 200:
            break

    print(f"Total time: {time.time()-st_all}", flush=True)
    wandb.log({'final_test': final_test})
    return final_test

def main():
    args = parse()
    avg_test = 0
    for _ in range(args.runs):
        final_test = run(args)
        avg_test += final_test
    avg_test /= args.runs
    print(f"Average test: {avg_test}")
    wandb.init(project='inductlink')
    wandb.log({'avg_test': avg_test})

if __name__ == '__main__':
    main()
