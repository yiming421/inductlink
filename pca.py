from model import PureGCN_v1, Hadamard_MLPPredictor, PureGCN, DotPredictor, MLP, DeepSetPredictor, SetTransformerPredictor
from data import load_all_train_data_feat, load_data, load_ogbl_data
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
import copy

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='Cora')
    parser.add_argument('--test_dataset', type=str, default='Cora')
    parser.add_argument('--metric', type=str, default='hits@100')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--test_batch_size', type=int, default=131072)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--hidden", default=256, type=int)
    parser.add_argument("--dp", default=0.2, type=float)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--maskinput', type=bool, default=False)
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--model', type=str, default='PureGCN_v1')
    parser.add_argument('--predictor', type=str, default='Hadamard')
    parser.add_argument('--sweep', type=bool, default=False)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_res', type=bool, default=False)
    parser.add_argument('--transformer_hidden', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--pma', type=bool, default=False)
    parser.add_argument('--transformer_layers', type=int, default=1)
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
        h = model(data.x, adj)

        pos_edge = train_pos_edge[perm]

        neg_edge = train_neg_edge[perm]
        neg_edge = torch.reshape(neg_edge, (-1, 2))
        neg_edge = neg_edge.to(data.x.device)

        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])

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

def train_all(model, data_list, split_edge_list, optimizer, pred, batch_size, maskinput, num_neg, test_dataset):
    tot_loss = 0
    for data, split_edge in zip(data_list, split_edge_list):
        train_pos_edge = split_edge['train']['edge'].t()
        loss = train(model, data, train_pos_edge, optimizer, pred, batch_size, maskinput, num_neg)
        print(f"Dataset {data.name} Loss: {loss}", flush=True)
        tot_loss += loss
    return tot_loss / (len(data_list))

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

def test_all(model, predictor, data_list, split_edge_list, test_dataset, evaluator, batch_size, metric='hits@100'):
    tot_train_metric, tot_valid_metric, tot_test_metric = 1, 1, 1
    for data, split_edge in zip(data_list, split_edge_list):
        pos_test_edge = split_edge['test']['edge'].t()
        neg_test_edge = split_edge['test']['edge_neg'].t()
        pos_valid_edge = split_edge['valid']['edge'].t()
        neg_valid_edge = split_edge['valid']['edge_neg'].t()
        pos_train_edge = split_edge['train']['edge'].t()
        train_metric, valid_metric, test_metric = test(model, predictor, data, pos_test_edge, neg_test_edge, pos_valid_edge, neg_valid_edge, pos_train_edge, evaluator, batch_size)
        print(f"Dataset {data.name}")
        for k, v in train_metric.items():
            print(f"Train {k}: {v:.4f}")
        for k, v in valid_metric.items():
            print(f"Valid {k}: {v:.4f}")
        for k, v in test_metric.items():
            print(f"Test {k}: {v:.4f}")
        tot_train_metric *= train_metric[metric]
        tot_valid_metric *= valid_metric[metric]
        tot_test_metric *= test_metric[metric]
    return tot_train_metric ** (1/(len(data_list))), tot_valid_metric ** (1/(len(data_list))), tot_test_metric ** (1/(len(data_list)))

def run(args):
    if args.sweep:
        wandb.init(project='inductlink')
        config = wandb.config
        for key in config.keys():
            setattr(args, key, config[key])
    else:
        wandb.init(project='inductlink', config=args)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    train_dataset = args.train_dataset.split(',')
    data_list, split_edge_list = load_all_train_data_feat(train_dataset)

    for data in data_list:
        data.x = data.x.to(device)
        if data.x.size(1) <= args.hidden:
            num_repeats = args.hidden // data.x.size(1) + 1
            xs = [data.x]
            h = data.x
            for _ in range(num_repeats):
                h = model(h, data.adj_t).detach()
                xs.append(h)
            data.x = torch.cat(xs, dim=1).to(device)

        st = time.time()
        U, S, V = torch.pca_lowrank(data.x, q=args.hidden)
        data.x = torch.mm(U, torch.diag(S)).to(device)
        print(f"PCA time: {time.time()-st}", flush=True)
        data.adj_t = data.adj_t.to(device)
        if 'full_adj_t' in data:
            data.full_adj_t = data.full_adj_t.to(device)

    for split_edge in split_edge_list:
        for key in split_edge.keys():
            for key2 in split_edge[key].keys():
                split_edge[key][key2] = split_edge[key][key2].to(device)
        
    if args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    elif args.model == 'PureGCN_v1':
        model = PureGCN_v1(args.hidden, args.num_layers, args.hidden, args.dp, args.norm, 'id')
    else:
        raise NotImplementedError

    if args.predictor == 'Hadamard':
        predictor = Hadamard_MLPPredictor(args.hidden, args.dp, args.mlp_layers, args.mlp_res, args.norm, args.scale) 
    elif args.predictor == 'Transformer':
        predictor = SetTransformerPredictor(args.transformer_hidden, args.hidden, args.dp, 
                                            args.mlp_layers, args.num_heads, args.transformer_layers,
                                            args.norm, args.pma)
    else:
        raise NotImplementedError
    model = model.to(device)
    predictor = predictor.to(device)
    evaluator = Evaluator('ogbl-ppa')

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in predictor.parameters())}', flush=True)

    st_all = time.time()
    best_valid = 0
    final_test = 0
    best_pred = None
    best_model = None
    best_epoch = 0

    for epoch in range(args.epochs):
        st = time.time()
        print(f"Epoch {epoch}", flush=True)
        loss = train_all(model, data_list, split_edge_list, optimizer, predictor, args.batch_size, args.maskinput, args.num_neg, args.test_dataset)
        train_metric, valid_metric, test_metric = test_all(model, predictor, data_list, split_edge_list, args.test_dataset, evaluator, args.test_batch_size)
        wandb.log({'train_loss': loss, 'train_metric': train_metric, 'valid_metric': valid_metric, 'test_metric': test_metric})
        en = time.time()
        print(f"Epoch time: {en-st}", flush=True)
        if valid_metric >= best_valid:
            best_valid = valid_metric
            best_epoch = epoch
            final_test = test_metric
            best_pred = copy.deepcopy(predictor.state_dict())
            best_model = copy.deepcopy(model.state_dict())

        if epoch - best_epoch >= 200:
            break

    print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9} GB", flush=True)
    print(f"Total time: {time.time()-st_all}", flush=True)
    wandb.log({'final_test': final_test})

    model.load_state_dict(best_model)
    predictor.load_state_dict(best_pred)
    if args.test_dataset.startswith('ogbl-'):
        data, split_edge = load_ogbl_data(args.test_dataset)
    else:
        data, split_edge = load_data(args.test_dataset)
        if data.x.size(1) <= args.hidden:
            num_repeats = args.hidden // data.x.size(1) + 1
            xs = [data.x]
            h = data.x
            for _ in range(num_repeats):
                h = model(h, data.adj_t).detach()
                xs.append(h)
            data.x = torch.cat(xs, dim=1).to(device)
    st = time.time()
    data.x = data.x.to(device)
    if data.x.size(1) <= args.hidden:
        num_repeats = args.hidden // data.x.size(1) + 1
        xs = [data.x]
        h = data.x
        for _ in range(num_repeats):
            h = model(h, data.adj_t).detach()
            xs.append(h)
        data.x = torch.cat(xs, dim=1).to(device)
    U, S, V = torch.pca_lowrank(data.x, q=args.hidden)
    data.x = torch.mm(U, torch.diag(S)).to(device)
    print(f"PCA time: {time.time()-st}", flush=True)
    data.adj_t = data.adj_t.to(device)
    if 'full_adj_t' in data:
        data.full_adj_t = data.full_adj_t.to(device)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t().to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device).t().to(device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device).t().to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device).t().to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device).t().to(device)

    st_total = time.time()
    st = time.time()
    train_metric, valid_metric, test_metric = test(model, predictor, data, pos_test_edge, neg_test_edge, pos_valid_edge, neg_valid_edge, pos_train_edge, evaluator, args.test_batch_size)
    print(f"Test time: {time.time()-st}", flush=True)
    print(f'induct_train_metric: {train_metric[args.metric]}, induct_valid_metric: {valid_metric[args.metric]}, induct_test_metric: {test_metric[args.metric]}', flush=True)
    wandb.log({'induct_train_metric': train_metric[args.metric], 'induct_valid_metric': valid_metric[args.metric], 'induct_test_metric': test_metric[args.metric]})
    print(f"Total time: {time.time()-st_total}", flush=True)

    #temporarily add for test
    '''for i in range(100):
        U, S, V = torch.pca_lowrank(data.x, q=args.hidden)
        data.x = torch.mm(U, torch.diag(S)).to(device)
        train_metric, valid_metric, test_metric = test(model, predictor, data, pos_test_edge, neg_test_edge, pos_valid_edge, neg_valid_edge, pos_train_edge, evaluator, args.test_batch_size)
        wandb.log({'pca_test': test_metric[args.metric]})
        print(f"PCA test {i}: {test_metric[args.metric]}", flush=True)'''

    return train_metric, valid_metric, test_metric

def main():
    args = parse()
    avg_train_metric = {}
    avg_valid_metric = {}
    avg_test_metric = {}
    for _ in range(args.runs):
        train_metric, valid_metric, test_metric = run(args)
        for k, v in train_metric.items():
            avg_train_metric[k] = avg_train_metric.get(k, 0) + v
        for k, v in valid_metric.items():
            avg_valid_metric[k] = avg_valid_metric.get(k, 0) + v
        for k, v in test_metric.items():
            avg_test_metric[k] = avg_test_metric.get(k, 0) + v
    for k in avg_valid_metric.keys():
        avg_train_metric[k] /= args.runs
        avg_valid_metric[k] /= args.runs
        avg_test_metric[k] /= args.runs
    print('Average Train Metric')
    print(avg_train_metric)
    print('Average Valid Metric')
    print(avg_valid_metric)
    print('Average Test Metric')
    print(avg_test_metric)
    wandb.init(project='inductlink')
    wandb.log({'avg_train_metric': avg_train_metric[args.metric], 'avg_valid_metric': avg_valid_metric[args.metric], 'avg_test_metric': avg_test_metric[args.metric]})

if __name__ == '__main__':
    main()
