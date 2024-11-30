from torch_geometric.utils import negative_sampling, add_self_loops
from torch_sparse import SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch
from torch import nn
from torch.nn import functional as F

def train(model, pred, optimizer, data, split_edge, args, clip_emb=False): # consider loss biased issue
    model.train()
    pred.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    neg_edge = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_train_edge.size(1))
    tot_loss = []
    loader = torch.utils.data.DataLoader(range(pos_train_edge.shape[1]), batch_size=args.batch_size, shuffle=True)
    if args.maskinput:
        adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)

    for perm in loader:
        if args.maskinput:
            adjmask[perm] = False
            edge = pos_train_edge[:, adjmask]
            edge = add_self_loops(edge, num_nodes=data.num_nodes)[0]
            adj = SparseTensor.from_edge_index(edge,sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(pos_train_edge.device)
            adj = adj.to_symmetric().coalesce()
            adjmask[perm] = True
        else:
            adj = data.adj_t

        optimizer.zero_grad()
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_out = pred(h[edge[0]], h[edge[1]])
        edge = neg_edge[:, perm]
        neg_out = pred(h[edge[0]], h[edge[1]])
        if args.loss_fn == 'bce':
            pos_loss = -F.logsigmoid(pos_out).mean()
            neg_loss = -F.logsigmoid(-neg_out).mean()
            loss = pos_loss + neg_loss
        elif args.loss_fn == 'auc':
            loss = torch.square(1 - (pos_out - neg_out)).sum()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        nn.utils.clip_grad_norm_(pred.parameters(), args.clip_grad_norm)
        if clip_emb:
            nn.utils.clip_grad_norm_(data.x, args.clip_grad_norm)

        optimizer.step()
        tot_loss.append(loss.item())
    return sum(tot_loss) / len(tot_loss)

def train_multiple(model, pred, optimizer, data, split_edge, args, clip_emb=False):
    tot_loss = 0
    for d, s in zip(data, split_edge):
        loss = train(model, pred, optimizer, d, s, args, clip_emb)
        tot_loss += loss
    return tot_loss / len(data)

@torch.no_grad()
def test(model, pred, data, split_edge, evaluator, batch_size, args):
    model.eval()
    pred.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t()
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device).t()
    pos_test_edge = split_edge['test']['edge'].to(data.x.device).t()
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device).t()
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device).t()

    h = model(data.x, data.adj_t)
    pos_train_pred = []
    loader = torch.utils.data.DataLoader(range(pos_train_edge.shape[1]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = pos_train_edge[:, perm]
        pos_train_pred.append(pred(h[edge[0]], h[edge[1]]).cpu())
    pos_train_pred = torch.cat(pos_train_pred, dim=0)

    pos_valid_pred = []
    loader = torch.utils.data.DataLoader(range(pos_valid_edge.shape[1]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = pos_valid_edge[:, perm]
        pos_valid_pred.append(pred(h[edge[0]], h[edge[1]]).cpu())
    pos_valid_pred = torch.cat(pos_valid_pred, dim=0)

    neg_valid_pred = []
    loader = torch.utils.data.DataLoader(range(neg_valid_edge.shape[1]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = neg_valid_edge[:, perm]
        neg_valid_pred.append(pred(h[edge[0]], h[edge[1]]).cpu())
    neg_valid_pred = torch.cat(neg_valid_pred, dim=0)

    pos_test_pred = []
    loader = torch.utils.data.DataLoader(range(pos_test_edge.shape[1]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = pos_test_edge[:, perm]
        pos_test_pred.append(pred(h[edge[0]], h[edge[1]]).cpu())
    pos_test_pred = torch.cat(pos_test_pred, dim=0)

    neg_test_pred = []
    loader = torch.utils.data.DataLoader(range(neg_test_edge.shape[1]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        edge = neg_test_edge[:, perm]
        neg_test_pred.append(pred(h[edge[0]], h[edge[1]]).cpu())
    neg_test_pred = torch.cat(neg_test_pred, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def test_all(model, pred, data, split_edge, evaluator, batch_size, args):
    results_list = []
    for d, s in zip(data, split_edge):
        results = test(model, pred, d, s, evaluator, batch_size, args)
        results_list.append(results)
    return results_list

@torch.no_grad()
def test_citation2(model, pred, data, split_edge, evaluator, batch_size):
    model.eval()
    pred.eval()

    train_source_edge = split_edge['eval_train']['source_node'].to(data.x.device)
    train_target_edge = split_edge['eval_train']['target_node'].to(data.x.device)
    valid_source_edge = split_edge['valid']['source_node'].to(data.x.device)
    valid_target_edge = split_edge['valid']['target_node'].to(data.x.device)
    test_source_edge = split_edge['test']['source_node'].to(data.x.device)
    test_target_edge = split_edge['test']['target_node'].to(data.x.device)
    valid_target_edge_neg = split_edge['valid']['target_node_neg'].to(data.x.device)
    test_target_edge_neg = split_edge['test']['target_node_neg'].to(data.x.device)
    train_target_edge_neg = split_edge['eval_train']['target_node_neg'].to(data.x.device)

    h = model(data.x, data.adj_t)
    pos_train_pred = []
    loader = torch.utils.data.DataLoader(range(train_source_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        pos_train_pred.append(
            pred(h, data.adj_t, torch.stack([train_source_edge[perm], train_target_edge[perm]], dim=0)).squeeze().cpu()
        )
    pos_train_pred = torch.cat(pos_train_pred, dim=0)
    train_source_edge = train_source_edge.view(-1, 1).repeat(1, 1000).view(-1)
    train_neg_edge = train_target_edge_neg.view(-1)
    neg_train_pred = []
    loader = torch.utils.data.DataLoader(range(train_neg_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        neg_train_pred.append(
            pred(h, data.adj_t, torch.stack([train_source_edge[perm], train_neg_edge[perm]], dim=0)).squeeze().cpu()
        )
    neg_train_pred = torch.cat(neg_train_pred, dim=0).view(-1, 1000)

    pos_valid_pred = []
    loader = torch.utils.data.DataLoader(range(valid_source_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        pos_valid_pred.append(
            pred(h, data.adj_t, torch.stack([valid_source_edge[perm], valid_target_edge[perm]], dim=0)).squeeze().cpu()
        )
    pos_valid_pred = torch.cat(pos_valid_pred, dim=0)
    valid_source_edge = valid_source_edge.view(-1, 1).repeat(1, 1000).view(-1)
    valid_target_edge_neg = valid_target_edge_neg.view(-1)
    neg_valid_pred = []
    loader = torch.utils.data.DataLoader(range(valid_target_edge_neg.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        neg_valid_pred.append(
            pred(h, data.adj_t, torch.stack([valid_source_edge[perm], valid_target_edge_neg[perm]], dim=0)).squeeze().cpu()
        )
    neg_valid_pred = torch.cat(neg_valid_pred, dim=0).view(-1, 1000)

    pos_test_pred = []
    loader = torch.utils.data.DataLoader(range(test_source_edge.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        pos_test_pred.append(
            pred(h, data.full_adj_t, torch.stack([test_source_edge[perm], test_target_edge[perm]], dim=0)).squeeze().cpu()
        )
    pos_test_pred = torch.cat(pos_test_pred, dim=0)
    neg_test_pred = []
    test_source_edge = test_source_edge.view(-1, 1).repeat(1, 1000).view(-1)
    test_target_edge_neg = test_target_edge_neg.view(-1)
    loader = torch.utils.data.DataLoader(range(test_target_edge_neg.shape[0]), batch_size=batch_size, shuffle=False)
    for perm in loader:
        neg_test_pred.append(
            pred(h, data.full_adj_t, torch.stack([test_source_edge[perm], test_target_edge_neg[perm]], dim=0)).squeeze().cpu()
        )
    neg_test_pred = torch.cat(neg_test_pred, dim=0).view(-1, 1000)

    results = {}
    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred
    })['mrr_list'].mean().item()
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred
    })['mrr_list'].mean().item()
    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred
    })['mrr_list'].mean().item()
    results['mrr'] = (train_mrr * 100, valid_mrr * 100, test_mrr * 100)

    return results