from model import PureGCN, DotPredictor, CosinePredictor
from data import load_data, load_ogbl_data
import argparse
import torch
from ogb.linkproppred import Evaluator
import time

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--predictor', type=str, default='DotPredictor')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=131072)
    parser.add_argument('--num_layers', type=int, default=2)

    return parser.parse_args()

def predict(h, pos_edge, neg_edge, predictor, batch_size):
    loader = torch.utils.data.DataLoader(range(pos_edge.size(1)), batch_size=batch_size, shuffle=True)
    pos_pred = []
    neg_pred = []
    for perm in loader:
        pos = pos_edge[:, perm]
        out = predictor(h[pos[0]], h[pos[1]])
        pos_pred.append(out)

    pos_pred = torch.cat(pos_pred, dim=-1)
    loader = torch.utils.data.DataLoader(range(neg_edge.size(1)), batch_size=batch_size, shuffle=True)
    for perm in loader:
        neg = neg_edge[:, perm]
        out = predictor(h[neg[0]], h[neg[1]])
        neg_pred.append(out)
    neg_pred = torch.cat(neg_pred, dim=-1)
    return pos_pred, neg_pred

def run():
    args = parse()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if args.dataset.startswith('ogbl-'):
        data, split_edge = load_ogbl_data(args.dataset)
    else:
        data, split_edge = load_data(args.dataset)

    data.x = data.x.to(device)
    data.adj_t = data.adj_t.to(device)
    if args.dataset == 'ogbl-collab':
        data.full_adj_t = data.full_adj_t.to(device)

    if args.predictor == 'DotPredictor':
        predictor = DotPredictor()
    elif args.predictor == 'CosinePredictor':
        predictor = CosinePredictor()
    else:
        raise NotImplementedError
    
    model = PureGCN(args.num_layers)
    model = model.to(device)
    predictor = predictor.to(device)
    evaluator = Evaluator('ogbl-ppa')

    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device).t()
    pos_test_edge = split_edge['test']['edge'].to(data.x.device).t()
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device).t()
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device).t()

    st = time.time()
    h = model(data.x, data.adj_t)

    pos_valid_pred, neg_valid_pred = predict(h, pos_valid_edge, neg_valid_edge, predictor, args.batch_size)
    if args.dataset != 'ogbl-collab':
        pos_test_pred, neg_test_pred = predict(h, pos_test_edge, neg_test_edge, predictor, args.batch_size)
    else:
        h = model(data.x, data.full_adj_t)
        pos_test_pred, neg_test_pred = predict(h, pos_test_edge, neg_test_edge, predictor, args.batch_size)

    print(f"Time: {time.time()-st}", flush=True)

    valid_metric = {}
    test_metric = {}
    
    for k in [20, 50, 100]:
        evaluator.K = k
        valid_metric[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{k}']
        test_metric[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{k}']
        print(f'Hits@{k} (Valid): {valid_metric[f"hits@{k}"]:.4f}')
        print(f'Hits@{k} (Test): {test_metric[f"hits@{k}"]:.4f}')
    return valid_metric, test_metric

def main():
    avg_valid_metric = {}
    avg_test_metric = {}
    for _ in range(5):
        valid_metric, test_metric = run()
        for k, v in valid_metric.items():
            avg_valid_metric[k] = avg_valid_metric.get(k, 0) + v
        for k, v in test_metric.items():
            avg_test_metric[k] = avg_test_metric.get(k, 0) + v
    for k in avg_valid_metric.keys():
        avg_valid_metric[k] /= 5
        avg_test_metric[k] /= 5
    print('Average Valid Metric')
    print(avg_valid_metric)
    print('Average Test Metric')
    print(avg_test_metric)

if __name__ == '__main__':
    main()