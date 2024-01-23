import numpy as np
import pandas as pd
import torch
import argparse
import time
from data_utils import OpenSiteRec, split
from eval_utils import PrecisionRecall_atK, NDCG_atK, get_label
from model import VanillaMF, NeuMF, RankNet, BasicCTRModel, WideDeep, DeepFM, xDeepFM, NGCF, LightGCN
import json
from torch import nn

MODEL = {'VanillaMF': VanillaMF, 'NeuMF': NeuMF, 'RankNet': RankNet,
         'DNN': BasicCTRModel, 'WideDeep': WideDeep, 'DeepFM': DeepFM, 'xDeepFM': xDeepFM,
         'NGCF': NGCF, 'LightGCN': LightGCN}
# global emb_path


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': 0,
        'epochs': 100,
        'weight_decay': 1e-4,
        'seed': 42,
        'model': 'VanillaMF',
        'dim': 100,
        'city': 'foursquare_nyc',
        'threshold': 5,
        'topk': [15],
        'patience': 5,
        'eval_freq': 10,
        'lr_reduce_freq': 10,
        'batch_size': 128,
        'save': 0,
        'region_rate': 0.005
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val)
    args = parser.parse_args()
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

emb_path = f'../{args.city}/{args.city}_emb_drop_norm.pth'
origin_emb = torch.load(emb_path).to(args.device)

new_embeddings = split(args.city, args.threshold, args.region_rate, origin_emb)
# print(f"Mean: {new_embeddings.mean()}, Std Dev: {new_embeddings.std()},shape{new_embeddings.shape}")
dataset = OpenSiteRec(args)
# print(dataset.testDataSize)
args.user_num, args.item_num, args.cate_num = dataset.n_user, dataset.m_item, dataset.k_cate
args.Graph = dataset.Graph

# emb_path = f'../{args.city}/{args.city}_poi_rec_emb_spabert.pth'

# new_embeddings = torch.load(emb_path).to(args.device)
transform_layer = nn.Linear(new_embeddings.size(
    1), args.dim, bias=False).to(args.device)
poi_embeddings = transform_layer(new_embeddings)
# print('new emb size: ', poi_embeddings.shape)

model = MODEL[args.model](args)
# print(str(model))

# print('old emb: ', model.item_embedding.weight.shape)
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)

model.item_embedding.weight.data = poi_embeddings

# model.item_embedding.weight.data = poi_embeddings[:args.item_num]
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
# print(f'Total number of parameters: {tot_params}')


def train():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        users, labels = torch.LongTensor(dataset.U[indices]).to(args.device), \
            torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(users)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_graph():
    model.train()
    model.mode = 'train'
    dataset.uniform_sampling()
    batch_num = dataset.trainDataSize // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.trainDataSize \
            else torch.arange(i * args.batch_size, dataset.trainDataSize)
        batch = dataset.S[indices]
        users, pos_items, neg_items = torch.LongTensor(batch[:, 0]).to(args.device), \
            torch.LongTensor(batch[:, 1]).to(args.device), \
            torch.LongTensor(batch[:, 2]).to(args.device)

        loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
        loss = loss + args.weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_CTR():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        instances = {'Brand_ID': torch.LongTensor(dataset.U[indices]).to(args.device),
                     'Cate1_ID': torch.LongTensor(dataset.bF[indices][:, 0]).to(args.device),
                     #  'Cate2_ID': torch.LongTensor(dataset.bF[indices][:, 1]).to(args.device),
                     #  'Cate3_ID': torch.LongTensor(dataset.bF[indices][:, 2]).to(args.device)
                     }
        labels = torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(instances)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def test():
    global best_rec, best_ndcg
    model.eval()
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        model.mode = 'test'
    testDict = dataset.testDict
    all_pos = dataset.allPos
    rec_at_k = {k: 0. for k in [1, 5, 10, 15, 20]}
    ndcg_at_k = {k: 0. for k in [1, 5, 10, 15, 20]}
    recommendation_pairs = []
    with torch.no_grad():
        users = list(testDict.keys())
        items = [testDict[u] for u in users]
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = users[i * args.batch_size: (i + 1) * args.batch_size] \
                if (i + 1) * args.batch_size <= len(users) else users[i * args.batch_size:]
            batch_items = [items[u] for u in batch_users]

            instances = torch.LongTensor(batch_users).to(args.device)
            ratings = model(instances)

            _, ratings_K = torch.topk(ratings, k=20)
            batch_ratings = ratings_K.cpu().numpy()
            r = get_label(batch_items, batch_ratings)

            for k in rec_at_k.keys():
                _, batch_rec = PrecisionRecall_atK(batch_items, r, k)
                batch_ndcg = NDCG_atK(batch_items, r, k)
                rec_at_k[k] += batch_rec * len(batch_users)
                ndcg_at_k[k] += batch_ndcg * len(batch_users)

            recommendation_pairs.extend([{"user_id": u, "top_20_recommendations": recs.tolist()}
                                         for u, recs in zip(batch_users, batch_ratings)])

        total_users = len(users)
        for k in rec_at_k:
            rec_at_k[k] /= total_users
            ndcg_at_k[k] /= total_users

        # Save results
        rec_ndcg_results = {"Recall": rec_at_k, "nDCG": ndcg_at_k}
        # with open(f'results_{args.city}_rec_ndcg.json', 'w') as f:
        #     json.dump(rec_ndcg_results, f, indent=2)

        # with open(f'recommendations_{args.city}.json', 'w') as f:
        #     json.dump(recommendation_pairs, f, indent=2)

        # print(f"Results saved for city: {args.city}")

    # print("Test Metrics:")
    for k in sorted(rec_at_k.keys()):
        # print(f"Recall@{k}: {rec_at_k[k]}, nDCG@{k}: {ndcg_at_k[k]}")
        if best_rec[k] < rec_at_k[k]:
            best_rec[k] = rec_at_k[k]
        if best_ndcg[k] < ndcg_at_k[k]:
            best_ndcg[k] = ndcg_at_k[k]

    # print("Best Test Metrics:")
    # for k in sorted(rec_at_k.keys()):
    #     print(f"Best Recall@{k}: {best_rec[k]}, Best nDCG@{k}: {best_ndcg[k]}")

    # print("Best Test Metrics:")
    # for k in sorted(rec_at_k.keys()):
    #     if best_rec < rec_at_k[k]:
    #         best_rec = rec_at_k[k]
    #     if best_ndcg < ndcg_at_k[k]:
    #         best_ndcg = ndcg_at_k[k]
    #     print(f"Best Recall@{k}: {best_rec}, Best nDCG@{k}: {best_ndcg}")

    # # Reset best metrics for the next epoch
    # best_rec, best_ndcg = 0., 0.


t_total = time.time()
# best_rec, best_ndcg = 0., 0.
best_rec = {k: 0. for k in [1, 5, 10, 15, 20]}
best_ndcg = {k: 0. for k in [1, 5, 10, 15, 20]}
for epoch in range(args.epochs):
    best_rec = {k: 0. for k in [1, 5, 10, 15, 20]}
    best_ndcg = {k: 0. for k in [1, 5, 10, 15, 20]}
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        train_graph()
    elif args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
        train_CTR()
    else:
        train()
    if (epoch + 1) % args.eval_freq == 0:
        # print(f'Epoch {epoch}')
        test()
        torch.cuda.empty_cache()

print(f"Final Best Results for {args.model}:")
for k in sorted(best_rec.keys()):
    print(
        f"Final Best Recall@{k}: {round(best_rec[k], 4)}, Final Best nDCG@{k}: {round(best_ndcg[k], 4)}")

# def test():
#     global best_rec, best_ndcg
#     model.eval()
#     if args.model in ['RankNet', 'NGCF', 'LightGCN']:
#         model.mode = 'test'
#     testDict = dataset.testDict
#     all_pos = dataset.allPos
#     rec, ndcg = 0., 0.
#     all_pre = []
#     all_true = []
#     recommendation_pairs = []
#     with torch.no_grad():
#         users = list(testDict.keys())
#         items = [testDict[u] for u in users]
#         batch_num = len(users) // args.batch_size + 1
#         for i in range(batch_num):
#             batch_users = users[i * args.batch_size: (i + 1) * args.batch_size] \
#                 if (i + 1) * args.batch_size <= len(users) else users[i * args.batch_size:]
#             # batch_pos = [all_pos[u] for u in batch_users]
#             # batch_items = [[it for it in items[u] if it not in all_pos[u]] for u in batch_users]
#             batch_items = [items[u] for u in batch_users]
#             if args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
#                 instances = {'Brand_ID': torch.LongTensor(dataset.U[batch_users]).to(args.device),
#                              'Cate1_ID': torch.LongTensor(dataset.F[batch_users][:, 0]).to(args.device),
#                              #  'Cate2_ID': torch.LongTensor(dataset.F[batch_users][:, 1]).to(args.device),
#                              #  'Cate3_ID': torch.LongTensor(dataset.F[batch_users][:, 2]).to(args.device)
#                              }
#             else:
#                 instances = torch.LongTensor(batch_users).to(args.device)

#             ratings = model(instances)

#             _, ratings_K = torch.topk(ratings, k=int(args.topk[-1]))
#             # print("ratings_K", ratings_K)
#             all_pre.append(ratings_K)
#             all_true.append(batch_items)
#             # print("test data", batch_items)
#             ratings_K = ratings_K.cpu().numpy()

#             r = get_label(batch_items, ratings_K)

#             for k in args.topk:
#                 k = int(k)
#                 _, batch_rec = PrecisionRecall_atK(batch_items, r, k)
#                 batch_ndcg = NDCG_atK(batch_items, r, k)
#                 rec += batch_rec * len(batch_users)
#                 ndcg += batch_ndcg * len(batch_users)

#         all_pre_list = [ratings.cpu().numpy().tolist() for ratings in all_pre]
#         recommendation_pairs.append(
#             {"bactch_id": i, "test_set": list(all_true), "recommendations": all_pre_list})

#         # save_result_path = 'recommendations.json'
#         # with open(save_result_path, 'w') as f:
#         #     json.dump(recommendation_pairs, f, indent=2)

#         numpy_pre = all_pre  # .numpy()
#         numpy_true = all_true  # .numpy()
#         df = pd.DataFrame({'pre': all_pre_list, 'true': all_true})
#         df.to_csv(f'result_{args.city}_random.csv', index=False)

#         rec /= len(users)
#         ndcg /= len(users)
#         if best_rec < rec:
#             best_rec = rec
#         if best_ndcg < ndcg:
#             best_ndcg = ndcg
#         # print(f'Recall@{k}: {rec}\nnDCG@{k}: {ndcg}')


# t_total = time.time()
# best_rec, best_ndcg = 0., 0.
# for epoch in range(args.epochs):
#     if args.model in ['RankNet', 'NGCF', 'LightGCN']:
#         train_graph()
#     elif args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
#         train_CTR()
#     else:
#         train()
#     torch.cuda.empty_cache()
#     if (epoch + 1) % args.eval_freq == 0:
#         # print(f'Epoch {epoch}')
#         test()
#         torch.cuda.empty_cache()

# print(
#     f'Best Results: \nRecall@{args.topk[-1]}: {round(best_rec, 4)}\nnDCG@{args.topk[-1]}: {round(best_ndcg, 4)}')
