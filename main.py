import numpy as np
from numpy.core.fromnumeric import shape
import scipy.sparse as sp
import torch
import torch.nn as nn
from aug import *
from model import *
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import random
import os
from pygod.utils import load_data
from torch_geometric.utils import to_dense_adj
import argparse
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_edgelist(file):
    network = nx.read_weighted_edgelist(file)
    A = np.asarray(nx.adjacency_matrix(network, nodelist=None, weight='None').todense())
    x = A
    return x

m_dic = {'cora': 5429,'Amazon':3695, 'Flickr':239738, 'disney':335, 'citeseer': 4732, 'pubmed': 44338, 'BlogCatalog': 171743, 'ACM': 71980, 'dblp': 8817, 'citation': 15098, 'citation_20':15098, 'dblp_20':8817,'weibo':407963,'reddit':168016, 'books':3695}

def get_ground_truthDataset(dataset,cache=None):
    data = load_data(dataset,cache_dir=cache)
    adj = to_dense_adj(data.edge_index)[0]
    adj = sp.csr_matrix(adj)
    feat = data.x
    feat = sp.lil_matrix(feat)
    label = np.array(data.y)
    return adj, feat, label

parser = argparse.ArgumentParser(description='''CARD:Community-Guided Contrastive Learning with
                                                Anomaly-Aware Reconstruction for Attributed Networks
                                                                                Anomaly Detection''')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  
parser.add_argument('--auc_test_rounds', type=int, default=150)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--earlystop', type=bool, default=True)
parser.add_argument('--gama', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--modelMode', type=str, default='gpu')
parser.add_argument('--IsSwap', type=bool, default=False)
parser.add_argument('--m', type=int)
parser.add_argument('--device', type=str, default=None, help='e.g. cuda:0, cuda:1 or cpu')
args = parser.parse_args()


if args.lr is None:
    args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'dblp', 'citation','reddit','books']:
        args.num_epoch = 100
    elif args.dataset in ['ACM','Flickr']:
        args.num_epoch = 400

print("reading edgelist")
normal_adj = load_edgelist('./edgelist/' + args.dataset + '.edgelist')


args.m = m_dic[args.dataset]

k1 = np.sum(normal_adj, axis=1)
k2 = k1.reshape(normal_adj.shape[0], 1)
k1k2 = k1 * k2
eij = k1k2 / (2 * args.m)
B = np.array(normal_adj - eij)
if args.dataset in ['ACM','pubmed']:
    args.batch_size = 500
batch_size = args.batch_size
subgraph_size = args.subgraph_size
print('Dataset: ', args.dataset)

print(args.gama, args.beta)


device = torch.device(args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

# device = torch.device('cpu')
# Set random seed
# DGL removed: RWR now uses Python random, NumPy and PyTorch seeds below.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
if args.dataset in ['reddit','books']:
    adj, features, ano_label = get_ground_truthDataset(args.dataset)
else:
    adj, features, labels, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

diff = np.load('./diff/diff_A_' + args.dataset + '.npy', allow_pickle=True)

b_adj = sp.csr_matrix(diff)
b_adj = (b_adj + sp.eye(b_adj.shape[0])).todense()
pyg_graph = adj_to_pyg_graph(None, adj)
raw_feature = features.todense()
features, _ = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]

c_features = features

c_features = torch.FloatTensor(c_features)
c_adj = adj.todense()
c_adj = torch.FloatTensor(c_adj).to(device)
c_features = rand_prop(features=c_features, dropnode_rate=0.5, A=c_adj, order=5, device=device)

c_features_pyg = adj_to_pyg_graph(c_features, c_adj)
print('unleash the memory')
c_features = c_features.cpu()
torch.cuda.empty_cache()  # 释放显存

adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
c_adj = adj
features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])

adj = torch.FloatTensor(adj[np.newaxis])
b_adj = torch.FloatTensor(b_adj[np.newaxis])
B = torch.FloatTensor(B[np.newaxis])
alpha = 0.1
if args.dataset in ['cora','citeseer']:
    alpha = 0.3    
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout,
                                                args.dropout, args.subgraph_size,adj.shape[1],alpha=alpha)
print('the running gama is %f, fpbal is %f' % (args.gama, args.beta))
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if torch.cuda.is_available():
    print('Using CUDA')
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_adj = b_adj.to(device)
    c_features = c_features.to(device)
    B = B.to(device)
    if args.IsSwap:
        c_features_pyg = c_features_pyg.to(device)
    else:
        c_features_pyg = c_features_pyg.to(device)

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
best_auc = 0
batch_num = nb_nodes // batch_size + 1

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:, -1, :] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.to(device)
    added_adj_zero_col = added_adj_zero_col.to(device)
    added_feat_zero_row = added_feat_zero_row.to(device)
mse_loss = nn.MSELoss(reduction='mean')
# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):

        loss_full_batch = torch.zeros((nb_nodes, 1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.to(device)

        model.train()

        all_idx = list(range(nb_nodes))

        random.shuffle(all_idx)
        total_loss = 0.
        subgraphs = generate_rwr_subgraph(pyg_graph, subgraph_size)
        p = 0
        i = 0
        Flag = False
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num-1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                Flag = True
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            # 拼成一个 cur_batch_size( 1 + negsamp_ratio) * 1 的tensor 对应1111100000
            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            ba = []
            bf = []
            br = []
            raw = []
            BA = []
            # cf = []

            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj_r = b_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj_B = B[:, subgraphs[i], :][:, :, subgraphs[i]]
                BA.append(cur_adj_B)
                cur_feat = features[:, subgraphs[i], :]
                raw_f = raw_feature[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)
                raw.append(raw_f)
                br.append(cur_adj_r)

            ba = torch.cat(ba)
            br = torch.cat(br)

            BA = torch.cat(BA)
            BA = torch.cat((BA, added_adj_zero_row), dim=1)
            BA = torch.cat((BA, added_adj_zero_col), dim=2)

            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            br = torch.cat((br, added_adj_zero_row), dim=1)
            br = torch.cat((br, added_adj_zero_col), dim=2)


            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)

            now1, logits,kl_1 = model(bf, ba, raw, BA)
            if Flag == True:
                now2, logits2, c_now, kl_2 = model(bf, br, raw, BA, c_features.unsqueeze(0), adj)
            else:
                now2, logits2, kl_2 = model(bf, br, raw, BA)


            kl = 0.5 * (kl_2 + kl_1)
            i = i + 1

            # 重构误差
            batch = now1.shape[0]
            loss_re = 0.5 * (mse_loss(now1, raw[:, -1, :]) + mse_loss(now2, raw[:, -1, :]))
            if Flag == True:
                loss_global_re = torch.mean(
                    torch.sqrt(torch.sum(torch.pow(c_now[:, :] - raw_feature[0, :, :], 2), 1))) * (
                                            args.batch_size / adj.shape[1])

            loss_all2 = b_xent(logits2, lbl)
            loss_all1 = b_xent(logits, lbl)
            loss_bce = (loss_all1 + loss_all2) / 2

            h_1 = F.normalize(logits[:batch, :], dim=1, p=2)
            h_2 = F.normalize(logits2[:batch, :], dim=1, p=2)
            coloss2 = 2 - 2 * (h_1 * h_2).sum(dim=-1).mean()
            if Flag == True:
                loss = (1 - args.beta) * (torch.mean(loss_bce) + coloss2 + args.gama * loss_re) \
                    + args.beta * loss_global_re + 0.5 * kl
                tmp_loss = torch.mean(loss_bce) + coloss2 + args.gama * loss_re
            else:
                loss = (1 - args.beta) * (torch.mean(loss_bce) + coloss2 + args.gama * loss_re) + 0.5 *  kl
            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()

            total_loss += loss
            p = p + 1
        if args.earlystop:
                with torch.no_grad():
                    now1, logits,_ = model(bf, ba, raw, BA)
                    now2, logits2, c_now,_ = model(bf, br, raw, BA, c_features.unsqueeze(0), adj)
                    # c_now = c_now.to(device)
                    # now2, logits2 = model(bf, br, raw)
                    logits = torch.squeeze(logits)
                    logits = torch.sigmoid(logits)

                    logits2 = torch.squeeze(logits2)
                    logits2 = torch.sigmoid(logits2)
                scaler1 = MinMaxScaler()
                scaler2 = MinMaxScaler()
                scaler3 = MinMaxScaler()
                    # 相当于就是说，前半部分是一个negative，后边部分是positive
                ano_score1 = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                ano_score2 = - (logits2[:cur_batch_size] - logits2[cur_batch_size:]).cpu().numpy()
                    # ano_score3 = - (logits3[:cur_batch_size] - logits3[cur_batch_size:]).cpu().numpy()
                    # 属性的重构误差
                pdist = nn.PairwiseDistance(p=2)
                score1 = (pdist(now1, raw[:, -1, :]) + pdist(now2, raw[:, -1, :])) / 2
                score_global_re = pdist(c_now.to(device), raw_feature[0, :, :])
                score_global_re = score_global_re.cpu().numpy()
                score_global_re = scaler3.fit_transform(score_global_re.reshape(-1, 1)).reshape(-1)
                score2 = (ano_score1 + ano_score2) / 2
                score1 = score1.cpu().numpy()
                ano_score_co = scaler1.fit_transform(score2.reshape(-1, 1)).reshape(-1)
                score_re = scaler2.fit_transform(score1.reshape(-1, 1)).reshape(-1)
                ano_scores = ano_score_co + args.gama * score_re
                final_test_score = score_global_re
                test_auc = roc_auc_score(ano_label, final_test_score)
                if test_auc > best_auc:
                    best_auc = test_auc
                else:
                    pbar.update(1)
                    cnt_wait += 1
                    if cnt_wait > 200:
                        break
                    else:
                        continue
            
        mean_loss = total_loss

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best.pkl')  # multi_round_ano_score_p[round, idx] = ano_score_p

        else:
            cnt_wait += 1

        pbar.update(1)

# # # # # Test model

print('testing_' + args.dataset)
print('Loading {}th epoch from the training'.format(best_t))

model.load_state_dict(torch.load('best.pkl'))

multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_global = np.zeros((args.auc_test_rounds, nb_nodes))
kk = 0

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('EVALUTION CARD')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        subgraphs = generate_rwr_subgraph(pyg_graph, subgraph_size)
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ba = []
            bf = []
            br = []
            raw = []
            BA = []
            #  cf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj2 = b_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                raw_f = raw_feature[:, subgraphs[i], :]
                # cur_c_feat = c_features[:, subgraphs[i], :]
                # cf.append(cur_c_feat)

                cur_adj_B = B[:, subgraphs[i], :][:, :, subgraphs[i]]
                BA.append(cur_adj_B)

                ba.append(cur_adj)
                br.append(cur_adj2)
                bf.append(cur_feat)
                raw.append(raw_f)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            br = torch.cat(br)
            br = torch.cat((br, added_adj_zero_row), dim=1)
            br = torch.cat((br, added_adj_zero_col), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            BA = torch.cat(BA)
            BA = torch.cat((BA, added_adj_zero_row), dim=1)
            BA = torch.cat((BA, added_adj_zero_col), dim=2)


            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)

            with torch.no_grad():
                now1, logits,_ = model(bf, ba, raw, BA)
                now2, logits2, c_now,_ = model(bf, br, raw, BA, c_features.unsqueeze(0), adj)
                # c_now = c_now.to(device)
                # now2, logits2 = model(bf, br, raw)
                logits = torch.squeeze(logits)
                logits = torch.sigmoid(logits)

                logits2 = torch.squeeze(logits2)
                logits2 = torch.sigmoid(logits2)



            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()
            ano_score1 = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            ano_score2 = - (logits2[:cur_batch_size] - logits2[cur_batch_size:]).cpu().numpy()
            pdist = nn.PairwiseDistance(p=2)
            score1 = (pdist(now1, raw[:, -1, :]) + pdist(now2, raw[:, -1, :])) / 2
            score_global_re = pdist(c_now.to(device), raw_feature[0, :, :])
            score_global_re = score_global_re.cpu().numpy()
            score_global_re = scaler3.fit_transform(score_global_re.reshape(-1, 1)).reshape(-1)
            multi_round_ano_score_global[round, :] = score_global_re
            score2 = (ano_score1 + ano_score2) / 2
            score1 = score1.cpu().numpy()

            ano_score_co = scaler1.fit_transform(score2.reshape(-1, 1)).reshape(-1)
            score_re = scaler2.fit_transform(score1.reshape(-1, 1)).reshape(-1)

            ano_scores = ano_score_co + args.gama * score_re
            multi_round_ano_score[round, idx] = ano_scores

        pbar_test.update(1)

scaler1 = MinMaxScaler()
resultList  = []
ano_score_final = (1 - args.beta) * np.mean(multi_round_ano_score, axis=0) + args.beta * np.mean(multi_round_ano_score_global, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)
print('the auc is ', auc)
