import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, dropout, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits






class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, dropout, subgraph_size, num_node, alpha):
        print('running with shared-lightweight decoder with a cheConv and local-aggregat recon')
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation, dropout)
        # self.conv = pygChebConv(n_in, n_h, K=2)
        self.num_node = num_node
        self.GlobalfeatEnc = GCN(n_in, n_h, activation, dropout)
        self.GlobalfeatDec = GCN(n_h, n_in, activation, dropout)
        self.alpha = alpha
        # 这里的B是一个矩阵，谨慎输入。
        self.B_network = nn.Sequential(
            nn.Linear(subgraph_size + 1, int(n_h / 2)),
            nn.PReLU(),
            nn.Linear(int(n_h / 2), n_h),
            nn.PReLU(),
        )
        self.combine_gcn = GCN(n_h, n_h, activation, dropout)



        self.hidden_size = 128
        self.act = nn.PReLU()
        # decode
        self.network = nn.Sequential(
            nn.Linear(n_h, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )

        self.subgraphRead = AvgReadout()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, seq2, B, globalGraph=None, c_adj=None):
        # adj是当前的adj，针对每一个batch的adj
        # 用正规化后的特征矩阵去成一次得到一个表示

        h_1 = self.gcn(seq1, adj, False)
        b_1 = self.B_network(B)
        # b_1 = b_1[:, idx, :]
        h_1 = self.combine_gcn(h_1 + self.alpha * b_1, adj, False)

        # 用原始的未正规的特征矩阵去乘得到一个表示
        h_row = self.gcn(seq2, adj, False)

        h_row = self.combine_gcn(h_row + self.alpha * b_1, adj, False)

        kl_loss_1 = -((0.5 / self.num_node) * torch.mean(
            torch.sum(1 + 2 * b_1 - torch.square(h_1) - torch.square(torch.exp(b_1)), dim=1)))

        kl_loss_2 = -((0.5 / self.num_node) * torch.mean(
            torch.sum(1 + 2 * b_1 - torch.square(h_row) - torch.square(torch.exp(b_1)), dim=1)))
        kl_loss = 0.5 * (kl_loss_2 + kl_loss_1)

        sub_size = h_row.shape[1]
        aa = h_row[:, :sub_size - 2, :]   # 直接readout。

        input_nei = self.subgraphRead(aa[:]) # 这里可以改成更好的readout方法
        now = self.network(input_nei)  # 线性层 重构层。 因此哲理的input_nei对应的是positive的样本


        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, : -1, :])
            h_mv = h_1[:, -1, :]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:, : -1, :], h_1[:, -2: -1, :])

        ret = self.disc(c, h_mv)  # 最后的logit 这里是discriminator
        if globalGraph is not None:
            return now, ret, self.global_reconstruct(globalGraph, c_adj), kl_loss
        return now, ret, kl_loss

    def global_reconstruct(self, globalGraph, c_adj):
        glob_h = self.GlobalfeatEnc(globalGraph, c_adj, False)
        glob_h = self.act(glob_h)
        return self.GlobalfeatDec(glob_h, c_adj, False)
