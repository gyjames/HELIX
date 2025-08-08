import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out

class FC(nn.Module):

    def __init__(self, input_shape, hidden_units, dropout_rate, batch_normalization, activation='relu', output_activation=True):

        super().__init__()
        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        model_list = []

        for i in range(len(hidden_units)):
            model_list.append(nn.Linear(input_shape, hidden_units[i]))
            if self.batch_normalization:
                model_list.append(nn.BatchNorm1d(hidden_units[i]))
            if i < len(hidden_units) - 1 or output_activation == True:
                if activation == 'relu':
                    model_list.append(nn.ReLU())
                elif activation == 'tanh':
                    model_list.append(nn.Tanh())
            model_list.append(nn.Dropout(dropout_rate))
            input_shape = hidden_units[i]

        self.layers = nn.ModuleList(model_list)
        self.bn = nn.BatchNorm1d(hidden_units[-1])

    def forward(self, x):
    
        for layer in self.layers:
            x = layer(x)

        return x

class Pangolin_mean_v2(nn.Module):
    def __init__(self, L, W, AR):
        super(Pangolin_mean_v2, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = FC(L, [L, 1], 0, False, activation='relu', output_activation=False)
        self.conv_last2 = FC(L, [L, 1], 0, False, activation='relu', output_activation=False)
        self.conv_last3 = FC(L, [L, 2], 0, False, activation='relu', output_activation=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        CL = 2 * np.sum(AR * (W - 1))
        skip = F.pad(skip, (-CL // 2, -CL // 2)).squeeze(-1)
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)
        out3 = self.sigmoid(self.conv_last3(skip))
        return out1, out2, out3

class Pangolin_tissues_RBP_v2_17(nn.Module):


    def __init__(self, L, W, AR, NR=1499, dropout_rate=0):

        super(Pangolin_tissues_RBP_v2_17, self).__init__()
        self.n_chans = L
        # self.n_tissue = n_tissue
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        
        self.rbp_net = FC(NR, [512, L], dropout_rate, True, activation='relu', output_activation=True)
        self.conv_last = FC(L + L + 32, [L], 0, False, activation='relu', output_activation=False)
        self.conv_last1 = FC(L, [L, 1], 0, False, activation='relu', output_activation=False)
        self.conv_last2 = FC(L, [L, 1], 0, False, activation='relu', output_activation=False)
        self.conv_last3 = FC(L, [L, 3], 0, False, activation='relu', output_activation=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rbp, site_emb):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        CL = 2 * np.sum(AR * (W - 1))
        skip = F.pad(skip, (-CL // 2, -CL // 2))
        rbp = self.rbp_net(rbp).unsqueeze(-1)
        skip = torch.concat([skip, rbp, site_emb.unsqueeze(-1)], dim=1).squeeze(-1)
        mid = self.conv_last(skip)
        out1 = self.conv_last1(mid)
        out2 = self.conv_last2(mid)
        out3 = self.conv_last3(mid)
        
        return out1, out2, out3

class Pangolin(nn.Module):
    def __init__(self, L, W, AR):
        super(Pangolin, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, L, 1)
        self.L = L
        self.W = W
        self.AR = AR

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        CL = 2 * np.sum(self.AR * (self.W - 1))
        skip = F.pad(skip, (-CL // 2, -CL // 2))
        print(skip.shape)
        out1 = self.conv_last1(skip).squeeze(-1)
        return out1

class iso_v3(nn.Module):

    def __init__(self):
        super(iso_v3, self).__init__()
        self.dim = 164
        self.L = self.dim # v2
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        self.RBPfc = FC(1499, [512, self.dim], dropout_rate=0, batch_normalization=False, activation='relu', output_activation=True)
        self.conv1 = Pangolin(self.L, self.W, self.AR)
        self.conv2 = Pangolin(self.L, self.W, self.AR)
        self.fc1 = FC(2*self.dim, [self.dim], dropout_rate=0, batch_normalization=False, activation='relu', output_activation=True)
        self.fc2 = FC(2*self.dim, [self.dim], dropout_rate=0, batch_normalization=False, activation='relu', output_activation=True)
        self.embedding = nn.Embedding(2, self.dim)
        self.lstm_iso_input_size = self.dim
        self.lstm_iso_cell_size = self.dim
        self.lstm_iso_n_layer = 2
        self.lstm_bidirectional = True
        self.dropout_rate = 0
        self.lstm_iso = nn.LSTM(self.lstm_iso_input_size, self.lstm_iso_cell_size, self.lstm_iso_n_layer, bias=True, bidirectional=self.lstm_bidirectional, dropout=self.dropout_rate, batch_first=True)
        self.iso_fc = FC(2*self.dim, [64, 1], dropout_rate=0, batch_normalization=False, activation='relu', output_activation=False)

    def forward(self, seq_tss, len_tss, seq_tes, len_tes, embeddings, rbps, labels, tp_labels, device):

        rbps = self.RBPfc(torch.stack(rbps)) # (B, 1499) -> (B, L)

        seq_tss = self.conv1(seq_tss)
        torch.cuda.empty_cache()
        seq_tss = torch.concat([torch.concat([rbps[i].tile(len_tss[i+1] - len_tss[i], 1) for i in range(len(len_tss)-1)], axis=0), seq_tss], axis=1)
        seq_tss = self.fc1(seq_tss)
        torch.cuda.empty_cache()
        seq_tes = self.conv2(seq_tes)
        torch.cuda.empty_cache()
        seq_tes = torch.concat([torch.concat([rbps[i].tile(len_tes[i+1] - len_tes[i], 1) for i in range(len(len_tes)-1)], axis=0), seq_tes], axis=1)
        seq_tes = self.fc2(seq_tes)
        seq_tss = [seq_tss[len_tss[i]:len_tss[i+1]] for i in range(len(len_tss)-1)]
        seq_tes = [seq_tes[len_tes[i]:len_tes[i+1]] for i in range(len(len_tes)-1)]
        torch.cuda.empty_cache()

        element_dict = {'tss':seq_tss, 'tes':seq_tes, 'ss':embeddings}
        embeddings = [torch.cat([element_dict[tp_labels[i][j].split('_')[0]][i][int(tp_labels[i][j].split('_')[1])].unsqueeze(0) for j in range(len(tp_labels[i]))], axis=0) for i in range(len(tp_labels))]
        seq_len = [len(i) for i in embeddings]
        embeddings = R.pad_sequence(embeddings, batch_first=True) # (batch_size, max_len, n_channel, seq_len)
        labels = [self.embedding(i) for i in labels]
        labels = R.pad_sequence(labels, batch_first=True) # (batch_size, max_len, n_channel, seq_len)
        embeddings = embeddings * labels
        embeddings = R.pack_padded_sequence(embeddings, lengths=seq_len, batch_first=True, enforce_sorted=False)

        _, (c, _) = self.lstm_iso(embeddings)
        c = torch.concat([c[2, :, :], c[3, :, :]], dim=-1)
        c = self.iso_fc(c)
        c = F.sigmoid(c)

        return c
