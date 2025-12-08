import Network_binary_freq
import pickle
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, ConcatDataset
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
import torch.nn.functional as F
import torch.nn.utils.rnn as R
import argparse as ap
from pyfaidx import Fasta
from params import params
from scipy.stats import spearmanr, pearsonr

import pangolin_test

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
ref_fn = './GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)

class IsoDataSet_shortreads(Dataset):

    def __init__(self, fn, rbp, length, site_emb):
    
        self.data_dict = {}
        self.site_emb = site_emb

        with open(fn, 'r') as f:
            for line in f:
                info = line.strip().split('\t')[0]
                g = line.strip().split('\t')[3]
                chr = line.strip().split('\t')[1]
                strand = line.strip().split('\t')[2]
                site = line.strip().split('\t')[5]
                tp = line.strip().split('\t')[4]
                if len(site) == 0:
                    continue
                if int(site) < length:
                    continue
                tissue = line.strip().split('\t')[6]
                if tissue not in rbp:
                    continue
                # label = line.strip().split('\t')[7]
                label = '1,0' if tp == 'a' else '0,1'
                freq = line.strip().split('\t')[7]
                reg_label = line.strip().split('\t')[8]
                # freq_list = [float(i) for i in freq.split(',')]
                # if [i for i in freq_list if i > 1]:
                #     continue
                if '|'.join([chr, strand, site]) not in self.site_emb:
                    continue
                # if '|'.join([chr, strand, tp, g, site]) not in self.site_emb:
                #     continue

                self.data_dict[info] = {'chr':chr, 'strand':strand, 'g':g, 'tp':tp, 'label':label, 'tissue':tissue, 'site':site, 'freq':freq, 'reg_label':reg_label}
        
        self.event_list = list(self.data_dict.keys())
        self.rbp = rbp
        self.length = length

    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
        key = self.event_list[idx]
        info = self.data_dict[key]
        strand = info['strand']
        chr = info['chr']
        tp = info['tp']
        site = int(info['site'])
        g = info['g']
        # freq = [float(i) for i in info['freq'].split(',')][-2:]
        freq = [float(info['freq']), 0] if tp == 'a' else [0, float(info['freq'])]
        tissue = info['tissue']
        label_a = int(info['label'].split(',')[0])
        label_d = int(info['label'].split(',')[1])
        reg_label = int(info['reg_label'])
        site_emb = self.site_emb['|'.join([chr, strand, str(site)])].cpu()
        # site_emb = self.site_emb['|'.join([chr, strand, tp, g, str(site)])].cpu()

        if strand == '+':
            seq = seq_transfer(fa.get_seq(chr, site - self.length, site + self.length).seq)
        else:
            seq = seq_transfer(fa.get_seq(chr, site - self.length, site + self.length, rc=True).seq)
        
        if len(seq) < 2 * self.length + 1:
            padding = torch.zeros((2 * self.length + 1 - len(seq), 4))
            seq = torch.concat([seq, padding])

        rbp_expr = torch.tensor(self.rbp[tissue])

        return key, seq, label_a, label_d, freq, rbp_expr, reg_label, site_emb
        
    def get_event(self, event_id): # 可以根据event id得到数据，测试用
        
        return self.data_dict[event_id]

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

def train(dataloader, net, epoch, log_fn_train, device, loss_fn_1, loss_fn_2, optimizer, mid=False):
    
    net.train()
    batch = 0
    n_batch_eval = 100

    for data in dataloader:

        batch += 1

        idx = [i[0] for i in data]
        X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
        label_a = torch.tensor([i[2] for i in data]).float().to(device)
        label_d = torch.tensor([i[3] for i in data]).float().to(device)
        freq_a = torch.tensor([i[4][0] for i in data]).to(device)
        freq_d = torch.tensor([i[4][1] for i in data]).to(device)
        rbp  = torch.stack([i[5] for i in data]).float().to(device)
        reg_label  = torch.LongTensor([i[6] for i in data]).to(device)
        site_emb = torch.stack([i[7] for i in data]).float().to(device)

        pred = net(X, rbp, site_emb)
        loss_1 = loss_fn_2(pred[0].squeeze()[label_a == 1], freq_a[label_a == 1])
        loss_2 = loss_fn_2(pred[1].squeeze()[label_d == 1], freq_d[label_d == 1])
        # loss_1 = loss_fn_2(pred[0].squeeze(), freq)
        if mid:
            loss_3 = loss_fn_1(pred[2].squeeze()[reg_label != 3], reg_label[reg_label != 3])
        else:
            loss_3 = loss_fn_1(pred[2].squeeze(), reg_label)

        loss = loss_1 + loss_2 + 0.1 * loss_3
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if batch % n_batch_eval == 0:

            spearman_a = spearmanr(freq_a[label_a == 1].cpu().numpy(), pred[0].squeeze()[label_a == 1].cpu().detach().numpy()).statistic
            pearson_a = pearsonr(freq_a[label_a == 1].cpu().numpy(), pred[0].squeeze()[label_a == 1].cpu().detach().numpy()).statistic
            spearman_d = spearmanr(freq_d[label_d == 1].cpu().numpy(), pred[1].squeeze()[label_d == 1].cpu().detach().numpy()).statistic
            pearson_d = pearsonr(freq_d[label_d == 1].cpu().numpy(), pred[1].squeeze()[label_d == 1].cpu().detach().numpy()).statistic
            
            recall = recall_score(reg_label.cpu().detach().long().numpy(), pred[2].cpu().detach().numpy().argmax(1).squeeze(), average='macro')
            precision = precision_score(reg_label.cpu().detach().long().numpy(), pred[2].cpu().detach().numpy().argmax(1).squeeze(), average='macro')

            with open(log_fn_train, 'a+') as f:
                print(epoch, batch, loss.item(), f'{spearman_a:.3f}', f'{pearson_a:.3f}', f'{spearman_d:.3f}', f'{pearson_d:.3f}', f'{recall:.3f}', f'{precision:.3f}', sep='\t', file=f)

def test(dataset, rbp, net, batch_size, epoch, log_fn_test, device, loss_fn_1, loss_fn_2, eval=False, log_fn_res='', mid=False):
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True, drop_last=True)

    net.eval()
    batch = 0

    for data in dataloader:

        batch += 1

        idx = [i[0] for i in data]
        X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
        label_a = torch.tensor([i[2] for i in data]).float().to(device)
        label_d = torch.tensor([i[3] for i in data]).float().to(device)
        freq_a = torch.tensor([i[4][0] for i in data]).to(device)
        freq_d = torch.tensor([i[4][1] for i in data]).to(device)
        rbp  = torch.stack([i[5] for i in data]).float().to(device)
        reg_label  = torch.LongTensor([i[6] for i in data]).to(device)
        site_emb = torch.stack([i[7] for i in data]).float().to(device)

        with torch.no_grad():
        
            pred = net(X, rbp, site_emb)
            loss_1 = loss_fn_2(pred[0].squeeze()[label_a == 1], freq_a[label_a == 1])
            loss_2 = loss_fn_2(pred[1].squeeze()[label_d == 1], freq_d[label_d == 1])
            if mid:
                loss_3 = loss_fn_1(pred[2].squeeze()[reg_label != 3], reg_label[reg_label != 3])
            else:
                loss_3 = loss_fn_1(pred[2].squeeze(), reg_label)

            loss = loss_1 + loss_2 + 0.1 * loss_3

            spearman_a = spearmanr(freq_a[label_a == 1].cpu().numpy(), pred[0].squeeze()[label_a == 1].cpu().detach().numpy()).statistic
            pearson_a = pearsonr(freq_a[label_a == 1].cpu().numpy(), pred[0].squeeze()[label_a == 1].cpu().detach().numpy()).statistic
            spearman_d = spearmanr(freq_d[label_d == 1].cpu().numpy(), pred[1].squeeze()[label_d == 1].cpu().detach().numpy()).statistic
            pearson_d = pearsonr(freq_d[label_d == 1].cpu().numpy(), pred[1].squeeze()[label_d == 1].cpu().detach().numpy()).statistic

            recall = recall_score(reg_label.cpu().detach().long().numpy(), pred[2].cpu().detach().numpy().argmax(1).squeeze(), average='macro')
            precision = precision_score(reg_label.cpu().detach().long().numpy(), pred[2].cpu().detach().numpy().argmax(1).squeeze(), average='macro')

            with open(log_fn_test, 'a+') as f:
                print(epoch, batch, loss.item(), f'{spearman_a:.3f}', f'{pearson_a:.3f}', f'{spearman_d:.3f}', f'{pearson_d:.3f}', f'{recall:.3f}', f'{precision:.3f}', sep='\t', file=f)

            if eval:
                with open(log_fn_res, 'a+') as f:
                    for i in range(len(pred[0])):
                        pred_clf = pred[2][i].squeeze().cpu().numpy()
                        print(epoch, idx[i], pred[0][i].squeeze().cpu().numpy(), pred[1][i].squeeze().cpu().numpy(), pred_clf[0], pred_clf[1], pred_clf[2], freq_a[i].cpu().detach().numpy(), freq_d[i].cpu().detach().numpy(), reg_label[i].cpu().detach().numpy(), sep='\t', file=f)

if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-v', required=True, action='store', help='prefix.')
    parser.add_argument('-d', required=True, action='store', help='training data directory.')
    parser.add_argument('-c', required=True, action='store', help='device')
    parser.add_argument('-e', required=True, action='store', help='emb_dict path.')
    parser.add_argument('-p', required=True, action='store', help='file prefix.')
    parser.add_argument('-r', required=True, action='store', default='/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/gene_expr/rbp_tpm_sample384.pickle', help='file prefix.')
    parser.add_argument('--nr', required=False, action='store', default=1499, help='n genes.')
    parser.add_argument('--rsm', required=False, action='store', default='', help='Resume.')
    parser.add_argument('--basemodel', required=False, action='store', default='', help='Pretrained model for single cell.')
    parser.add_argument('--grd', required=False, action='store_true', default=False, help='add negtive data')
    parser.add_argument('--negpct', required=False, action='store', default=0, help='add negtive data')
    parser.add_argument('--cosine', required=False, action='store_true', default=False, help='cosine anealing')
    parser.add_argument('--test', required=False, action='store_true', default=False, help='If test')
    parser.add_argument('--log', required=False, action='store_true', default=False, help='If test')
    parser.add_argument('--mid', required=False, action='store_true', default=False, help='If test')
    parser.add_argument('-lr', required=False, action='store', default=0.0001, help='learning_rate')
    parser.add_argument('-s', required=False, action='store', default='set1', help='learning_rate')


    args = parser.parse_args()
    print(args)

    version = args.v
    data_dir = args.d
    device = args.c
    rsm = args.rsm
    basemodel = args.basemodel
    grd = args.grd
    cosine = args.cosine
    if_test = args.test
    emb_fn = args.e
    prefix = args.p
    negpct = float(args.negpct)
    rbp_fn = args.r
    NR = int(args.nr)
    mid = args.mid
    log = args.log
    lr = float(args.lr)
    s = args.s
    

    model_dir = './Var'

    model_path = '%s/%s/model' % (model_dir, version)
    metric_path = '%s/%s/metrics' % (model_dir, version)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)


    device = torch.device(device)
    torch.set_num_threads(4)


    # # v2
    L = 64 # v2
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                    10, 10, 10, 10])
    length = 1000


    net = pangolin_test.Pangolin_tissues_RBP_v2_17(L, W, AR, NR=NR, dropout_rate=0.2).to(device)
    print('Model constructed...')

    # 载入模型权重

    if basemodel:
        model_fn = basemodel
        pretrained_model = torch.load(model_fn).to(device)
        names = []
        for name, param in pretrained_model.named_parameters():
            names.append(name)

        if s == 'set1':
            train_names = [i for i in names if i.split('.')[0] in ['rbp_net', 'conv_last', 'conv_last1', 'conv_last2', 'conv_last3']]
            resblocks = [i for i in names if i.split('.')[0] == 'resblocks']
            train_names.extend([i for i in resblocks if i.split('.')[1] in ['8', '9', '10', '11']])
            train_names.extend(['convs.2.weight', 'convs.2.bias'])
            untrain_names = [i for i in names if i not in train_names]
        elif s == 'set2':        
            train_names = [i for i in names if i.split('.')[0] in ['rbp_net', 'conv_last1', 'conv_last2', 'conv_last3']]
            resblocks = [i for i in names if i.split('.')[0] == 'resblocks']
            train_names.extend([i for i in resblocks if i.split('.')[1] in ['8', '9', '10', '11']])
            train_names.remove('rbp_net.layers.0.weight')
            train_names.remove('rbp_net.layers.0.bias')
            untrain_names = [i for i in names if i not in train_names]

        pretrained_dict = {k:v for k, v in pretrained_model.state_dict().items() if k in untrain_names}
        old_state = net.state_dict()
        old_state.update(pretrained_dict)
        net.load_state_dict(old_state, strict=False)
        net = net.to(device)

        for name, param in net.named_parameters():
            param.requires_grad = False
            if name in train_names:
                param.requires_grad = True

    # loss_fn_1 = nn.BCEWithLogitsLoss()
    loss_fn_1 = nn.CrossEntropyLoss()
    loss_fn_2 = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    
    if cosine:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_epoch = 50
    batch_size = 128

    log_fn_train = '%s/performance_train.txt' % metric_path
    log_fn_test = '%s/performance_test.txt' % metric_path
    log_fn_test_res = '%s/performance_test_res.txt' % metric_path
    
    rbp = pickle.load(open(rbp_fn, 'rb'))
    
    emb_dict = pickle.load(open(emb_fn, 'rb'))
    print('Necessary data loaded...')

    if rsm:
        rsm_state = torch.load(rsm)
        rsm_epoch = rsm_state['epoch']
        net.load_state_dict(rsm_state['model_state_dict'])
        optimizer.load_state_dict(rsm_state['optimizer_state_dict'])
        scheduler.load_state_dict(rsm_state['scheduler_state_dict'])
    else:
        rsm_epoch = -1


    train_fn_neg = '%s/%s_neg_train.txt' % (data_dir, prefix)
    dataset_neg = IsoDataSet_shortreads(train_fn_neg, rbp, length=length, site_emb=emb_dict)
    print('Negative data loaded...')
    train_fn_pos = '%s/%s_pos_train.txt' % (data_dir, prefix)
    dataset_pos = IsoDataSet_shortreads(train_fn_pos, rbp, length=length, site_emb=emb_dict)
    print('Positive data loaded...')
    if mid:
        train_fn_mid = '%s/%s_mid_train.txt' % (data_dir, prefix)
        dataset_mid = IsoDataSet_shortreads(train_fn_mid, rbp, length=length, site_emb=emb_dict)
        print('Mid data loaded...')

    if if_test:
        test_fn_neg = '%s/%s_neg_test.txt' % (data_dir, prefix)
        dataset_neg_test = IsoDataSet_shortreads(test_fn_neg, rbp, length=length, site_emb=emb_dict)
        print('Negative data loaded...')
        test_fn_pos = '%s/%s_pos_test.txt' % (data_dir, prefix)
        dataset_pos_test = IsoDataSet_shortreads(test_fn_pos, rbp, length=length, site_emb=emb_dict)
        print('Positive data loaded...')
        if mid:
            test_fn_mid = '%s/%s_mid_test.txt' % (data_dir, prefix)
            dataset_mid_test = IsoDataSet_shortreads(test_fn_mid, rbp, length=length, site_emb=emb_dict)
            print('Mid data loaded...')

    for epoch in range(n_epoch):

        if epoch <= rsm_epoch:
            continue

        if grd:
            _, dataset_neg_sub = random_split(dataset_neg, lengths=[1 - 0.001 * (epoch + 1), 0.001 * (epoch + 1)])
        if log:
            prop_neg = ((np.log(len(dataset_neg) / len(dataset_pos)) + 1) * 0.3 * len(dataset_pos)) / len(dataset_neg)# 1769542
            print(prop_neg)
            prop_mid = ((np.log(len(dataset_mid) / len(dataset_pos)) + 1) * 0.3 * len(dataset_pos)) / len(dataset_mid)
            print(prop_mid)
            prop_pos = ((np.log(len(dataset_pos) / len(dataset_pos)) + 1) * 0.3 * len(dataset_pos)) / len(dataset_pos)
            print(prop_pos)
            _, dataset_neg_sub = random_split(dataset_neg, lengths=[1 - prop_neg, prop_neg])
            _, dataset_mid_sub = random_split(dataset_mid, lengths=[1 - prop_mid, prop_mid])
            _, dataset_pos_sub = random_split(dataset_pos, lengths=[1 - prop_pos, prop_pos])
        else:
            _, dataset_neg_sub = random_split(dataset_neg, lengths=[1 - negpct, negpct])

        if mid:
            dataset = ConcatDataset([dataset_pos_sub, dataset_neg_sub, dataset_mid_sub])
        else:
            dataset = ConcatDataset([dataset_pos, dataset_neg_sub])
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True, drop_last=True)

        train(dataloader, net, epoch, log_fn_train, device, loss_fn_1, loss_fn_2, optimizer, mid=mid)
        print('Epoch %s training finished...' % epoch)
        scheduler.step()
        torch.save({'epoch':epoch, 
                    'model_state_dict':net.state_dict(), 
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()}, 
                    '%s/checkpoint_%s.pth.tar' % (model_path, epoch))
        torch.save(net, '%s/model_%s.pth' % (model_path, epoch))

        if if_test:
            if log:
                _, dataset_neg_sub = random_split(dataset_neg_test, lengths=[1 - prop_neg, prop_neg])
                if mid:
                    _, dataset_mid_sub = random_split(dataset_mid_test, lengths=[1 - prop_mid, prop_mid])
                    _, dataset_pos_sub = random_split(dataset_pos_test, lengths=[1 - prop_pos, prop_pos])
            else:
                _, dataset_neg_sub = random_split(dataset_neg_test, lengths=[1 - negpct, negpct])
            
            if mid:
                dataset_test = ConcatDataset([dataset_pos_sub, dataset_neg_sub, dataset_mid_sub])
            else:
                dataset_test = ConcatDataset([dataset_pos_test, dataset_neg_sub])
            test(dataset_test, rbp, net, batch_size, epoch, log_fn_test, device, loss_fn_1, loss_fn_2, eval=True, log_fn_res=log_fn_test_res, mid=mid)
            print('Epoch %s test finished...' % epoch)
