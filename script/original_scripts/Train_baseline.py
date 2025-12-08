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
import argparse as ap
from scipy.stats import spearmanr, pearsonr

import pangolin_test

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
ref_fn = '/data/workdir/zhouzh/resource/GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)
length = 1000

class IsoDataSet_shortreads(Dataset):

    def __init__(self, fn):
    
        self.data_dict = {}

        with open(fn, 'r') as f:
            for line in f:
                info = line.strip().split('\t')[0]
                g = line.strip().split('\t')[3]
                chr = line.strip().split('\t')[1]
                strand = line.strip().split('\t')[2]
                site = line.strip().split('\t')[5]
                if len(site) == 0:
                    continue
                # tissue = line.strip().split('\t')[6]
                label = line.strip().split('\t')[4]
                freq = line.strip().split('\t')[6]
                # freq_list = [float(i) for i in freq.split(',')]
                # if [i for i in freq_list if i > 1]:
                #     continue

                # self.data_dict[info] = {'chr':chr, 'strand':strand, 'g':g, 'label':label, 'tissue':tissue, 'site':site, 'freq':freq}
                self.data_dict[info] = {'chr':chr, 'strand':strand, 'g':g, 'label':label, 'site':site, 'freq':freq}
        
        self.event_list = list(self.data_dict.keys())
                
    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
        key = self.event_list[idx]
        info = self.data_dict[key]
        strand = info['strand']
        chr = info['chr']
        site = int(info['site'])
        freq = float(info['freq'])
        label = info['label']
        label_a = 1 if label == 'a' else 0
        label_d = 1 if label == 'd' else 0
        freq = [freq, 0] if label == 'a' else [0, freq]
        # label_a = int(info['label'].split(',')[2])
        # label_d = int(info['label'].split(',')[3])

        if strand == '+':
            seq = seq_transfer(fa.get_seq(chr, site - length, site + length).seq)
        else:
            seq = seq_transfer(fa.get_seq(chr, site - length, site + length, rc=True).seq)
        
        if seq.shape[0] < 2001:
            seq = torch.concat([seq, torch.zeros(2001 - seq.shape[0], 4)])

        return key, seq, label_a, label_d, freq
        
    def get_event(self, event_id): # 
        
        return self.data_dict[event_id]

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

def train(dataloader, net, epoch, log_fn_train, device, loss_fn_1, loss_fn_2, optimizer):
    
    net.train()
    batch = 0
    n_batch_eval = 100

    for data in dataloader:

        batch += 1

        idx = [i[0] for i in data]
        X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
        label_a = torch.tensor([i[2] for i in data]).float().to(device)
        label_d = torch.tensor([i[3] for i in data]).float().to(device)
        freq_a = torch.tensor([i[4][0] for i in data]).float().to(device)
        freq_d = torch.tensor([i[4][1] for i in data]).float().to(device)

        pred = net(X)
        loss_1 = loss_fn_1(pred[0].squeeze(), label_a)
        loss_2 = loss_fn_1(pred[1].squeeze(), label_d)
        loss_3 = loss_fn_2(pred[2][:, 0].squeeze(), freq_a)
        loss_4 = loss_fn_2(pred[2][:, 1].squeeze(), freq_d)

        loss = loss_1 + loss_2 + loss_3 + loss_4
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        if batch % n_batch_eval == 0:

            aupr_a = roc_auc_score(label_a.cpu().detach().long().numpy(), pred[0].squeeze().cpu().detach().numpy())
            aupr_d = roc_auc_score(label_d.cpu().detach().long().numpy(), pred[1].squeeze().cpu().detach().numpy())
            spearman_a = spearmanr(freq_a.cpu().numpy(), pred[2][:, 0].squeeze().cpu().detach().numpy()).statistic
            pearson_a = pearsonr(freq_a.cpu().numpy(), pred[2][:, 0].squeeze().cpu().detach().numpy()).statistic
            spearman_d = spearmanr(freq_d.cpu().numpy(), pred[2][:, 1].squeeze().cpu().detach().numpy()).statistic
            pearson_d = pearsonr(freq_d.cpu().numpy(), pred[2][:, 1].squeeze().cpu().detach().numpy()).statistic

            with open(log_fn_train, 'a+') as f:
                print(epoch, batch, loss.item(), f'{aupr_a:.3f}', f'{aupr_d:.3f}', f'{spearman_a:.3f}', f'{pearson_a:.3f}', f'{spearman_d:.3f}', f'{pearson_d:.3f}', sep='\t', file=f)

def test(dataloader, net, batch_size, epoch, log_fn_test, device, loss_fn_1, loss_fn_2, eval=False, log_fn_res=''):
    
    net.eval()
    batch = 0

    for data in dataloader:

        batch += 1

        idx = [i[0] for i in data]
        X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
        label_a = torch.tensor([i[2] for i in data]).float().to(device)
        label_d = torch.tensor([i[3] for i in data]).float().to(device)
        freq_a = torch.tensor([i[4][0] for i in data]).float().to(device)
        freq_d = torch.tensor([i[4][1] for i in data]).float().to(device)

        with torch.no_grad():
        
            pred = net(X)
            loss_1 = loss_fn_1(pred[0].squeeze(), label_a)
            loss_2 = loss_fn_1(pred[1].squeeze(), label_d)
            loss_3 = loss_fn_2(pred[2][:, 0].squeeze(), freq_a)
            loss_4 = loss_fn_2(pred[2][:, 1].squeeze(), freq_d)

            loss = loss_1 + loss_2 + loss_3 + loss_4

            aupr_a = roc_auc_score(label_a.cpu().detach().long().numpy(), pred[0].squeeze().cpu().detach().numpy())
            aupr_d = roc_auc_score(label_d.cpu().detach().long().numpy(), pred[1].squeeze().cpu().detach().numpy())
            spearman_a = spearmanr(freq_a.cpu().numpy(), pred[2][:, 0].squeeze().cpu().detach().numpy()).statistic
            pearson_a = pearsonr(freq_a.cpu().numpy(), pred[2][:, 0].squeeze().cpu().detach().numpy()).statistic
            spearman_d = spearmanr(freq_d.cpu().numpy(), pred[2][:, 1].squeeze().cpu().detach().numpy()).statistic
            pearson_d = pearsonr(freq_d.cpu().numpy(), pred[2][:, 1].squeeze().cpu().detach().numpy()).statistic

            with open(log_fn_test, 'a+') as f:
                print(epoch, batch, loss.item(), f'{aupr_a:.3f}', f'{aupr_d:.3f}', f'{spearman_a:.3f}', f'{pearson_a:.3f}', f'{spearman_d:.3f}', f'{pearson_d:.3f}', sep='\t', file=f)

            if eval:
                with open(log_fn_res, 'a+') as f:
                    for i in range(len(pred[0])):
                        print(epoch, idx[i], F.sigmoid(pred[0][i].squeeze()).cpu().numpy(), F.sigmoid(pred[1][i].squeeze()).cpu().numpy(), pred[2][i][0].squeeze().cpu().numpy(), pred[2][i][1].squeeze().cpu().numpy(), label_a[i].cpu().detach().numpy(), label_d[i].cpu().detach().numpy(), freq_a[i].cpu().detach().numpy(), freq_d[i].cpu().detach().numpy(), sep='\t', file=f)

if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-v', required=True, action='store', help='prefix.')
    parser.add_argument('-d', required=True, action='store', help='training data directory.')
    parser.add_argument('-c', required=True, action='store', help='device')
    parser.add_argument('-tp', required=True, action='store', help='training file prefix for postive.')
    parser.add_argument('-tn', required=True, action='store', help='training file prefix for negative.')
    parser.add_argument('-p', required=True, action='store', help='proportion')
    parser.add_argument('--test', required=False, action='store_true', default=False, help='If test is required.')
    parser.add_argument('--rsm', required=False, action='store', default=-1, help='rsm epoch model.')

    args = parser.parse_args()
    print(args)

    version = args.v
    data_dir = args.d
    device = args.c
    prefix_p = args.tp
    prefix_n = args.tn
    test_required = args.test
    proportion = float(args.p)
    rsm = int(args.rsm)
    

    model_dir = '/data/workdir/zhouzh/ProjectIsoPred/TS/Train/Mean'

    model_path = '%s/%s/model' % (model_dir, version)
    metric_path = '%s/%s/metrics' % (model_dir, version)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)


    device = torch.device(device)
    torch.set_num_threads(4)

    L = 32
    # convolution window size in residual units
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21])
    # W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
    #                 21, 21, 21, 21, 41, 41, 41, 41])
    # atrous rate in residual units
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                    10, 10, 10, 10])
    
    net = pangolin_test.Pangolin_mean_v2(L, W, AR).to(device)

    loss_fn_1 = nn.BCEWithLogitsLoss()
    loss_fn_2 = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if rsm >= 0:
        rsm_state = torch.load('%s/%s/model/checkpoint_%s.pth.tar' % (model_dir, version, int(rsm)))
        rsm_epoch = rsm_state['epoch']
        net.load_state_dict(rsm_state['model_state_dict'])
        optimizer.load_state_dict(rsm_state['optimizer_state_dict'])
        scheduler.load_state_dict(rsm_state['scheduler_state_dict'])

    n_epoch = 50
    batch_size = 64

    # leave_out_chr_fn = '%s/Mean_trainingdata_nsample394_counts_thr20_sample_thr30_test.txt' % data_dir

    log_fn_train = '%s/performance_train.txt' % metric_path
    log_fn_test = '%s/performance_test.txt' % metric_path
    log_fn_test_res = '%s/performance_test_res.txt' % metric_path
    
    # 训练

    train_pos_fn = '%s/%s_train.txt' % (data_dir, prefix_p)
    dataset_pos_train = IsoDataSet_shortreads(train_pos_fn)
    train_neg_fn = '%s/%s_train.txt' % (data_dir, prefix_n)
    dataset_neg_train = IsoDataSet_shortreads(train_neg_fn)

    if test_required:
        test_pos_fn = '%s/%s_test.txt' % (data_dir, prefix_p)
        dataset_pos_test = IsoDataSet_shortreads(test_pos_fn)
        test_neg_fn = '%s/%s_test.txt' % (data_dir, prefix_n)
        dataset_neg_test = IsoDataSet_shortreads(test_neg_fn)

    for epoch in range(rsm + 1, n_epoch):

        dataset_neg_train_sub, _ = random_split(dataset_neg_train, lengths=[proportion, 1 - proportion])
        dataset_train = ConcatDataset([dataset_pos_train, dataset_neg_train_sub])

        dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)
        
        train(dataloader, net, epoch, log_fn_train, device, loss_fn_1, loss_fn_2, optimizer)
        
        scheduler.step()
        torch.save({'epoch':epoch, 
                    'model_state_dict':net.state_dict(), 
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()}, 
                    '%s/checkpoint_%s.pth.tar' % (model_path, epoch))
        torch.save(net, '%s/model_%s.pth' % (model_path, epoch))

        if test_required:
            dataset_neg_test_sub, _ = random_split(dataset_neg_test, lengths=[proportion, 1 - proportion])
            dataset_test = ConcatDataset([dataset_pos_test, dataset_neg_test_sub])

            dataloader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True, drop_last=True)
            test(dataloader, net, batch_size, epoch, log_fn_test, device, loss_fn_1, loss_fn_2, eval=True, log_fn_res=log_fn_test_res)

