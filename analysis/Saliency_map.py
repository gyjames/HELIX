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
from scipy.stats import spearmanr, pearsonr, zscore
import pangolin_test
from collections import Counter

# saliency map

# ======================
# model1
# ======================

# model

model_pth = 'xx'
device = torch.device('cuda:0')

net_mean = torch.load(model_pth).to(device)

# data

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
ref_fn = 'xx/GRCh38.primary_assembly.genome.fa'
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
        
    def get_event(self, event_id): 
        
        return self.data_dict[event_id]

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

batch_size = 64
data_fn = 'xx/Mean_trainingdata_nsample390_counts_thr50_sample_thr30_notissue_specific.txt'

dataset_train = IsoDataSet_shortreads(data_fn)
dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)

# calculate saliency map

net_mean.eval()

batch = 0
saliency_dict = {}

for data in dataloader:

    print(batch)
    batch += 1
    idx = [i[0] for i in data]
    X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
    label_a = torch.tensor([i[2] for i in data]).float().to(device)
    label_d = torch.tensor([i[3] for i in data]).float().to(device)

    X.requires_grad_()
    pred = net_mean.forward(X)
    pred_a_v = pred[2][:, 0]
    pred_d_v = pred[2][:, 1]

    pred_a_v.backward(torch.ones_like(pred_a_v), retain_graph=True)
    saliency_a_v = X.grad.data

    pred_d_v.backward(torch.ones_like(pred_d_v), retain_graph=True)
    saliency_d_v = X.grad.data

    for i in range(len(idx)):
        if idx[i] not in saliency_dict:
            saliency_dict[idx[i]] = {}
            if label_a[i] == 1:
                saliency_dict[idx[i]]['v'] = saliency_a_v[i].cpu().numpy()
                saliency_dict[idx[i]]['pred'] = pred_a_v[i].cpu().detach().numpy()
            elif label_a[i] == 0:
                saliency_dict[idx[i]]['v'] = saliency_d_v[i].cpu().numpy()
                saliency_dict[idx[i]]['pred'] = pred_d_v[i].cpu().detach().numpy()

with open('xx/saliency_map_mean.pickle', 'wb') as f:
    pickle.dump(saliency_dict, f)

# ==============================
# model2
# ==============================

# model

device = torch.device('cuda:0')
model_path_diff = 'xx/model_45.pth'
net_diff = torch.load(model_path_diff).to(device)
rbp_fn = 'xx/rbp_tpm_sample384.pickle'
emb_fn = 'xx/emb_dict_final_full_region_390_12.pickle'

# data

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
ref_fn = 'xx/GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)
length = 1000

rbp = pickle.load(open(rbp_fn, 'rb'))
emb_dict = pickle.load(open(emb_fn, 'rb'))

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

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
        
    def get_event(self, event_id): 
        
        return self.data_dict[event_id]

batch_size = 64
data_fn = 'xx/model2_input.txt'

dataset_train = IsoDataSet_shortreads(data_fn, rbp, length=length, site_emb=emb_dict)
dataloader = DataLoader(dataset_train, shuffle=False, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)

# calculate saliency map

net_diff.eval()
n_batch_eval = 100
saliency_dict = {}

batch = 0

for data in dataloader:

    print(batch)
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

    X.requires_grad_()
    rbp.requires_grad_()
    pred = net_diff.forward(X, rbp, site_emb)

    pred_a_p = pred[0]
    pred_d_p = pred[1]
    # pred_a_v = pred[2][label_a == 1, 0]
    # pred_d_v = pred[2][label_d == 1, 1]

    pred_a_p.backward(torch.ones_like(pred_a_p), retain_graph=True)
    saliency_a_p_seq = X.grad.data
    saliency_a_p_rbp = rbp.grad.data

    pred_d_p.backward(torch.ones_like(pred_d_p), retain_graph=True)
    saliency_d_p_seq = X.grad.data
    saliency_d_p_rbp = rbp.grad.data

    for i in range(len(idx)):
        if idx[i] not in saliency_dict:
            saliency_dict[idx[i]] = {}
            if label_a[i] == 1:
                saliency_dict[idx[i]]['p_seq'] = saliency_a_p_seq[i].cpu().numpy()
                saliency_dict[idx[i]]['p_rbp'] = saliency_a_p_rbp[i].cpu().numpy()
                saliency_dict[idx[i]]['pred_v'] = pred_a_p[i].cpu().detach().numpy()
                saliency_dict[idx[i]]['reg_label'] = reg_label[i].cpu().numpy()
            elif label_a[i] == 0:
                saliency_dict[idx[i]]['p_seq'] = saliency_d_p_seq[i].cpu().numpy()
                saliency_dict[idx[i]]['p_rbp'] = saliency_d_p_rbp[i].cpu().numpy()
                saliency_dict[idx[i]]['pred_v'] = pred_d_p[i].cpu().detach().numpy()
                saliency_dict[idx[i]]['reg_label'] = reg_label[i].cpu().numpy()

with open('xx/saliency_map_diff_rbp.pickle', 'wb') as f:
    pickle.dump(saliency_dict, f)
