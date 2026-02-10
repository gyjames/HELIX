import utils_real
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
comp_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
ref_fn = 'xx/GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)
length = 1000
device = torch.device('cuda:0')
batch_size = 64
torch.set_num_threads(4)
L = 200

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

def comp(seq, comp_dict=comp_dict):
    seq = ''.join([comp_dict[i] for i in seq][::-1])
    return seq

def get_tp_order(tp_label):
    n_tss = 0
    n_tes = 0
    n_ss = 0
    tp_order = []
    for i in tp_label:
        if i == 'tss':
            tp_order.append(i + '_' + str(n_tss))
            n_tss += 1
        elif i == 'tes':
            tp_order.append(i + '_' + str(n_tes))
            n_tes += 1
        else:
            tp_order.append('ss' + '_' + str(n_ss))
            n_ss += 1
    return tp_order

def get_mutated_seq(chr, strand, site, length, mut_info):

    seq_alt_plus = list(fa.get_seq(chr, site - length, site + length).seq)
    for mut in mut_info:
        v_pos = int(mut.split('|')[0])
        ref = mut.split('|')[1]
        m = mut.split('|')[2]
        alt_dist_start = v_pos - (site - 1000)
        alt_dist_end = v_pos + len(ref) - (site - 1000)

        for i in range(alt_dist_start,min(alt_dist_end, 2001)):
            if i == alt_dist_start:
                seq_alt_plus[i] = m
            else:
                seq_alt_plus[i] = ''

    seq_alt_plus_left = ''.join(seq_alt_plus[:1000])
    if len(seq_alt_plus_left) > 1000:
        seq_alt_plus_left = seq_alt_plus_left[len(seq_alt_plus_left) - 1000:]
    elif len(seq_alt_plus_left) < 1000:
        seq_alt_plus_left = 'N'*(1000 - len(seq_alt_plus_left)) + seq_alt_plus_left
    
    seq_alt_plus_right = ''.join(seq_alt_plus[1001:])
    if len(seq_alt_plus_right) > 1000:
        seq_alt_plus_right = seq_alt_plus_right[:1000]
    elif len(seq_alt_plus_right) < 1000:
        seq_alt_plus_right =  seq_alt_plus_right + 'N'*(1000 - len(seq_alt_plus_right))

    seq_alt_plus = ''.join([seq_alt_plus_left, seq_alt_plus[1000], seq_alt_plus_right])
    if len(seq_alt_plus) > 2001:
        seq_alt_plus = seq_alt_plus[:2001]

    if strand == '-':
        seq_alt_minus = comp(seq_alt_plus)
        seq = seq_transfer(seq_alt_minus)
    else:
        seq = seq_transfer(seq_alt_plus)

    return seq

class IsoDataSet_mean(Dataset):

    def __init__(self, fn):
    
        self.data_dict = {}

        with open(fn, 'r') as f:
            for line in f:
                info = line.strip().split('\t')[0]
                g = line.strip().split('\t')[3]
                chr = line.strip().split('\t')[1]
                strand = line.strip().split('\t')[2]
                site_mut = line.strip().split('\t')[5]
                if ':' in site_mut:
                    mut = list(set(site_mut.split(':')[1:]))
                    site = site_mut.split(':')[0]
                else:
                    mut = 'none'
                    site = site_mut
                if len(site) == 0:
                    continue
                # tissue = line.strip().split('\t')[6]
                label = line.strip().split('\t')[4]
                freq = line.strip().split('\t')[6]
                # freq_list = [float(i) for i in freq.split(',')]
                # if [i for i in freq_list if i > 1]:
                #     continue

                # self.data_dict[info] = {'chr':chr, 'strand':strand, 'g':g, 'label':label, 'tissue':tissue, 'site':site, 'freq':freq}
                self.data_dict[info] = {'chr':chr, 'strand':strand, 'g':g, 'label':label, 'site':site, 'freq':freq, 'mut':mut}
        
        self.event_list = list(self.data_dict.keys())
                
    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
        key = self.event_list[idx]
        info = self.data_dict[key]
        strand = info['strand']
        chr = info['chr']
        site = int(info['site'])
        mut_info = info['mut']
        freq = info['freq']
        if freq != 'unknown':
            freq = float(freq)
        label = info['label']
        label_a = 1 if label == 'a' else 0
        label_d = 1 if label == 'd' else 0
        freq = [freq, 0] if label == 'a' else [0, freq]
        # label_a = int(info['label'].split(',')[2])
        # label_d = int(info['label'].split(',')[3])

        if mut_info == 'none':
            if strand == '+':
                seq = seq_transfer(fa.get_seq(chr, site - length, site + length).seq)
            else:
                seq = seq_transfer(fa.get_seq(chr, site - length, site + length, rc=True).seq)
        else:
            seq_alt_plus = list(fa.get_seq(chr, site - length, site + length).seq)
            for mut in mut_info:
                v_pos = int(mut.split('|')[0])
                ref = mut.split('|')[1]
                m = mut.split('|')[2]
                alt_dist_start = v_pos - (site - 1000)
                alt_dist_end = v_pos + len(ref) - (site - 1000)

                # if len(ref) > len(m):
                #     diff = len(ref) - len(m)
                #     if alt_dist_start < 1000:
                #         seq_alt_plus = list(fa.get_seq(chr, site - 1000 - diff , site + 1000).seq)
                #     else:
                #         seq_alt_plus = list(fa.get_seq(chr, site - 1000 , site + 1000 + diff).seq)
                # else:
                #     seq_alt_plus = list(seq_ref_plus)

                for i in range(alt_dist_start,min(alt_dist_end, 2001)):
                    if i == alt_dist_start:
                        seq_alt_plus[i] = m
                    else:
                        seq_alt_plus[i] = ''

            seq_alt_plus_left = ''.join(seq_alt_plus[:1000])
            if len(seq_alt_plus_left) > 1000:
                seq_alt_plus_left = seq_alt_plus_left[len(seq_alt_plus_left) - 1000:]
            elif len(seq_alt_plus_left) < 1000:
                seq_alt_plus_left = 'N'*(1000 - len(seq_alt_plus_left)) + seq_alt_plus_left
            
            seq_alt_plus_right = ''.join(seq_alt_plus[1001:])
            if len(seq_alt_plus_right) > 1000:
                seq_alt_plus_right = seq_alt_plus_right[:1000]
            elif len(seq_alt_plus_right) < 1000:
                seq_alt_plus_right =  seq_alt_plus_right + 'N'*(1000 - len(seq_alt_plus_right))

            seq_alt_plus = ''.join([seq_alt_plus_left, seq_alt_plus[1000], seq_alt_plus_right])
            if len(seq_alt_plus) > 2001:
                seq_alt_plus = seq_alt_plus[:2001]

            # if len(seq_alt_plus) > 2001:
            #     if alt_dist_start > 1000:
            #         seq_alt_plus = seq_alt_plus[:2001]
            #     else:
            #         seq_alt_plus = seq_alt_plus[-2001:]

            if strand == '-':
                seq_alt_minus = comp(seq_alt_plus)
                seq = seq_transfer(seq_alt_minus)
            else:
                seq = seq_transfer(seq_alt_plus)
        
        if seq.shape[0] < 2001:
            seq = torch.concat([seq, torch.zeros(2001 - seq.shape[0], 4)])

        return key, seq, label_a, label_d, freq
        
    def get_event(self, event_id): 
        
        return self.data_dict[event_id]

input_dir = 'xx/input_mean_intron'
files = os.listdir(input_dir)
output_dir = 'xx/output_mean'


model_pth = 'xx/model.pth'
net = torch.load(model_pth).to(device)

net.eval()
fmap_block = []
input_block = []

def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)

net.conv_last1.register_forward_hook(forward_hook)

for file in files:
    # if file == 'input_mean.txt':
    #     continue
    mean_fn = '%s/%s' % (input_dir, file)
    dataset_mean = IsoDataSet_mean(mean_fn)
    dataloader = DataLoader(dataset_mean, shuffle=False, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)

    result_fo = '%s/value/%s' % (output_dir, file.split('_')[-1])
    embedding_fo = '%s/embedding/%s.pickle' % (output_dir, file.split('_')[-1].strip('.txt'))
    if os.path.exists(result_fo):
        continue

    emb_dict = {}
    with torch.no_grad():

        n_batch = 0

        for data in dataloader:

            idx = [i[0] for i in data]
            print(len(idx))
            X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)

            pred = net(X)
            for i, j in enumerate(idx):
                emb_dict[j] = input_block[0][0][i].squeeze().detach().cpu()

            input_block = []
            fmap_block = []
            n_batch += 1

            with open(result_fo, 'a+') as fo:
                for i in range(len(pred[0])):
                    print(idx[i], F.sigmoid(pred[0][i].squeeze()).cpu().numpy(), F.sigmoid(pred[1][i].squeeze()).cpu().numpy(), pred[2][i][0].squeeze().cpu().numpy(), pred[2][i][1].squeeze().cpu().numpy(), sep='\t', file=fo)

    with open(embedding_fo, 'wb') as f:
        pickle.dump(emb_dict, f)

