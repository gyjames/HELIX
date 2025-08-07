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


parser = ap.ArgumentParser()
parser.add_argument('-a', required=True, action='store', help='Model path')
parser.add_argument('-b', required=True, action='store', help='Model path')
parser.add_argument('-d', required=True, action='store', help='Data path')
parser.add_argument('-r', required=True, action='store', help='rbp path')
parser.add_argument('-o', required=True, action='store', help='Output path')
parser.add_argument('-c', required=True, action='store', help='Device')
parser.add_argument('--embedding', required=False, action='store_true', help='If get embedding')
args = parser.parse_args()

model_path_mean = args.a
model_path_diff = args.b
data_path = args.d
rbp_path = args.r
output_path = args.o
device = args.c
if_embedding = args.embedding
if_split = args.split

# ================
# Load HELIX
# ================

device = torch.device(device)
net_mean = torch.load(model_path_mean).to(device)
net_diff = torch.load(model_path_diff).to(device)
torch.set_num_threads(4)
print('model loaded.')

# ==============
# Load data
# ==============

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
ref_fn = '/data/workdir/zhouzh/resource/GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)
length = 1000

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

class IsoDataSet_shortreads_var_predict(Dataset):

    def __init__(self, fn, rbp, length):
    
        self.data_dict = {}

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
                label = '1,0' if tp == 'a' else '0,1'
                freq = line.strip().split('\t')[7]
                reg_label = line.strip().split('\t')[8]

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
        if info['freq'] == 'unknown':
            freq = [np.nan, np.nan]
        else:
            freq = [float(info['freq']), 0] if tp == 'a' else [0, float(info['freq'])]
        tissue = info['tissue']
        label_a = int(info['label'].split(',')[0])
        label_d = int(info['label'].split(',')[1])
        if info['reg_label'] == 'unknown':
            reg_label = np.nan
        else:
            reg_label = int(info['reg_label'])

        if strand == '+':
            seq = seq_transfer(fa.get_seq(chr, site - self.length, site + self.length).seq)
        else:
            seq = seq_transfer(fa.get_seq(chr, site - self.length, site + self.length, rc=True).seq)
        
        if len(seq) < 2 * self.length + 1:
            padding = torch.zeros((2 * self.length + 1 - len(seq), 4))
            seq = torch.concat([seq, padding])

        rbp_expr = torch.tensor(self.rbp[tissue])

        return key, seq, label_a, label_d, freq, rbp_expr, reg_label
        
    def get_event(self, event_id): 
        
        return self.data_dict[event_id]

rbp = pickle.load(open(rbp_path, 'rb'))
batch_size = 128
dataset = IsoDataSet_shortreads_var_predict(data_path, rbp, length)
dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)
print('data_loaded.')

if if_embedding:
    embedding_dir = '%s/embedding' % output_path
    os.makedirs(embedding_dir, exist_ok=True)

if if_split:
    embedding_dir_split = '%s/embedding_split' % output_path
    value_dir_split = '%s/value_split' % output_path
    os.makedirs(embedding_dir_split, exist_ok=True)
    os.makedirs(value_dir_split, exist_ok=True)

# =========
# Predict
# =========

net_mean.eval()
net_diff.eval()
fmap_block_mean = []
input_block_mean = []

def forward_hook_mean(module, data_input, data_output):
    fmap_block_mean.append(data_output)
    input_block_mean.append(data_input)

net_mean.conv_last1.register_forward_hook(forward_hook_mean)

if if_embedding:

    fmap_block_diff = []
    input_block_diff = []

    def forward_hook_diff(module, data_input, data_output):
        fmap_block_diff.append(data_output)
        input_block_diff.append(data_input)

    net_diff.conv_last.register_forward_hook(forward_hook_diff)

fn = '%s/res.txt' % output_path

print('predicting...')
with torch.no_grad():

    n_batch = 0
    embedding_dict = {}

    for data in dataloader:

        idx = [i[0] for i in data]
        X = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
        label_a = [i[2] for i in data]
        label_d = [i[3] for i in data]
        labels = ['a' if label_a[i] == 1 else 'd' for i in range(len(label_a))]
        rbp  = torch.stack([i[5] for i in data]).float().to(device)

        pred_mean = net_mean(X)
        mean_input = torch.stack([input_block_mean[0][0][i].squeeze().detach() for i in range(len(idx))])
        input_block_mean = []
        fmap_block_mean = []

        pred_diff = net_diff(X, rbp, mean_input)
        for i in range(len(idx)):
            with open(fn, 'a+') as f:
                clf_prd = pred_diff[2][i].cpu().numpy()
                print(idx[i], labels[i], float(pred_mean[0][i]), float(pred_mean[1][i]), float(pred_mean[2][i][0]), float(pred_mean[2][i][1]), float(pred_diff[0][i]), float(pred_diff[1][i]), clf_prd[0], clf_prd[1], clf_prd[2], file=f, sep='\t')
            if if_embedding:
                embedding_dict[idx[i]] = input_block_diff[0][0][i].squeeze().detach().cpu().numpy()

        input_block_diff = []
        fmap_block_diff = []

    if if_embedding:
        with open('%s/embedding.pickle' % (embedding_dir), 'wb') as fo:
            pickle.dump(embedding_dict, fo)


if if_split:

    embedding = embedding_dict
    samples = [i.split('|')[-1] for i in embedding.keys()]
    samples = list(set(samples))
    for s in samples:
        sites = [i for i in embedding.keys() if i.split('|')[-1] == s]
        embedding_sample = {i:embedding[i] for i in sites}
        with open('%s/%s.pickle' % (embedding_dir_split, s), 'wb') as f:
            pickle.dump(embedding_sample, f)
                
    fn = '%s/res.txt' % output_path
    value_dict = {}
    with open(fn, 'r') as f:
        for line in f:
            site = line.strip().split('\t')[0]
            sample = site.split('|')[-1]
            if sample not in value_dict:
                value_dict[sample] = {}
            idx = '|'.join(site.split('|')[:-1])
            tp = line.strip().split('\t')[1]
            a_value = line.strip().split('\t')[2]
            d_value = line.strip().split('\t')[3]
            clf_value = line.strip().split('\t')[4:]

            if tp == 'a':
                value_dict[sample][idx] = [float(a_value)] + clf_value
            elif tp == 'd':
                value_dict[sample][idx] = [float(d_value)] + clf_value
            elif tp == 'unknown':
                value_dict[sample][idx] = [0, 0, 0, 0]

    for i, j in value_dict.items():
        with open('%s/%s.pickle' % (value_dir_split, i), 'wb') as f:
            pickle.dump(j, f)

