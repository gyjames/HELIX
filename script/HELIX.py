import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as R
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import argparse as ap
from pyfaidx import Fasta
from HELIX_models import *
import pickle
import pandas as pd
import os
import gc
from pkg_resources import resource_filename

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0], 'a':[1, 0, ,0, 0], 't':[0, 1, 0, 0], 'c':[0, 0, 1, 0], 'g':[0, 0, 0, 1]}

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

class IsoDataSet_transcript(Dataset):

    def __init__(self, fn, rbp, embeddings, genome, key_mode='simp'):
    
        self.data_dict = {}
        idx = 0
        L = 200
        length = 1000
        with open(fn, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                sample = info[0]
                chr = info[2]
                strand = info[3]
                g = info[1]
                t = info[5]
                chain = info[6].split(',')
                tp_label = info[9].split(',')
                tss = [int(chain[i]) for i in range(len(tp_label)) if tp_label[i] == 'tss']
                tes = [int(chain[i]) for i in range(len(tp_label)) if tp_label[i] == 'tes']
                ss = [int(chain[i]) for i in range(len(tp_label)) if tp_label[i] == 'a' or tp_label[i] == 'd']
                ss_label = [i for i in tp_label if i == 'a' or i == 'd']
                tp_order = get_tp_order(tp_label)
                if key_mode == 'simp':
                    embeddings_key = ['|'.join([chr, strand, str(i)]) for i in ss]
                elif key_mode == 'complete':
                    embeddings_key = ['|'.join([chr, strand, j, g, str(i),sample]) for i, j in zip(ss, ss_label)]
                if_in_embedding = [i for i in embeddings_key if i in embeddings]
                if len(if_in_embedding) != len(embeddings_key):
                    continue
                if len([i for i in tss if i < length]) > 0 or len([i for i in tes if i < length]) > 0:
                    continue
                chain_label = [int(i) for i in info[8].split(',')]
                prop = float(info[10])

                key = '|'.join([sample, t])
            
                self.data_dict[key] = {'sample':sample, 'chr':chr, 'strand':strand, 't':t, 'tss':tss, 'tes':tes, 'tp_order':tp_order, 'chain':embeddings_key, 'chain_label':chain_label, 'prop':prop}
                idx += 1

        self.event_list = list(self.data_dict.keys())
        self.rbp = rbp
        self.embeddings = embeddings
        self.genome = genome
        self.L = L
                
    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
        fa = Fasta(self.genome)
        key = self.event_list[idx]
        info = self.data_dict[key]
        sample = info['sample']
        tss = info['tss']
        tes = info['tes']
        strand = info['strand']
        chr = info['chr']
        t = info['t']
        tp_order = info['tp_order']
        chain = info['chain']
        chain_label = info['chain_label']
        prop = info['prop']

        if strand == '+':
            tss_seq = [seq_transfer(fa.get_seq(chr, i - self.L, i + self.L).seq) for i in tss]
            tes_seq = [seq_transfer(fa.get_seq(chr, i - self.L, i + self.L).seq) for i in tes]
        else:
            tss_seq = [seq_transfer(fa.get_seq(chr, i - self.L, i + self.L, rc=True).seq) for i in tss]
            tes_seq = [seq_transfer(fa.get_seq(chr, i - self.L, i + self.L, rc=True).seq) for i in tes]

        rbp_expr = torch.from_numpy(self.rbp[sample])
        label = torch.LongTensor(chain_label)
        
        embeddings = torch.stack([torch.tensor(self.embeddings[i]) for i in chain])

        return tss_seq, tes_seq, rbp_expr, embeddings, label, prop, key, tp_order
        
    def get_event(self, event_id): # 可以根据event id得到数据，测试用
        
        return self.data_dict[event_id]

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

class IsoDataSet_splicesite(Dataset):

    def __init__(self, fn, rbp, genome):
    
        self.data_dict = {}
        length = 1000

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
        self.genome = genome

    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
        fa = Fasta(self.genome)
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

def get_arg():

    parser = ap.ArgumentParser()
    parser.add_argument('-ds', '--ssinput', required=True, action='store', help='Input for splice site model.')
    parser.add_argument('-dt', '--txinput', required=True, action='store', help='Input for transcript model.')
    parser.add_argument('-rbp', '--rbpinput', required=True, action='store', help='Normalized RBP path.')
    parser.add_argument('-o', '--out', required=True, action='store', help='Output directory.')
    parser.add_argument('-g', '--genome', required=True, action='store', help='Reference genome path.')
    parser.add_argument('-d', '--device', required=True, action='store', help='Device (CPU or GPU index)')
    parser.add_argument('-c', '--core', required=False, action='store', default=1, help='Number of CPU core. Default 1.')
    parser.add_argument('-bs', '--batchsizes', required=False, action='store', default=64, help='Batch size for splice site model prediction. Default 64.')
    parser.add_argument('-bt', '--batchsizet', required=False, action='store', default=32, help='Batch size for transcript model prediction. Default 32.')
    args = parser.parse_args()

    return args

def HELIX_predict():

    args = get_arg()
    splice_input = args.ssinput
    tx_input = args.txinput
    rbp_path = args.rbpinput
    output_path = args.out
    device = args.device
    n_cpu = args.core
    batchsizes = args.batchsizes
    batchsizet = args.batchsizet
    genome = args.genome

    # ================
    # Splice site model
    # ================

    device = torch.device(device)

    L = 32
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10])
    net_mean = Baseline_module(L, W, AR)
    weight = torch.load(resource_filename('HELIX', 'models/Baseline_splicing_module.pth'))
    net_mean.load_state_dict(weight)
    net_mean.to(device)

    L = 64 # v2
    net_diff = Regulatory_module(L, W, AR)
    weight = torch.load(resource_filename('HELIX', 'models/Regulatory_splicing_module.pth'))
    net_diff.load_state_dict(weight)
    net_diff.to(device)
    torch.set_num_threads(n_cpu)

    print('HELIX splice site model loaded')

    # Load data

    rbp_dict = pickle.load(open(rbp_path, 'rb'))
    batch_size = batchsizes
    dataset = IsoDataSet_splicesite(splice_input, rbp_dict, genome)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)
    print('Splice site data loaded')

    embedding_dir = '%s/embedding' % output_path
    os.makedirs(embedding_dir, exist_ok=True)

    # Predict

    net_mean.eval()
    net_diff.eval()
    fmap_block_mean = []
    input_block_mean = []

    def forward_hook_mean(module, data_input, data_output):
        fmap_block_mean.append(data_output)
        input_block_mean.append(data_input)

    net_mean.conv_last1.register_forward_hook(forward_hook_mean)

    fmap_block_diff = []
    input_block_diff = []

    def forward_hook_diff(module, data_input, data_output):
        fmap_block_diff.append(data_output)
        input_block_diff.append(data_input)

    net_diff.conv_last.register_forward_hook(forward_hook_diff)

    fn = '%s/splice_site_output.txt' % output_path

    print('Predicting...')
    
    with torch.no_grad():

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
                    clf_prd_softmax = pred_diff[2][i].cpu().softmax(0).numpy()
                    if idx[i].split('|')[2] == 'a':
                        reg_value = np.insert(clf_prd, 0, float(pred_diff[0][i]))
                    else:
                        reg_value = np.insert(clf_prd, 0, float(pred_diff[1][i]))
                    print(
                        idx[i], 
                        labels[i], 
                        '%.3f' % float(pred_mean[0][i].sigmoid()), 
                        '%.3f' % float(pred_mean[1][i].sigmoid()), 
                        '%.3f' % float(pred_mean[2][i][0]), 
                        '%.3f' % float(pred_mean[2][i][1]), 
                        '%.3f' % float(pred_diff[0][i]), 
                        '%.3f' % float(pred_diff[1][i]), 
                        '%.3f' % clf_prd_softmax[0], 
                        '%.3f' % clf_prd_softmax[1], 
                        '%.3f' % clf_prd_softmax[2], 
                        file=f, 
                        sep='\t'
                        )
                embedding_dict[idx[i]] = np.hstack([input_block_diff[0][0][i].squeeze().detach().cpu().numpy(), reg_value])

            input_block_diff = []
            fmap_block_diff = []

        with open('%s/embedding.pickle' % (embedding_dir), 'wb') as fo:
            pickle.dump(embedding_dict, fo)

    # =========
    # Transcript model
    # =========
    
    # Load model

    net = Transcript_model()
    weight = torch.load(resource_filename('HELIX', 'models/Transcript_model.pth'))
    net.load_state_dict(weight)
    net.to(device)
    print('HELIX transcript model loaded')

    # Load data
    
    embedding_dir = '%s/embedding' % output_path
    log_fn_res = '%s/tx_output.txt' % output_path
    embeddings = pickle.load(open('%s/embedding.pickle' % (embedding_dir), 'rb'))
    dataset = IsoDataSet_transcript(tx_input, rbp_dict, embeddings, genome, key_mode='complete')
    print('Transcript data loaded')
    
    # Prediction

    print('Predicting...')
    batch_size = batchsizet

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)

    net.eval()
    batch = 0

    with torch.no_grad():

        for data in dataloader:

            batch += 1

            tss_seq = torch.concat([torch.stack(i[0]).transpose(1, 2).float().to(device) for i in data], axis=0) # [(NTSS, 4, L)]
            tss_len = [len(i[0]) for i in data]
            tss_len = [sum(tss_len[:i]) for i in range(len(tss_len)+1)]
            tes_seq = torch.concat([torch.stack(i[1]).transpose(1, 2).float().to(device) for i in data], axis=0)
            tes_len = [len(i[1]) for i in data]
            tes_len = [sum(tes_len[:i]) for i in range(len(tes_len)+1)]
            embeddings = [i[3].float().to(device) for i in data]
            rbps = [i[2].float().to(device) for i in data]# (1499,)
            labels = [torch.tensor(i[4]).to(device) for i in data]
            prop = torch.tensor([i[5] for i in data]).float().to(device)
            tp_order = [i[7] for i in data]
            idx = [i[6] for i in data]
            pred = net(tss_seq, tss_len, tes_seq, tes_len, embeddings, rbps, labels, tp_order, device).squeeze(1)
            
            with open(log_fn_res, 'a+') as f:
                for i in range(len(prop)):
                    print(idx[i], pred[i].cpu().detach().numpy(), sep='\t', file=f)

def main():
    HELIX_predict()

if __name__ == '__main__':
    main()