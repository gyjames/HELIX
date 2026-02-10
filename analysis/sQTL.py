import pyarrow.parquet as pq
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, ConcatDataset
import pandas as pd
from pyfaidx import Fasta
import numpy as np
import pickle
import torch.nn.functional as F
import os
import argparse as ap

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
comp_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
ref_fn = 'xx/GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)

def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

def comp(seq, comp_dict=comp_dict):
    seq = ''.join([comp_dict[i] for i in seq][::-1])
    return seq

class sQTL_dataset(Dataset):

    def __init__(self, sqtl_fn, expr_dict, strand_dict, dst=1000, length=1000):

        p_file = pq.ParquetFile(sqtl_fn)
        data = p_file.read().to_pandas()
        data['up'] = [int(i.split(':')[1]) for i in data['phenotype_id']]
        data['down'] = [int(i.split(':')[2]) for i in data['phenotype_id']]
        data['SNP'] = [int(i.split('_')[1]) for i in data['variant_id']]

        data_sub_up = data.loc[(abs(data['up'] - data['SNP']) < dst)]
        data_sub_up['tp'] = 'UP'
        data_sub_down = data.loc[(abs(data['down'] - data['SNP']) < dst)]
        data_sub_down['tp'] = 'DOWN'

        data_sub = pd.concat([data_sub_up, data_sub_down])

        self.data_dict = {}
        self.expr = expr_dict
        tissue = sqtl_fn.split('.')[0].split('associations_')[1]

        for i in range(data_sub.shape[0]):

            info = data_sub.iloc[i, :]
            ph_id = info['phenotype_id']
            ss_1 = int(ph_id.split(':')[1])
            ss_2 = int(ph_id.split(':')[2])
            gene = ph_id.split(':')[4].split('.')[0]
            chr = ph_id.split(':')[0]
            v_id = info['variant_id']
            v_pos = int(v_id.split('_')[1])
            ref = v_id.split('_')[2]
            alt = v_id.split('_')[3]
            af = info['af']
            pval_nominal = info['pval_nominal']
            slope = info['slope']
            tp = info['tp']

            if int(ss_1) < length or (ss_2) < length:
                continue
            
            if gene not in strand_dict:
                continue

            key_idx = '|'.join([ph_id, v_id, tp])

            self.data_dict[key_idx] = {'chr':chr, 
                                    'ph_id':ph_id, 
                                    'ss_1':ss_1, 
                                    'ss_2':ss_2, 
                                    'v_id':v_id, 
                                    'v_pos':v_pos, 
                                    'ref':ref, 
                                    'alt':alt, 
                                    'af':af, 
                                    'pval_nominal':pval_nominal, 
                                    'slope':slope, 
                                    'tp':tp, 
                                    'tissue':tissue, 
                                    'gene':gene}
        
        self.event_list = list(self.data_dict.keys())
        self.length = length
        self.strand_dict = strand_dict

    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
        key = self.event_list[idx]
        info = self.data_dict[key]
        chr = info['chr']
        ss_1 = info['ss_1']
        ss_2 = info['ss_2']
        v_pos = info['v_pos']
        ref = info['ref']
        alt = info['alt']
        tp = info['tp']
        tissue = info['tissue']
        gene = info['gene']
        pval_nominal = info['pval_nominal']
        slope = info['slope']
        
        if tp == 'UP': 
            ss = ss_1 + 1
        else:
            ss = ss_2 - 1

        alt_dist_start = v_pos - (ss - 1000)
        alt_dist_end = v_pos + len(ref) - (ss - 1000)
        seq_ref_plus = fa.get_seq(chr, ss - 1000, ss + 1000).seq

        if len(ref) > 1:
            if alt_dist_start < 1000:
                seq_alt_plus = list(fa.get_seq(chr, ss - 1000 - (len(ref) - 1) , ss + 1000).seq)
            else:
                seq_alt_plus = list(fa.get_seq(chr, ss - 1000 , ss + 1000 + (len(ref) - 1)).seq)
            seq_alt_plus[alt_dist_start + 1:alt_dist_end] = ''
        else:
            seq_alt_plus = list(seq_ref_plus)
            seq_alt_plus[alt_dist_start] = alt

        seq_alt_plus = ''.join(seq_alt_plus)
        if len(seq_alt_plus) > 2001:
            if alt_dist_start > 1000:
                seq_alt_plus = seq_alt_plus[:2001]
            else:
                seq_alt_plus = seq_alt_plus[-2001:]

        seq_ref_minus = comp(seq_ref_plus)
        seq_alt_minus = comp(seq_alt_plus)
        
        strand = self.strand_dict[gene]
        if strand == '+':
            seq_ref = seq_ref_plus
            seq_alt = seq_alt_plus
        else:
            seq_ref = seq_ref_minus
            seq_alt = seq_alt_minus

        seq_ref = seq_transfer(seq_ref)
        seq_alt = seq_transfer(seq_alt)

        if len(seq_ref) < 2 * self.length + 1:
            padding = torch.zeros((2 * self.length + 1 - len(seq_ref), 4))
            seq_ref = torch.concat([seq_ref, padding])
        if len(seq_alt) < 2 * self.length + 1:
            padding = torch.zeros((2 * self.length + 1 - len(seq_alt), 4))
            seq_alt = torch.concat([seq_alt, padding])

        rbp_expr = torch.tensor(self.expr[tissue])

        return key, seq_ref, seq_alt, rbp_expr, pval_nominal, strand, slope
        
    def get_event(self, event_id):
        
        return self.data_dict[event_id]

parser = ap.ArgumentParser()
parser.add_argument('-t', required=True, action='store', help='tissue')
parser.add_argument('-c', required=True, action='store', help='device')

args = parser.parse_args()
tissue = args.t
device = args.c

expr_fn = 'xx/rbp_expr_qn_mm_rbp.pickle'
with open(expr_fn, 'rb') as f:
    expr = pickle.load(f)

strand_fn = 'xx/gene_strand.pickle'
with open(strand_fn, 'rb') as f:
    strand_dict = pickle.load(f)

batch_size = 64

# model1

model_path_mean = 'xx/model_12.pth'
model_path_diff = 'xx/model_45.pth'

device = torch.device(device)
net_mean = torch.load(model_path_mean).to(device)
net_diff = torch.load(model_path_diff).to(device)
torch.set_num_threads(4)

net_mean.eval()
net_diff.eval()

fmap_block_mean = []
input_block_mean = []

def forward_hook_mean(module, data_input, data_output):
    fmap_block_mean.append(data_output)
    input_block_mean.append(data_input)

net_mean.conv_last1.register_forward_hook(forward_hook_mean)

data_dir = 'xx/GTEX'
file_list = os.listdir(data_dir)
file_list = [i for i in file_list if i.split('associations_')[1].split('.')[0] == tissue]
if len(file_list) == 0:
    print('no files')
output_dir = 'xx/output'

for sqtl_fn in file_list:

    sqtl_fn = '%s/%s' % (data_dir, sqtl_fn)
    tissue = sqtl_fn.split('.')[0].split('associations_')[1]
    chr = sqtl_fn.split('.')[-2]
    if os.path.exists('%s/%s_%s.txt' % (output_dir, tissue, chr)):
        continue

    dataset = sQTL_dataset(sqtl_fn, expr, strand_dict)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True, drop_last=False)

    with torch.no_grad():

        for data in dataloader:

            idx = [i[0] for i in data]
            X_ref = torch.stack([i[1] for i in data]).transpose(1, 2).float().to(device)
            X_alt = torch.stack([i[2] for i in data]).transpose(1, 2).float().to(device)
            rbp  = torch.stack([i[3] for i in data]).float().to(device)
            pval_nominal  = [i[4] for i in data]
            strand = [i[5] for i in data]
            slope = [i[6] for i in data]

            pred_mean = net_mean(X_ref)
            mean_input = torch.stack([input_block_mean[0][0][i].squeeze().detach() for i in range(len(idx))])
            input_block_mean = []
            fmap_block_mean = []
            pred_diff = net_diff(X_ref, rbp, mean_input)

            with open('%s/%s_%s.txt' % (output_dir, tissue, chr), 'a+') as f:
                for i in range(len(idx)):
                    pred_clf = F.softmax(pred_diff[2][i].squeeze().cpu())
                    print(tissue, 
                        'ref',
                        strand[i], 
                        pval_nominal[i], 
                        slope[i], 
                        idx[i], 
                        F.sigmoid(pred_mean[0][i].squeeze()).cpu().numpy(), 
                        F.sigmoid(pred_mean[1][i].squeeze()).cpu().numpy(), 
                        pred_mean[2][i][0].squeeze().cpu().numpy(), 
                        pred_mean[2][i][1].squeeze().cpu().numpy(), 
                        pred_diff[0][i].squeeze().cpu().numpy(), 
                        pred_diff[1][i].squeeze().cpu().numpy(), 
                        pred_clf[0].numpy(), 
                        pred_clf[1].numpy(), 
                        pred_clf[2].numpy(), 
                        sep='\t', file=f)

            pred_mean = net_mean(X_alt)
            mean_input = torch.stack([input_block_mean[0][0][i].squeeze().detach() for i in range(len(idx))])
            input_block_mean = []
            fmap_block_mean = []
            pred_diff = net_diff(X_alt, rbp, mean_input)

            with open('%s/%s_%s.txt' % (output_dir, tissue, chr), 'a+') as f:
                for i in range(len(idx)):
                    pred_clf = F.softmax(pred_diff[2][i].squeeze().cpu())
                    print(tissue, 
                        'alt',
                        strand[i], 
                        pval_nominal[i], 
                        slope[i], 
                        idx[i], 
                        F.sigmoid(pred_mean[0][i].squeeze()).cpu().numpy(), 
                        F.sigmoid(pred_mean[1][i].squeeze()).cpu().numpy(), 
                        pred_mean[2][i][0].squeeze().cpu().numpy(), 
                        pred_mean[2][i][1].squeeze().cpu().numpy(), 
                        pred_diff[0][i].squeeze().cpu().numpy(), 
                        pred_diff[1][i].squeeze().cpu().numpy(), 
                        pred_clf[0].numpy(), 
                        pred_clf[1].numpy(), 
                        pred_clf[2].numpy(), 
                        sep='\t', file=f)
                    

