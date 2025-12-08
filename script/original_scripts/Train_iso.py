
import os
import TS_Network_iso
import pickle
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, ConcatDataset
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import torch.nn.utils.rnn as R
import argparse as ap
from pyfaidx import Fasta
import gc

one_hot_dict = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1], 'N':[0, 0, 0, 0]}
ref_fn = '/data/workdir/zhouzh/resource/GRCh38.primary_assembly.genome.fa'
fa = Fasta(ref_fn)
L = 200

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

class IsoDataSet(Dataset):

    def __init__(self, fn, rbp, embeddings, value_dict):
    
        self.data_dict = {}
        
        idx = 0
        with open(fn, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                sample = info[0]
                chr = info[2]
                strand = info[3]
                t = info[5]
                g = info[1].split('.')[0]
                chain = info[6].split(',')
                tp_label = info[9].split(',')
                tss = [int(chain[i]) for i in range(len(tp_label)) if tp_label[i] == 'tss']
                tes = [int(chain[i]) for i in range(len(tp_label)) if tp_label[i] == 'tes']
                ss = [int(chain[i]) for i in range(len(tp_label)) if tp_label[i] == 'a' or tp_label[i] == 'd']
                ss_tp = [i for i in tp_label if i == 'a' or i == 'd']
                tp_order = get_tp_order(tp_label)
                # embeddings_key = ['|'.join([chr, strand, ss_tp[i], g, str(ss[i]), sample]) for i in range(len(ss))]
                embeddings_key = ['|'.join([chr, strand, str(ss[i])]) for i in range(len(ss))]
                if_in_embedding = [i for i in embeddings_key if i in embeddings]
                if len(if_in_embedding) != len(embeddings_key):
                    continue
                if len([i for i in tss if i < 1000]) > 0 or len([i for i in tes if i < 1000]) > 0:
                    continue
                chain_label = [int(i) for i in info[8].split(',')]
                prop = float(info[10])

                key = '|'.join([sample, t])
            
                self.data_dict[key] = {'sample':sample, 'chr':chr, 'strand':strand, 't':t, 'tss':tss, 'tes':tes, 'tp_order':tp_order, 'chain':embeddings_key, 'chain_label':chain_label, 'prop':prop}
                idx += 1

        self.event_list = list(self.data_dict.keys())
        self.rbp = rbp
        self.embeddings = embeddings
        value_dict_new = {i + '|' + sample:j for i, j in value_dict.items()}
        self.predict_var = value_dict_new
                
    def __len__(self):
    
        return len(self.data_dict)
        
    def __getitem__(self, idx):
    
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
            tss_seq = [seq_transfer(fa.get_seq(chr, i - L, i + L).seq) for i in tss]
            tes_seq = [seq_transfer(fa.get_seq(chr, i - L, i + L).seq) for i in tes]
        else:
            tss_seq = [seq_transfer(fa.get_seq(chr, i - L, i + L, rc=True).seq) for i in tss]
            tes_seq = [seq_transfer(fa.get_seq(chr, i - L, i + L, rc=True).seq) for i in tes]

        rbp_expr = torch.from_numpy(self.rbp[sample])
        label = torch.LongTensor(chain_label)
        
        embeddings = torch.stack([torch.tensor(self.embeddings[i]) for i in chain])
        # values = torch.stack([F.softmax(torch.tensor([float(j) for j in self.predict_var[i]])) for i in chain])
        values = torch.stack([F.softmax(torch.tensor([float(j) for j in self.predict_var[i + '|' + sample]])) for i in chain])
        embeddings = torch.concat([embeddings, values], axis=1)

        return tss_seq, tes_seq, rbp_expr, embeddings, label, prop, key, tp_order
        
    def get_event(self, event_id): 
        
        return self.data_dict[event_id]


def seq_transfer(seq, one_hot_dict=one_hot_dict):

    seq = torch.from_numpy(np.array([one_hot_dict[i] for i in seq]))

    return seq

def train(dataset, net, batch_size, epoch, log_fn, device, loss_fn, optimizer):

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)
        
    net.train()
    batch = 0

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

        pred = net(tss_seq, tss_len, tes_seq, tes_len, embeddings, rbps, labels, tp_order, device).squeeze(1)

        loss = loss_fn(pred, prop)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        r2 = r2_score(prop.cpu().numpy(), pred.cpu().detach().numpy())
        print(prop.cpu().numpy(), pred.detach().cpu().numpy(), r2)
        with open(log_fn, 'a+') as f:
            print(epoch, batch, loss.item(), r2, sep='\t', file=f)

def test(dataset, net, batch_size, epoch, log_fn_eval, log_fn_res, device, loss_fn):

    dataloader_t = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x:x, pin_memory=True)
    
    net.eval()
    batch = 0

    for data in dataloader_t:

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

        with torch.no_grad():

            pred = net(tss_seq, tss_len, tes_seq, tes_len, embeddings, rbps, labels, tp_order, device).squeeze(1)
            loss = loss_fn(pred, prop)
            r2 = r2_score(prop.cpu().numpy(), pred.cpu().detach().numpy())
        
        with open(log_fn_eval, 'a+') as f:
            print(epoch, batch, loss.item(), r2, sep='\t', file=f)
        
        with open(log_fn_res, 'a+') as f:
            for i in range(len(prop)):
                print(epoch, idx[i], prop[i].cpu().numpy(), pred[i].cpu().detach().numpy(), sep='\t', file=f)

if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-v', required=True, action='store', help='prefix.')
    parser.add_argument('-d', required=True, action='store', help='training data directory.')
    parser.add_argument('-ed', required=True, action='store', help='emb_dict path.')
    parser.add_argument('-vd', required=True, action='store', help='value_dict path.')
    parser.add_argument('-m', required=True, action='store', help='model dir.')
    parser.add_argument('-c', required=True, action='store', help='cuda')
    parser.add_argument('-r', required=True, action='store', help='rbp path')
    parser.add_argument('--rsm', required=False, action='store', default=False, help='if rsm.')
    parser.add_argument('--test', required=False, action='store_true', default=False, help='if test.')
    parser.add_argument('--ts', required=False, action='store',help='test sample path.')
    parser.add_argument('--basemodel', required=False, action='store',help='Base model path.')
    parser.add_argument('--pn', required=False, action='store', default=0.05, help='Base model path.')
    parser.add_argument('--pm', required=False, action='store', default=0.2, help='Base model path.')
    parser.add_argument('--train', required=False, action='store_true', default=False, help='If train')
    parser.add_argument('--nt', required=False, action='store', default=1499, help='Number of trans feature')

    '''

    args = parser.parse_args()
    print(args)

    version = args.v
    data_dir = args.d
    embedding_dir = args.ed
    value_dir = args.vd
    model_dir = args.m
    device = args.c
    rbp_fn = args.r
    nt = int(args.nt)

    rsm = args.rsm
    if_test = args.test
    if_train = args.train
    test_sample_path = args.ts
    basemodel = args.basemodel
    pn = float(args.pn)
    pm = float(args.pm)


    model_path = '%s/%s/model' % (model_dir, version)
    metric_path = '%s/%s/metrics' % (model_dir, version)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)


    device = torch.device(device)
    torch.set_num_threads(8)
    net = TS_Network_iso.iso_v3(nRBP=nt).to(device)
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print('Model loaded.')

    if rsm:
        rsm_state = torch.load(rsm)
        rsm_epoch = rsm_state['epoch']
        net.load_state_dict(rsm_state['model_state_dict'])
        optimizer.load_state_dict(rsm_state['optimizer_state_dict'])
        scheduler.load_state_dict(rsm_state['scheduler_state_dict'])
    else:
        rsm_epoch = -1
        
    batch_size = 16


    if basemodel:
        model_fn = basemodel
        pretrained_model = torch.load(model_fn).to(device)
        names = []
        for name, param in pretrained_model.named_parameters():
            names.append(name)
        
        train_params = ['RBPfc.layers.3.weight', 'RBPfc.layers.3.bias', 'conv1.resblocks.6', 'conv1.resblocks.7', 'conv2.resblocks.6', 'conv2.resblocks.7', 'embedding.weight' ,'l1', 'iso_fc.layers.3']
        train_names = []
        for i in train_params:
            train_names.extend([p for p in names if i in p])
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
    

    samples = os.listdir(embedding_dir)
    samples = ['.'.join(i.split('.')[:-1]) for i in samples]

    test_samples = []

    if if_test:
        with open(test_sample_path, 'r') as f:
            for line in f:
                # test_samples.append(line.strip())
                test_samples.append(line.strip() + '.fastq.gz')
    
    if if_train:
        train_samples = [i for i in samples if i not in test_samples]

    # np.random.seed(10)
    # np.random.shuffle(samples)
    # n_test_samples = 25

    # test_samples = samples[:n_test_samples]
    # train_samples = samples[n_test_samples:]

    log_fn_train = '%s/performance_train.txt' % metric_path

    log_fn_test_eval = '%s/performance_test_testdatachr1_eval.txt' % metric_path
    log_fn_test_res = '%s/performance_test_testdatachr1_res.txt' % metric_path

    rbp = pickle.load(open(rbp_fn, 'rb'))

    n_epoch = 100

    if if_test:

        test_dataset_list_neg = []
        test_dataset_list_pos = []
        test_dataset_list_middle = []

        for sample in test_samples:
            embeddings = pickle.load(open('%s/%s.pickle' % (embedding_dir, sample), 'rb'))
            values = pickle.load(open('%s/%s.pickle' % (value_dir, sample), 'rb'))
            fn_neg = '%s/%s_neg' % (data_dir, sample)
            fn_pos = '%s/%s_pos' % (data_dir, sample)
            fn_middle = '%s/%s_mid' % (data_dir, sample)
            if os.path.exists(fn_neg):
                dataset_neg_test = IsoDataSet(fn_neg, rbp, embeddings, values)
                if len(dataset_neg_test) > 0:
                    test_dataset_list_neg.append(dataset_neg_test)
            if os.path.exists(fn_pos):
                dataset_pos_test = IsoDataSet(fn_pos, rbp, embeddings, values)
                if len(dataset_pos_test) > 0:
                    test_dataset_list_pos.append(dataset_pos_test)
            if os.path.exists(fn_middle):
                dataset_middle_test = IsoDataSet(fn_middle, rbp, embeddings, values)
                if len(dataset_middle_test) > 0:
                    test_dataset_list_middle.append(dataset_middle_test)
        dataset_neg_test = ConcatDataset(test_dataset_list_neg)
        dataset_pos_test = ConcatDataset(test_dataset_list_pos)
        dataset_middle_test = ConcatDataset(test_dataset_list_middle)
        print('test data loaded.')

    # 训练
    for epoch in range(rsm_epoch + 1, n_epoch):

        if if_train:
            for i in range(len(train_samples) // 8 + 1):
                sub_samples = train_samples[i * 8:(i+1) * 8]
                dataset_list_neg = []
                dataset_list_pos = []
                dataset_list_middle = []
                for sample in sub_samples:
                    values = pickle.load(open('%s/%s.pickle' % (value_dir, sample), 'rb'))
                    embeddings = pickle.load(open('%s/%s.pickle' % (embedding_dir, sample), 'rb'))
                    fn_neg = '%s/%s_neg' % (data_dir, sample)
                    fn_pos = '%s/%s_pos' % (data_dir, sample)
                    fn_middle = '%s/%s_mid' % (data_dir, sample)
                    if os.path.exists(fn_neg):
                        dataset_neg = IsoDataSet(fn_neg, rbp, embeddings, values)
                        if len(dataset_neg) > 0:
                            dataset_list_neg.append(dataset_neg)
                    if os.path.exists(fn_pos):
                        dataset_pos = IsoDataSet(fn_pos, rbp, embeddings, values)
                        print(len(dataset_pos))
                        if len(dataset_pos) > 0:
                            dataset_list_pos.append(dataset_pos)
                    if os.path.exists(fn_middle):
                        dataset_middle = IsoDataSet(fn_middle, rbp, embeddings, values)
                        if len(dataset_middle) > 0:
                            dataset_list_middle.append(dataset_middle)
                if len(dataset_list_neg) > 0 and len(dataset_list_pos) > 0 and len(dataset_list_middle) > 0:
                    dataset_neg = ConcatDataset(dataset_list_neg)
                    dataset_neg, _ = random_split(dataset_neg, lengths=[pn, 1 - pn])
                    dataset_middle = ConcatDataset(dataset_list_middle)
                    dataset_middle, _ = random_split(dataset_middle, lengths=[pm, 1 - pm])
                    dataset_pos = ConcatDataset(dataset_list_pos)
                    dataset = ConcatDataset([dataset_neg, dataset_middle, dataset_pos])
                    # print(len(dataset))
                else:
                    continue

                train(dataset, net, batch_size, epoch, log_fn_train, device, loss_fn, optimizer)
                torch.save({'epoch':epoch, 
                            'model_state_dict':net.state_dict(), 
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict':scheduler.state_dict()}, 
                            '%s/checkpoint_%s.pth.tar' % (model_path, epoch))

            scheduler.step()
            torch.save({'epoch':epoch, 
                        'model_state_dict':net.state_dict(), 
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict()}, 
                        '%s/checkpoint_%s.pth.tar' % (model_path, epoch))
            
            torch.save(net, '%s/model_%s.pth' % (model_path, epoch))

        if if_test:

            dataset_neg, _ = random_split(dataset_neg_test, lengths=[pn, 1 - pn])
            dataset_middle, _ = random_split(dataset_middle_test, lengths=[pm, 1 - pm])
            dataset_test = ConcatDataset([dataset_neg, dataset_pos_test, dataset_middle])
            test(dataset_test, net, batch_size, epoch, log_fn_test_eval, log_fn_test_res, device, loss_fn)

