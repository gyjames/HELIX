import os
import pandas as pd
import numpy as np
import pickle
from functools import reduce
import shutil
import sys

chr_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']

# protein coding gene list

p_g = []
fn = '/data/workdir/zhouzh/resource/gencode.v45.primary_assembly.annotation.gtf.protein_coding_gene'
with open(fn, 'r') as f:
    for line in f:
        p_g.append(line.strip())

# output gtf

t_dict = {}
gtf_complete = open('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/gtf_bambu.gtf', 'a+')

for chri in chr_list:
    print(chri)
    fn = '/data/workdir/zhouzh/ProjectIsoPred/Third_generation/Training_data/bambu_out_0.1/%s/extended_annotations.gtf' % chri
    with open(fn, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            if info[0] != chri:
                continue
            tp = info[2]
            strand = info[6]
            start = int(info[3])
            end = int(info[4])
            supp = info[-1]
            gene_id = supp.split('; ')[0].split(' ')[1].strip('"')
            transcript_id = supp.split('; ')[1].split(' ')[1].strip('";')

            if tp == 'exon':
                exon_number = supp.split('; ')[2].split(' ')[1].strip('";')

            if transcript_id.startswith('Bambu'):
                transcript_id = chri + '_' + transcript_id
                print(transcript_id)
            if gene_id.startswith('Bambu'):
                gene_id = chri + '_' + gene_id
            
            if tp == 'transcript':
                if gene_id.startswith(chri) or transcript_id.startswith(chri):
                    supp = '; '.join(['gene_id "%s"' % gene_id, 'transcript_id "%s"' % transcript_id])
                if transcript_id not in t_dict:
                    t_dict[transcript_id] = []
                    print(chri, 'Bambu', tp, start, end, '.', strand, '.', supp, sep='\t', file=gtf_complete)

            if tp == 'exon':
                if gene_id.startswith(chri) or transcript_id.startswith(chri):
                    supp = '; '.join(['gene_id "%s"' % gene_id, 'transcript_id "%s"' % transcript_id, 'exon_number "%s"' % exon_number])
                if exon_number not in t_dict[transcript_id]:
                    t_dict[transcript_id].append(exon_number)
                    print(chri, 'Bambu', tp, start, end, '.', strand, '.', supp, sep='\t', file=gtf_complete)     
            
# To dict

gtf_dict = {}
for chri in chr_list:
    print(chri)
    fn = '/data/workdir/zhouzh/ProjectIsoPred/Third_generation/Training_data/bambu_out_0.1/%s/extended_annotations.gtf' % chri
    with open(fn, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            if info[0] != chri or info[2] != 'exon':
                continue
            strand = info[6]
            start = int(info[3])
            end = int(info[4])
            supp = info[-1]
            gene_id = supp.split('; ')[0].split(' ')[1].strip('"')
            transcript_id = supp.split('; ')[1].split(' ')[1].strip('"')

            if transcript_id.startswith('Bambu'):
                transcript_id = chri + '_' + transcript_id
            if gene_id.startswith('Bambu'):
                gene_id = chri + '_' + gene_id
            
            if gene_id not in gtf_dict:
                gtf_dict[gene_id] = {'strand':strand, 'chr':chri, 't':{transcript_id:[start, end]}}
            elif transcript_id not in gtf_dict[gene_id]['t']:
                gtf_dict[gene_id]['t'][transcript_id] = [start, end]
            else:
                gtf_dict[gene_id]['t'][transcript_id].extend([start, end])

# 5'->3'
for k, v in gtf_dict.items():
    if v['strand'] == '+':
        for t, tv in v['t'].items():
            v['t'][t] = sorted(tv)
    elif v['strand'] == '-':
        for t, tv in v['t'].items():
            v['t'][t] = sorted(tv)[::-1]

with open('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/gtf_dict_gene%s.pickle' % len(gtf_dict), 'wb') as f:
    pickle.dump(gtf_dict, f)

# Keep protein coding genes; transcript isoforms with >= 2 exons

gtf_dict_v2 = {}
for k, v in gtf_dict.items():
    if k in p_g:
        gtf_dict_v2[k] = v

gtf_dict_v3 = {}
for k, v in gtf_dict_v2.items():
    for t, tv in v['t'].items():
        if len(tv) > 2:
            if k not in gtf_dict_v3:
                gtf_dict_v3[k] = {'strand':v['strand'], 'chr':v['chr'], 't':{t:tv}}
            else:
                gtf_dict_v3[k]['t'][t] = tv

with open('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/gtf_dict_pg_multiple_exons_gene%s.pickle' % len(gtf_dict_v3), 'wb') as f:
    pickle.dump(gtf_dict_v3, f)


new_tx_list = []
ref_tx_list = []

for chri in chr_list:

    print(chri)

    fn = '/data/workdir/zhouzh/ProjectIsoPred/Third_generation/Training_data/bambu_out_0.1/%s/counts_transcript.txt' % chri

    df_tmp = pd.read_csv(fn, sep='\t', index_col=0)
    tx = [i for i in df_tmp.index if i.startswith('ENST')]
    new_tx = [i for i in df_tmp.index if i.startswith('Bambu')]
    df_newtx = df_tmp.loc[new_tx]
    gene_newtx = [chri + '_' + i if i.startswith('Bambu') else i for i in df_newtx['GENEID']]
    df_newtx['GENEID'] = gene_newtx
    new_idx = [chri + '_' + i for i in new_tx]
    df_newtx.index = new_idx
    new_tx_list.append(df_newtx)

    df_tmp = df_tmp.loc[tx]
    ref_tx_list.append(df_tmp)

def add(a, b):
    return a.iloc[:, 1:] + b.iloc[:, 1:]

df_ref_tx = reduce(add, ref_tx_list)
df_ref_tx['GENEID'] = df_tmp['GENEID']
df_new_tx = pd.concat(new_tx_list, axis=0)
df = pd.concat([df_new_tx, df_ref_tx], axis=0)
df = df.fillna(0)

df.to_csv('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/tx_count_sample%s_tx%s.tsv' % (df.shape[1] - 1, df.shape[0]), sep='\t')

df.iloc[:, 1:] = df.iloc[:, 1:] / df.iloc[:, 1:].sum(axis=0) * 1000000
df.to_csv('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/tx_cpm_sample%s_tx%s.tsv' % (df.shape[1] - 1, df.shape[0]), sep='\t')

# TSS group

fn = '/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/tss_group.tsv'

for k, v in gtf_dict_v3.items():
    
    transcript = v['t']
    t_list = []
    tss_list = []
    for t in v['t'].keys():
        t_list.append(t)
        tss_list.append(v['t'][t][0])

    if v['strand'] == '+':
        idx_sort = np.argsort(tss_list)
    else:
        idx_sort = np.argsort(tss_list)[::-1]

    tss_list = [tss_list[i] for i in idx_sort]
    t_list = [t_list[i] for i in idx_sort]

    idx = 1
    i = 0
    start_site = tss_list[i]
    thr = 150
    tss_group_dict = {k + '.' + str(idx):[t_list[i]]}

    if len(tss_list) >= 2:

        while i < len(tss_list) - 1:

            i += 1
            end_site = tss_list[i]
            if abs(end_site - start_site) < thr:
                tss_group_dict[k + '.' + str(idx)].append(t_list[i])
            else:
                idx += 1
                tss_group_dict[k + '.' + str(idx)] = [t_list[i]]
            
            start_site = end_site

    for group_id, group_v in tss_group_dict.items():
        for t in group_v:
            with open(fn, 'a+') as fo:
                print(k, group_id, t, v['t'][t][0], v['strand'], v['chr'], len(group_v), sep='\t', file=fo)

# Splice site chain
 
group_chain_dict = {}
for g in list(set(group_df['tss_group'])):
    print(g)
    gene = '.'.join(g.split('.')[:-1])
    strand = list(group_df.loc[group_df['tss_group']==g, 'strand'])[0]
    print(strand)
    ss_chain = []
    t_list = group_df.loc[group_df['tss_group']==g, 'tx']
    for t in t_list:
        ss_chain_site = gtf_dict_v3[gene]['t'][t]
        ss_chain_label = ['tss'] + ['d', 'a'] * int(len(ss_chain_site) / 2 - 1) + ['tes']
        ss_chain.extend([str(ss_chain_site[i]) + '.' + ss_chain_label[i] for i in range(len(ss_chain_site))])
    if strand == '+':
        group_chain_dict[g] = sorted(list(set(ss_chain)), key = lambda x:int(x.split('.')[0]))
    else:
        group_chain_dict[g] = sorted(list(set(ss_chain)), key = lambda x:int(x.split('.')[0]))[::-1]

with open('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/tss_group_chain_dict.pickle', 'wb') as f:
    pickle.dump(group_chain_dict, f)
