import os
import pandas as pd
import numpy as np
import pickle
from functools import reduce
import shutil
import sys
import argparse as ap

def gtf_process(gtf, sample_list, out_d):

    gtf_dict = {}
    site_list = []

    with open(gtf, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            info = line.strip().split('\t')
            if info[2] != 'exon':
                continue
            chr = line.strip().split('\t')[0]
            strand = info[6]
            start = int(info[3])
            end = int(info[4])
            supp = info[-1]
            gene_id = supp.split('; ')[0].split(' ')[1].strip('"')
            transcript_id = supp.split('; ')[1].split(' ')[1].strip('"')
            if strand == '+':
                site_list.append('%s|%s|%s|%s|%s' % (chr, strand, 'a', gene_id, start))
                site_list.append('%s|%s|%s|%s|%s' % (chr, strand, 'd', gene_id, end))
            else:
                site_list.append('%s|%s|%s|%s|%s' % (chr, strand, 'd', gene_id, start))
                site_list.append('%s|%s|%s|%s|%s' % (chr, strand, 'a', gene_id, end))
            
            if gene_id not in gtf_dict:
                gtf_dict[gene_id] = {'strand':strand, 'chr':chr, 't':{transcript_id:[start, end]}}
            elif transcript_id not in gtf_dict[gene_id]['t']:
                gtf_dict[gene_id]['t'][transcript_id] = [start, end]
            else:
                gtf_dict[gene_id]['t'][transcript_id].extend([start, end])
    
    site_list = list(set(site_list))

    with open('%s/splice_site_input.txt' % out_d, 'a+') as f:
        for s in sample_list:
            for i in site_list:
                print('%s|%s' % (i, s), i.split('|')[0], i.split('|')[1], i.split('|')[3], i.split('|')[2], i.split('|')[4], s, 0, 0, file=f, sep='\t')
    
    for k, v in gtf_dict.items():
        if v['strand'] == '+':
            for t, tv in v['t'].items():
                v['t'][t] = sorted(tv)
        elif v['strand'] == '-':
            for t, tv in v['t'].items():
                v['t'][t] = sorted(tv)[::-1]

    gtf_dict_long_t = {}
    for k, v in gtf_dict.items():
        for t, tv in v['t'].items():
            if len(tv) > 2:
                if k not in gtf_dict_long_t:
                    gtf_dict_long_t[k] = {'strand':v['strand'], 'chr':v['chr'], 't':{t:tv}}
                else:
                    gtf_dict_long_t[k]['t'][t] = tv

    with open('%s/gtf_dict.pickle' % (out_d), 'wb') as f:
        pickle.dump(gtf_dict_long_t, f)

    return gtf_dict_long_t

def TSS_group_split(gtf_dict, out_d):

    fn = '%s/tss_group.tsv' % out_d

    for k, v in gtf_dict.items():
        
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

    group_df = pd.read_csv(fn, sep='\t', names=['gene', 'tss_group', 'tx', 'start_site', 'strand', 'chr', 'n'])
    group_df = group_df.loc[group_df['n'] >= 2]

    return group_df

def output_tx_model_input(gtf_dict, TSS_group, sample_list):

    # Splice site chain
    
    group_chain_dict = {}
    for g in list(set(TSS_group['tss_group'])):
        gene = '.'.join(g.split('.')[:-1])
        strand = list(TSS_group.loc[TSS_group['tss_group']==g, 'strand'])[0]
        ss_chain = []
        t_list = TSS_group.loc[TSS_group['tss_group']==g, 'tx']
        for t in t_list:
            ss_chain_site = gtf_dict[gene]['t'][t]
            ss_chain_label = ['tss'] + ['d', 'a'] * int(len(ss_chain_site) / 2 - 1) + ['tes']
            ss_chain.extend([str(ss_chain_site[i]) + '.' + ss_chain_label[i] for i in range(len(ss_chain_site))])
        if strand == '+':
            group_chain_dict[g] = sorted(list(set(ss_chain)), key = lambda x:int(x.split('.')[0]))
        else:
            group_chain_dict[g] = sorted(list(set(ss_chain)), key = lambda x:int(x.split('.')[0]))[::-1]

    with open('%s/tss_group_chain_dict.pickle' % out_d, 'wb') as f:
        pickle.dump(group_chain_dict, f)

    # output tx model input

    fn = '%s/tx_input.txt' % out_d

    tss_groups = list(set(TSS_group['tss_group']))
    for g in tss_groups:
        gene = '.'.join(g.split('.')[:-1])
        chr = gtf_dict[gene]['chr']
        strand = gtf_dict[gene]['strand']
        t_list = list(TSS_group.loc[TSS_group['tss_group'] == g, 'tx'])
        for t in t_list:
            g_chain = group_chain_dict[g]
            g_chain_site = [int(i.split('.')[0]) for i in g_chain]
            t_chain = gtf_dict[gene]['t'][t]
            label = [1 if i in t_chain else 0 for i in g_chain_site]
            tp_label = [i.split('.')[1] for i in g_chain]

            t_chain_label = ','.join([str(i) for i in t_chain])
            g_chain_label = ','.join([str(i) for i in g_chain_site])
            label = ','.join([str(i) for i in label])
            tp_label = ','.join([str(i) for i in tp_label])
            with open(fn, 'a+') as fo:
                for sample in sample_list:
                    print(sample, gene, chr, strand, g, t, g_chain_label, t_chain_label, label, tp_label, 0, sep='\t', file=fo)

def extend_genenames(sc_df, genes):
    sc_samples = sc_df.columns
    ref_df = pd.DataFrame({'ref':0}, index=genes)
    sc_df = pd.merge(left=ref_df, left_index=True, right=sc_df, right_index=True, how='left')
    sc_df = sc_df.fillna(0)
    sc_df = sc_df[sc_samples]
    return sc_df

def quantile_normalization(df, ref_mean):

    expr_bulk_quantiled = []
    for sample in df.columns:
        tmp = df[sample].sort_values()
        new_index = tmp.index
        new_tmp = pd.Series(ref_mean.values, index=new_index, name=sample)
        expr_bulk_quantiled.append(new_tmp)

    expr_bulk_quantiled = pd.concat(expr_bulk_quantiled, axis=1)
    return expr_bulk_quantiled

def min_max_normalization(df, min_value, max_value):

    df = df.T
    df = (df - min_value)/(max_value - min_value)
    df = df.T
    df = df.fillna(0)
    df[df > 1] = 1
    df[df < 0] = 0
    return df

# parameters

parser = ap.ArgumentParser()
parser.add_argument('-g', required=True, action='store', help='Transcript annotation in gtf format.')
parser.add_argument('-o', required=True, action='store', help='Output directory.')
parser.add_argument('-r', required=True, action='store', help='Data path')
args = parser.parse_args()

gtf = args.g
out_d = args.o
expr_fn = args.r

# generate RBP input

expr = pd.read_csv(expr_fn, sep='\t', index_col=0)
expr = expr.loc[~expr.index.duplicated(), :]

with open('/data/workdir/zhouzh/ProjectIsoPred/TS/Training_data/gene_expr/sample358_rbp_normalization_material.pickle', 'rb') as f:
    norm_mat = pickle.load(f)

expr_sc = extend_genenames(expr, norm_mat['gene_name'])
expr_sc_qn = quantile_normalization(expr_sc, norm_mat['ref_mean'])
expr_sc_qn = expr_sc_qn.loc[norm_mat['min_value'].index]
expr_sc_qn_mm = min_max_normalization(expr_sc_qn, norm_mat['min_value'], norm_mat['max_value'])
expr_sc_qn_mm = expr_sc_qn_mm.loc[norm_mat['min_value'].index]
expr_sc_qn_mm.to_csv('%s/rbp_expr_qn_mm_rbp.tsv' % out_d, sep='\t')

expr_dict = {}
clusters = expr_sc_qn_mm.columns
for cluster in clusters:
    expr_dict[cluster] = np.asarray(expr_sc_qn_mm[cluster])

with open('%s/rbp_expr_qn_mm_rbp.pickle' % out_d, 'wb') as f:
    pickle.dump(expr_dict, f)

# generate splice site / transcript model input

sample_list = list(expr_dict.keys())

gtf_dict = gtf_process(gtf, sample_list, out_d)
TSS_group = TSS_group_split(gtf_dict, out_d)
output_tx_model_input(gtf_dict, TSS_group, sample_list)

