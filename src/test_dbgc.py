import argparse
import json
from tqdm import tqdm
import os
import pathlib

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.set_style('white')

from sequence_models.convolutional import ByteNetLM
from sequence_models.utils import parse_fasta
from sequence_models.collaters import TokenCollater
from sequence_models.datasets import ListDataset


# Setting up parser
parser = argparse.ArgumentParser(description='Process hyperparameters')

# Defining required arguments for loading data, tokenization, and saving model checkpoints
parser.add_argument('--model_fpath', type=str, default=os.getenv('PT_MAP_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--out_fpath', type=str, required=False, default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--six', action='store_true')
parser.add_argument('--mibig', action='store_true')
parser.add_argument('--freeze', action='store_true')

# Defining optional arguments for hyperparameterization; parsing arguments
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--d_embedding', type=int, default=1280, help='dimension of embedding')
parser.add_argument('--d_model', type=int, default=256, help='dimension to use within ByteNet model')
parser.add_argument('--n_layers', type=int, default=32, help='number of layers of ByteNet block')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel width')
parser.add_argument('--r', type=int, default=128, help='used to calculate dilation factor')
parser.add_argument('--wide', action='store_true')
parser.add_argument('--window', type=int, default=64, help='number of domains in sliding window')
args = parser.parse_args()

try:
    data_fpath = os.getenv('PT_DATA_DIR') + '/bgc/'
except:
    home = str(pathlib.Path.home())
    data_fpath = home + '/data/bgc/'
if args.mibig:
    val_fpath = data_fpath + 'output/MiBIG_1406_dataset.txt'
    out_fpath = args.out_fpath + 'mibig.npz'
    df = pd.read_csv(val_fpath, header=None)
    df.columns = ['name', 'activity', 'domains']
else
    if args.six:
        val_fpath = data_fpath + 'output/6_genomes.csv'
        out_fpath = args.out_fpath + '6_genomes_preds.npz'
    else:
        val_fpath = data_fpath + 'output/deepBGC_validation.txt'
        out_fpath = args.out_fpath + 'deep_bgc_preds.npz'
    df = pd.read_csv(val_fpath, sep="\s*[,]\s*", engine='python')
# df.columns = ['name','activity','domains','cluster_labels']
data_fpath += 'dedup/'
with open(data_fpath + 'final_domain_vocab.json') as f:
    tokens = json.load(f)
specials = tokens['specials']
domains = tokens['domains']
domain_tokens = np.array([domains[d] for d in domains])
n_tokens = tokens['size']
padding_idx = specials['-']
mask_idx = specials['#']
tokens = []
_, names = parse_fasta(data_fpath + 'final_pfams.fasta', return_names=True)
pfam_to_domain = {}
for name in names:
    s = name.split(';')
    domain = s[-2]
    pfam = s[-3].split(' ')[-1].split('.')[0]
    pfam_to_domain[pfam] = domain
if not os.path.exists(out_fpath):
    # Load data
    # Tokenize

    for _, row in df.iterrows():
        t = []
        for d in row['domains'].split(';'):
            if d in pfam_to_domain:
                d = pfam_to_domain[d]
            else:
                d = 'UNK'
            if d in domains:
                t.append(domains[d])
            else:
                t.append(domains['UNK'])
        if args.mibig:
            tokens.append(torch.tensor([mask_idx] + t))
        else:
            for i in range(len(t) - args.window + 1):
                tokens.append(torch.tensor([mask_idx] + t[i:i + args.window]))
    n_valid = len(tokens)
    # tokens = torch.stack(tokens)
    collater = TokenCollater(padding_idx)
    batch_size = args.batch_size
    torch.manual_seed(0)
    ds = ListDataset(tokens)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collater, shuffle=False, num_workers=16, pin_memory=True)

    # Load model
    d_embedding = args.d_embedding
    d_model = args.d_model
    n_layers = args.n_layers
    kernel_size = args.kernel_size
    r = args.r
    torch.cuda.set_device(args.gpu)

    device = torch.device('cuda:%d' %args.gpu)
    if args.freeze:
        model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                      padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu', n_frozen_embs=19450)
    else:
        model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                      padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu')

    metrics = pd.read_csv(args.model_fpath + 'metrics.csv', header=None)
    metrics.columns = ['loss', 'accu', 'epoch', 'steps']
    epoch = int(metrics.sort_values(['loss']).iloc[0]['epoch'])
    weight_fpath = args.model_fpath + 'checkpoint%d.tar' %epoch

    sd = torch.load(weight_fpath, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
    model.load_state_dict(msd)
    model = model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('%d model parameters' % n_parameters)
    print('%d validation sequences' % n_valid)
    firsts = []
    hs = []
    losses = []
    model = model.eval()
    with torch.no_grad(), tqdm(total=len(ds)) as pbar:
        for i, (src, ) in enumerate(dl):
            src = src.to(device)
            mask = (src != padding_idx).float().unsqueeze(-1)
            output = model(src, input_mask=mask)
            firsts.append(output[:, 0, 3:58].cpu().detach().numpy())
            p = F.softmax(output, dim=-1)
            for tgt, pre in zip(src, output):
                loss = F.cross_entropy(pre[1:], tgt[1:], reduction='none')
                losses.append(loss.cpu().detach().numpy())
            logp = F.log_softmax(output, dim=-1)
            entropy = -(p * logp).sum(dim=-1)
            hs.append(entropy.cpu().detach().numpy())
            pbar.update(len(src))
    firsts = np.concatenate(firsts)
    if args.mibig:
        np.savez_compressed(out_fpath, first=firsts)
    else:
        hs = np.concatenate(hs)
        losses = np.stack(losses)
        np.savez_compressed(out_fpath, first=firsts, h=hs, loss=losses)

outputs = np.load(out_fpath)
firsts = outputs['first']
if args.mibig:
    classes = ['Alkaloid', 'Saccharide', 'NRP', 'Terpene', 'Polyketide', 'Other', 'RiPP']
    class_to_col = {c:i for i, c in enumerate(classes)}
    out_dir = args.out_fpath + 'mibig'
    with open(data_fpath + 'bgc_class_mapping.json') as f:
        mapping = json.load(f)
    token_to_columns = {}
    for k, v in specials.items():
        if v > 2:
            c = mapping[k]
            if isinstance(c, dict):
                c = list(c.values())
            else:
                c = [c]
            token_to_columns[v - 3] = [class_to_col[cl] for cl in c]
    predictions = np.zeros((len(firsts), len(classes)))
    firsts = softmax(firsts, axis=-1)
    for t, c in token_to_columns.items():
        predictions[:, c] += firsts[:, [t]]
    tgts = np.zeros((len(firsts), len(classes)))
    for i, row in df.iterrows():
        labels = str(row['activity'])
        labels = labels.split(';')
        for label in labels:
            if label in class_to_col:
                tgts[i, class_to_col[label]] = 1
    aucs = np.zeros(len(classes))
    for cl, col in class_to_col.items():
        auc = roc_auc_score(tgts[:, col], predictions[:, col])
        print(cl + '\t%.4f\t%d' %(auc, sum(tgts[:, col])))
        aucs[col] = auc
    np.mean(aucs)
    print(np.mean(aucs))

else:
    hs = outputs['h']
    losses = outputs['loss']
    # print(firsts.shape, hs.shape, losses.shape)
    window = args.window
    p = softmax(firsts, axis=1)
    lp = np.log(p)
    h1 = -(lp * p).sum(axis=-1)
    tgt = [row['cluster_labels'] for i, row in df.iterrows()]
    tgt = [t.split(';') for t in tgt]
    tgt = [[int(tt) for tt in t] for t in tgt]
    is_start = [np.array([t[0] == 1] + [t[i - 1] == 0 and t[i] == 1 for i in range(1, len(t) - window + 1)]) for t in tgt]
    start_pos = [np.where(s == 1)[0] for s in is_start]
    pos = [np.arange(len(t) - window + 1) for t in tgt]
    d_to_start = []
    for s, p in zip(start_pos, pos):
        dists = p[:, None] - s[None, :]
        adist = np.abs(dists)
        idx = np.argmin(adist, axis=-1)
        dists = dists[np.arange(len(dists)), idx]
        dists[np.argwhere(dists > 20)] = 21
        dists[np.argwhere(dists < -20)] = -21
        d_to_start.append(dists)
    df2 = pd.DataFrame()
    df2['in BGC'] = np.concatenate([tg[:-window + 1] for tg in tgt])
    df2['neg start entropy'] = -h1
    pos = []
    seq = []
    for i, tg in enumerate(tgt):
       pos.append(np.arange(len(tg) - window + 1))
       seq.append(np.ones(len(tg) - window + 1) * i)
    df2['pos'] = np.concatenate(pos)
    df2['sequence'] = np.concatenate(seq).astype(int)
    df2['distance to BGC start'] = np.concatenate(d_to_start)
    df2['BGC start'] = np.concatenate(is_start)
    df2['neg cross entropy'] = -losses.mean(axis=1)
    df2['neg pfam entropy'] = -hs[:, 1:].mean(axis=1)
    df2['sum'] = df2['neg start entropy'] + df2['neg cross entropy'] + df2['neg pfam entropy']
    print('score\t\t\tauc')
    for col in ['neg start entropy', 'neg cross entropy', 'neg pfam entropy', 'sum']:
        if col == 'sum':
            tabs = '\t\t\t'
        else:
            tabs = '\t'
        print(col + tabs + '%.4f' %roc_auc_score(df2['BGC start'], df2[col]))

    out_dir = args.out_fpath
    if args.six:
        out_dir += '6'
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    pal = sns.color_palette()
    _ = sns.stripplot(data=df2, x='distance to BGC start', y='sum', alpha=0.1, ax=ax, color=pal[0])
    _ = sns.boxplot(data=df2, x='distance to BGC start', y='sum', ax=ax, color='white')
    _ = fig.savefig(out_dir + 'dists.png', bbox_inches='tight')

    # print(spearmanr(df2['sum'], np.abs(df2['distance to BGC start'])))

    fig, ax = plt.subplots(1, 1)
    _ = sns.stripplot(data=df2, x='BGC start', y='sum', alpha=0.1, ax=ax, color='black')
    _ = sns.boxplot(data=df2, x='BGC start', y='sum', ax=ax)

    _ = fig.savefig(out_dir + 'starts.png', bbox_inches='tight')

    ells = [len(t) for t in tgt]
    all_h = np.empty((sum(ells), window)) + np.nan
    all_L = np.empty((sum(ells), window)) + np.nan
    all_s = np.empty((sum(ells), window)) + np.nan
    i = 0  # idx in hs
    j = 0  # idx in all_h
    idx = np.arange(window)
    for tg in tgt:
        ell = len(tg)
        for k, t in enumerate(tg):
            if k == ell - window + 1:
                j += window - 1
                break
            all_h[j + idx, idx] = hs[i, 1:]
            all_L[j + idx, idx] = losses[i, :]
            all_s[j + idx, idx] = h1[i]
            i += 1
            j += 1

    df3 = pd.DataFrame()
    df3['mean neg start ent'] = -np.nanmean(all_s, axis=1)
    df3['in BGC'] = np.concatenate(tgt)
    fig, ax = plt.subplots(1, 1)
    _ = sns.stripplot(data=df3, x='in BGC', y='mean neg start ent', alpha=0.01, ax=ax, color='black')
    _ = sns.boxplot(data=df3, x='in BGC', y='mean neg start ent', ax=ax, color='white')
    _ = fig.savefig(out_dir + 'pfams.png', bbox_inches='tight')
    print('domain auc: %.4f' %roc_auc_score(np.concatenate(tgt), -np.nanmean(all_s, axis=1)))
    print('domain auc: %.4f' %roc_auc_score(np.concatenate(tgt), -np.nanmin(all_L, axis=1)))
    print('domain auc: %.4f' %roc_auc_score(np.concatenate(tgt), -np.nanmean(all_h, axis=1)))

# h = np.nanmax(all_h, axis=1)
# print(roc_auc_score(np.concatenate(tgt), -np.nanmax(all_h, axis=1)))
# print(roc_auc_score(np.concatenate(tgt), -np.nanmean(all_s, axis=1)))
#
# print(roc_auc_score(np.concatenate(tgt), -np.nanmax(all_L, axis=1)))
# print(roc_auc_score(np.concatenate(tgt), -np.nanmean(all_s, axis=1) - np.nanmean(all_L, axis=1) - np.nanmean(all_h, axis=1)))
#
#
# roc_auc_score(np.concatenate(is_start), -losses.mean(axis=1) - h1 - hs[:, 1:].mean(axis=1))
# spearmanr(np.abs(df2['distance to BGC start']), -losses.mean(axis=1) - h1 - hs[:, 1:].mean(axis=1))
# all_h[j]
# ells[0]
# all_h[14263]
# p.std()