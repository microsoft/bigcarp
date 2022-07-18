import argparse
import json
from tqdm import tqdm
import os
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from apex.optimizers import FusedAdam

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from sequence_models.layers import PositionFeedForward
from sequence_models.convolutional import ByteNetLM
from sequence_models.utils import parse_fasta

# Setting up parser
parser = argparse.ArgumentParser(description='Process hyperparameters')

# Defining required arguments for loading data, tokenization, and saving model checkpoints
parser.add_argument('--map_dir', type=str, default=os.getenv('PT_MAP_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--out_fpath', type=str, required=False, default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--gpu', '-g', type=int, default=0)

# Defining optional arguments for hyperparameterization; parsing arguments
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--window_size', type=int, default=256, help='window size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--d_embedding', type=int, default=1280, help='dimension of embedding')
parser.add_argument('--d_model', type=int, default=256, help='dimension to use within ByteNet model')
parser.add_argument('--n_layers', type=int, default=32, help='number of layers of ByteNet block')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel width')
parser.add_argument('--r', type=int, default=128, help='used to calculate dilation factor')
parser.add_argument('--wide', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

args = parser.parse_args()
if 'freeze' in args.map_dir:
    args.freeze = True
# Load data
try:
    data_fpath = os.getenv('PT_DATA_DIR') + '/bgc/'
except:
    home = str(pathlib.Path.home())
    data_fpath = home + '/data/bgc/'
df_test = pd.read_csv(data_fpath + 'output/6_genomes.csv')
# Tokenize
with open(data_fpath + 'dedup/final_domain_vocab.json') as f:
    tokens = json.load(f)
specials = tokens['specials']
domains = tokens['domains']
n_tokens = tokens['size']
padding_idx = specials['-']
mask_idx = specials['#']
_, names = parse_fasta(data_fpath + 'dedup/final_pfams.fasta', return_names=True)
pfam_to_domain = {}
for name in names:
    s = name.split(';')
    domain = s[-2]
    pfam = s[-3].split(' ')[-1].split('.')[0]
    pfam_to_domain[pfam] = domain

window_size = args.window_size
test_tokens = []
test_labels = []
for _, row in df_test.iterrows():
    t = []
    pfams = row['domains'].split(';')
    for d in pfams:
        if d in pfam_to_domain:
            d = pfam_to_domain[d]
        if d in domains:
            t.append(domains[d])
        else:
            t.append(domains['UNK'])
    labels = row['cluster_labels '].split(';')
    assert len(labels) == len(t)
    for i in range(len(t) - window_size + 1):
        test_tokens.append(t[i: i + window_size])
    test_labels.append([int(lab) for lab in labels])
test_tokens = torch.tensor(test_tokens)
test_labels = [np.array(tl) for tl in test_labels]

batch_size = args.batch_size
torch.manual_seed(0)
np.random.seed(0)
ds = TensorDataset(test_tokens)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=16)
# Load model
d_embedding = args.d_embedding
d_model = args.d_model
n_layers = args.n_layers
kernel_size = args.kernel_size
r = args.r
lr = args.lr
epochs = args.epochs
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:%d' % args.gpu)
if args.freeze:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                      padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu', n_frozen_embs=19450)
else:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                      padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu')

embedder = model.embedder
decoder = nn.Sequential(nn.LayerNorm(d_model), nn.GELU(),
                        PositionFeedForward(d_model, d_model), nn.GELU(), PositionFeedForward(d_model, 1))
model = nn.ModuleDict({'embedder': embedder, 'decoder': decoder})
metrics = pd.read_csv(args.map_dir + 'metrics.csv', header=None)
metrics.columns = ['train_loss', 'train_accu', 'valid_loss', 'valid_accu', 'valid_auc']
metrics['epoch'] = np.arange(len(metrics))
for _, row in metrics.iterrows():
    model = model.to('cpu')
    epoch = row['epoch']
    weight_fpath = args.map_dir + 'checkpoint%d.tar' %epoch
    sd = torch.load(weight_fpath, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
    model.load_state_dict(msd)
    model = model.to(device).eval()
    preds = []
    with torch.no_grad(), tqdm(total=len(dl)) as pbar:
        for i, (src,) in enumerate(dl):
            src = src.to(device)
            output = model['embedder'](src)
            output = model['decoder'](output).squeeze()
            preds.append(output.detach().cpu().numpy())
            pbar.update(1)
    preds = np.concatenate(preds)
    ells = [len(t) for t in test_labels]
    all_p = np.empty((sum(ells), window_size)) + np.nan
    i = 0  # idx in hs
    j = 0  # idx in all_h
    idx = np.arange(window_size)
    for tg in test_labels:
        ell = len(tg)
        for k, t in enumerate(tg):
            if k == ell - window_size + 1:
                j += window_size - 1
                break
            all_p[j + idx, idx] = preds[i]
            i += 1
            j += 1
    all_preds = np.nanmean(all_p, axis=-1)
    np.savez_compressed(args.out_fpath + '%dp.npz' %epoch, p=all_preds)
    metrics.loc[epoch, 'test_auc'] = roc_auc_score(np.concatenate(test_labels), all_preds)
    metrics.to_csv(args.out_fpath + 'metrics.csv', index=False)

