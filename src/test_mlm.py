import argparse
import json
from datetime import datetime
from tqdm import tqdm
import random
import pathlib

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from apex import amp
from apex.optimizers import FusedAdam

import pandas as pd

from sequence_models.convolutional import ByteNetLM
from sequence_models.metrics import MaskedAccuracy
from sequence_models.losses import MaskedCrossEntropyLoss
from sequence_models.collaters import _pad

# Setting up parser
parser = argparse.ArgumentParser(description='Process hyperparameters')

# Defining required arguments for loading data, tokenization, and saving model checkpoints
parser.add_argument('--model_fpath', type=str, default=os.getenv('PT_MAP_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--out_fpath', type=str, required=False, default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--freeze', action='store_true')


# Defining optional arguments for hyperparameterization; parsing arguments
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--d_embedding', type=int, default=1280, help='dimension of embedding')
parser.add_argument('--d_model', type=int, default=256, help='dimension to use within ByteNet model')
parser.add_argument('--n_layers', type=int, default=32, help='number of layers of ByteNet block')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel width')
parser.add_argument('--r', type=int, default=128, help='used to calculate dilation factor')
parser.add_argument('--wide', action='store_true')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--unconditional', action='store_true')
parser.add_argument('--ar', action='store_true')
args = parser.parse_args()

# Load data
try:
    data_fpath = os.getenv('PT_DATA_DIR') + '/bgc/dedup/'
except:
    home = str(pathlib.Path.home())
    data_fpath = home + '/data/bgc/dedup/'
df = pd.read_csv(data_fpath + 'pfams_6_genomes_deduplicate.csv')
# Tokenize
with open(data_fpath + 'final_domain_vocab.json') as f:
    tokens = json.load(f)
specials = tokens['specials']
domains = tokens['domains']
domain_tokens = np.array([domains[d] for d in domains])
n_tokens = tokens['size']
padding_idx = specials['-']
mask_idx = specials['#']
tokens = []
for _, row in df.iterrows():
    if args.unconditional:
        t = []
    else:
        t = [specials[row['function']]]
    for d in row['domains'].split(';'):
        if d in domains:
            t.append(domains[d])
        else:
            t.append(domains['UNK'])
    tokens.append(torch.tensor(t))
test_tokens = [tokens[i] for i in df[df['split'] == 'test'].index]


def mlmcollate(batch):
    data = tuple(zip(*batch))
    tgt = data[0]
    src = []
    mask = []
    for t in tgt:
        s = t.clone().detach()
        if len(t) == 0:
            tgt.remove(t)
            continue
        mod_idx = random.sample(list(range(len(t))), int(len(t) * 0.15))
        if len(mod_idx) == 0:
            mod_idx = [np.random.choice(len(t))]  # make sure at least one domain token is chosen
        for idx in mod_idx:
            p = np.random.uniform()
            if p <= 0.10:  # do nothing
                mod = t[idx]
            elif 0.10 < p <= 0.20:  # replace with random pfam domain token
                mod = np.random.choice(domain_tokens[domain_tokens != t[idx].item()])
            else:  # mask
                mod = mask_idx
            s[idx] = mod
        src.append(s)
        m = torch.zeros(len(t))
        m[mod_idx] = 1
        mask.append(m)
    src = _pad(src, padding_idx)
    tgt = _pad(tgt, padding_idx)
    mask = _pad(mask, 0)
    return src, tgt, mask


class ListDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return (self.data[item],)

    def __len__(self):
        return len(self.data)


batch_size = args.batch_size
torch.manual_seed(0)
ds_test = ListDataset(test_tokens)
ells = [len(t) for t in test_tokens]
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=mlmcollate)

# Load model
d_embedding = args.d_embedding
d_model = args.d_model
n_layers = args.n_layers
kernel_size = args.kernel_size
r = args.r
torch.cuda.set_device(args.gpu)

device = torch.device('cuda:%d' % args.gpu)
if args.freeze:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                      padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu', n_frozen_embs=19450)
else:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                      padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu')

metrics = pd.read_csv(args.model_fpath + 'metrics.csv', header=None)
metrics.columns = ['loss', 'accu', 'epoch', 'steps']
epoch = int(metrics.sort_values(['loss']).iloc[0]['epoch'])
weight_fpath = args.model_fpath + 'checkpoint%d.tar' % epoch

sd = torch.load(weight_fpath, map_location=torch.device('cpu'))
msd = sd['model_state_dict']
model.load_state_dict(msd)
model = model.to(device)

n_parameters = sum(p.numel() for p in model.parameters())
print('%d model parameters' % n_parameters)
loss_func = MaskedCrossEntropyLoss()
accu_func = MaskedAccuracy()


# Train
def epoch(model):
    model = model.eval()
    loader = dl_test
    domain_losses = 0
    class_losses = 0
    domain_accus = 0
    class_accus = 0
    domain_ns = 0
    class_ns = 0
    chunk_time = datetime.now()
    n_total = len(ds_test)
    for i, batch in enumerate(loader):
        dloss, daccu, closs, caccu, nd, nc = step(model, batch)
        domain_losses += dloss * nd
        domain_accus += daccu * nd
        class_losses += closs * nc
        class_accus += caccu * nc
        domain_ns += nd
        class_ns += nc
    return domain_losses / domain_ns, domain_accus / domain_ns, class_losses / class_ns, class_accus / class_ns


def step(model, batch):
    src, tgt, mask = batch
    src = src.to(device)
    tgt = tgt.to(device)
    mask = mask.to(device)
    input_mask = (src != padding_idx).float()
    outputs = model(src, input_mask=input_mask.unsqueeze(-1))
    domain_loss = loss_func(outputs[:, 1:], tgt[:, 1:], mask[:, 1:]).item()
    domain_accu = accu_func(outputs[:, 1:], tgt[:, 1:], mask[:, 1:]).item()
    nc = mask[:, 0].sum().item()
    if nc > 0:
        class_loss = loss_func(outputs[:, 0], tgt[:, 0], mask[:, 0]).item()
        class_accu = accu_func(outputs[:, 0], tgt[:, 0], mask[:, 0]).item()
    else:
        class_loss = 0
        class_accu = 0
    return domain_loss, domain_accu, class_loss, class_accu, mask[:, 1:].sum().item(), nc


n_parameters = sum(p.numel() for p in model.parameters())
print('%d model parameters' % n_parameters)
for e in tqdm(range(100)):
    with torch.no_grad():
        dl, da, cl, ca = epoch(model)
    with open(args.out_fpath + '/test.csv', 'a') as f:
        f.write('%f,%f,%f,%f\n' %(dl, da, cl, ca))

