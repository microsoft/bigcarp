import argparse
import json
from datetime import datetime
import time
import random
import csv

import numpy as np
import torch
import mlflow
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from apex import amp
from apex.optimizers import FusedAdam

import pandas as pd

from sequence_models.layers import PositionFeedForward
from sequence_models.convolutional import ByteNetLM
from sequence_models.metrics import MaskedAccuracy
from sequence_models.losses import MaskedCrossEntropyLoss
from sequence_models.collaters import _pad
from sequence_models.constants import MASK

# Setting up parser
parser = argparse.ArgumentParser(description='Process hyperparameters')

# Defining required arguments for loading data, tokenization, and saving model checkpoints
parser.add_argument('data_dir', type=str, help='directory with ordered protein domain data')
parser.add_argument('token_dir', type=str, help='directory with domain vocabualary')
parser.add_argument('model_dir', type=str, help='directory for saved model checkpoints')
parser.add_argument('results_dir', type=str, help='directory for saving results')
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
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_dir)
# Tokenize
with open(args.token_dir) as f:
    tokens = json.load(f)
specials = tokens['specials']
domains = tokens['domains']
n_tokens = tokens['size']
padding_idx = specials['-']
mask_idx = specials['#']
tokens = []
bgc_test = {}
for _, row in df.iterrows():
    t = [specials[row['function']]]
    for d in row['domains'].split(';'):
        if d in domains:
            t.append(domains[d])
        else:
            t.append(domains['UNK'])
    t.append(specials['*'])
    tokens.append(torch.tensor(t))
    if row['split'] == 'test':
        if row['function'] in bgc_test:
            bgc_test[row['function']].append(torch.tensor(t))
        else:
            bgc_test[row['function']] = [torch.tensor(t)]
test_tokens = [tokens[i] for i in df[df['split'] == 'test'].index]
n_test = len(test_tokens)

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
        mod_idx = [0] # mask first token -> bgc class token
        # mod_idx = random.sample(list(range(len(t))), int(len(t) * 0.15)) 
        # if len(mod_idx) == 0:
        #     mod_idx = [np.random.choice(len(t))]  # make sure at least one domain token is chosen
        for idx in mod_idx:
        #     p = np.random.uniform()
        #     if p <= 0.10:  # do nothing
        #         mod = t[idx]
        #     elif 0.10 < p <= 0.20:  # replace with random pfam domain token
        #         mod = np.random.choice([domains[d] for d in domains if domains[d] != t[idx]])
        #     else:  # mask
        #         mod = mask_idx
            s[idx] = mask_idx 
        src.append(s)
        m = torch.zeros(len(t))
        m[mod_idx] = 1
        mask.append(m)
    src = _pad(src, padding_idx)
    tgt = _pad(tgt, padding_idx)
    mask = _pad(mask, 0)
    return src, tgt, mask

# Define ListDataset class for ds/dl
class ListDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return (self.data[item], )

    def __len__(self):
        return len(self.data)

# Hyperparameters
batch_size = args.batch_size
torch.manual_seed(0)
# ells = [len(t) for t in test_tokens]
d_embedding = args.d_embedding
d_model = args.d_model
n_layers = args.n_layers
kernel_size = args.kernel_size
r = args.r
lr = args.lr
opt_level = 'O2'
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:%d' %args.gpu)

# Initializing model class and loading saved state_dict
if args.freeze:
    n_frozen_embs = len(domains) - 1 # -1 for the UNK domain
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                  padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu', n_frozen_embs=n_frozen_embs)
else:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                  padding_idx=mask_idx, causal=False, final_ln=True, activation='gelu')
optimizer = FusedAdam(model.parameters(), lr=lr)
model = model.to(device)
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
loss_func = MaskedCrossEntropyLoss()
accu_func = MaskedAccuracy()

## load the model checkpoint
checkpoint = torch.load(args.model_dir)
## load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
print('Previously trained model weights state_dict loaded...')
## load trained optimizer state_dict
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Previously trained optimizer state_dict loaded...')

# Test
def epoch(model):
    start_time = datetime.now()
    model = model.eval()
    ds_test = ListDataset(bgc_test[key])
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=mlmcollate)
    loader = dl_test
    t = 'Testing:'
    losses = []
    accus = []
    ns = []
    n_seen = 0
    n_total = len(ds_test)
    for i, batch in enumerate(loader):
        new_loss, new_accu, new_n = step(model, batch)
        losses.append(new_loss * new_n)
        accus.append(new_accu * new_n)
        ns.append(new_n)
        n_seen += len(batch[0])
        total_n = sum(ns)
        rloss = sum(losses) / total_n
        raccu = sum(accus) / total_n
        print('\r%s Example %d of %d loss = %.4f accu = %.4f'
              % (t, n_seen, n_total, rloss, raccu),
              end='')
    mlflow.log_metrics({key + '_loss': rloss,
                        key + '_accu': raccu})
    with open(args.results_dir, 'a', newline='') as f:
        results = csv.writer(f)
        results.writerow([key, n_total, rloss, raccu])
        f.close()
    print('\nTesting complete in ' + str(datetime.now() - start_time))
    return i

def step(model, batch):
    src, tgt, mask = batch
    src = src.to(device)
    tgt = tgt.to(device)
    mask = mask.to(device)
    input_mask = (src != padding_idx).float()
    outputs = model(src, input_mask=input_mask.unsqueeze(-1))
    loss = loss_func(outputs, tgt, mask)
    accu = accu_func(outputs, tgt, mask)
    return loss.item(), accu.item(), mask.sum().item()

# Setting up run
results_columns = ['bgc_class', 'n_sequences', 'loss', 'accuracy']
with open(args.results_dir, 'w', newline='') as f:
    results = csv.writer(f)
    results.writerow(results_columns)

mlflow.set_experiment('bgc')
n_parameters = sum(p.numel() for p in model.parameters())
print('%d model parameters' % n_parameters)
print('%d test sequences' % n_test)
for key in bgc_test:
    with torch.no_grad():
        _ = epoch(model)