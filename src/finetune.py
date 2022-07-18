import argparse
import json
from datetime import datetime
import time
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
parser.add_argument('--finetune', type=str)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--naive', type=str)

# Defining optional arguments for hyperparameterization; parsing arguments
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--window_size', type=int, default=256, help='window size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--d_embedding', type=int, default=1280, help='dimension of embedding')
parser.add_argument('--d_model', type=int, default=256, help='dimension to use within ByteNet model')
parser.add_argument('--n_layers', type=int, default=32, help='number of layers of ByteNet block')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel width')
parser.add_argument('--r', type=int, default=128, help='used to calculate dilation factor')
parser.add_argument('--wide', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

args = parser.parse_args()
args.naive = args.naive == 'True'
args.finetune = args.finetune == 'True'
# Load data
try:
    data_fpath = os.getenv('PT_DATA_DIR') + '/bgc/'
except:
    home = str(pathlib.Path.home())
    data_fpath = home + '/data/bgc/'
df_train = pd.read_csv(data_fpath + 'deepBGC_training_set_1.csv')
df_train = df_train.rename(columns={'in_cluster': 'cluster_labels'})
df_valid = pd.read_csv(data_fpath + 'output/deepBGC_validation.txt')

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

train_tokens = []
n_train = 0
train_labels = []
valid_tokens = []
valid_labels = []
n_valid = 0
window_size = args.window_size

for _, row in df_valid.iterrows():
    t = []
    pfams = row[' domains'].split(';')
    for d in pfams:
        if d in pfam_to_domain:
            d = pfam_to_domain[d]
        if d in domains:
            t.append(domains[d])
        else:
            t.append(domains['UNK'])
    tokens = torch.tensor(t)[:-(len(t) % window_size)].view(-1, window_size)
    valid_tokens.append(tokens)
    labels = row[' cluster_labels '].split(';')
    assert len(labels) == len(t)

    labels = [int(lab) for lab in labels][:-(len(t) % window_size)]
    valid_labels.append(torch.tensor(labels).view(-1, window_size))
    n_valid += len(labels)
valid_tokens = torch.cat(valid_tokens)
valid_labels = torch.cat(valid_labels).float()

for _, row in df_train.iterrows():
    t = []
    pfams = row['pfams'][1:-1].split(', ')
    pfams = [d[1:-1] for d in pfams]
    for d in pfams:
        if d in pfam_to_domain:
            d = pfam_to_domain[d]
        if d in domains:
            t.append(domains[d])
        else:
            t.append(domains['UNK'])
    train_tokens.append(torch.tensor(t))
    labels = row['cluster_labels'][1:-1].split(', ')
    labels = [int(lab) for lab in labels]
    train_labels.append(torch.tensor(labels).float())
    n_train += len(labels)
    assert len(labels) == len(t)

n_positive = sum([tl.sum() for tl in train_labels])
pc = (n_train - n_positive) / n_positive
class ShuffleDataset(Dataset):

    def __init__(self, tokens, labels, window_size):
        super().__init__()
        self.tokens = tokens
        self.labels = labels
        self.w = window_size
        self.n = sum(len(t) for t in tokens)
        self.n_cut = self.n % self.w
        self.idx = np.arange(len(tokens))
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.idx)
        flat_tokens = torch.cat([self.tokens[i] for i in self.idx])[:-self.n_cut]
        flat_labels = torch.cat([self.labels[i] for i in self.idx])[:-self.n_cut]
        self.stacked_tokens = flat_tokens.view(-1, self.w)
        self.stacked_labels = flat_labels.view(-1, self.w)

    def __getitem__(self, item):
        return (self.stacked_tokens[item], self.stacked_labels[item])

    def __len__(self):
        return self.n // self.w


batch_size = args.batch_size
torch.manual_seed(0)
np.random.seed(0)
ds_train = ShuffleDataset(train_tokens, train_labels, window_size)
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=16)
ds_valid = TensorDataset(valid_tokens, valid_labels)
dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=16)
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
if not args.naive:
    print('Loading pretrained weights...')
    metrics = pd.read_csv(args.map_dir + 'metrics.csv', header=None)
    metrics.columns = ['loss', 'accu', 'epoch', 'steps']
    epoch = int(metrics.sort_values(['loss']).iloc[0]['epoch'])
    weight_fpath = args.map_dir + 'checkpoint%d.tar' %epoch

    sd = torch.load(weight_fpath, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
    model.load_state_dict(msd)

# model.embedder.embedder.weight = nn.Parameter(model.embedder.embedder.weight * 1.25)
embedder = model.embedder
decoder = nn.Sequential(nn.LayerNorm(d_model), nn.GELU(),
                        PositionFeedForward(d_model, d_model), nn.GELU(), PositionFeedForward(d_model, 1))
model = nn.ModuleDict({'embedder': embedder, 'decoder': decoder}).to(device)
if args.finetune:
    optimizer = FusedAdam(model.parameters(), lr=lr)
else:
    optimizer = FusedAdam(decoder.parameters(), lr=lr)
# loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.ones(window_size) * pc).to(device)
loss_func = nn.BCEWithLogitsLoss().to(device)


# Train
def epoch(model, train, current_step=0):
    start_time = datetime.now()
    if train:
        model = model.train()
        loader = dl_train
        t = 'Training:'
    else:
        model = model.eval()
        loader = dl_valid
        t = 'Validating:'
    losses = []
    accus = []
    ns = []
    preds = []
    tgts = []
    chunk_time = datetime.now()
    n_seen = 0
    if train:
        n_total = len(ds_train)
    else:
        n_total = len(ds_valid)
    for i, batch in enumerate(loader):
        new_loss, new_accu, new_n, new_preds, new_tgt = step(model, batch, train)
        preds.append(new_preds)
        tgts.append(new_tgt)
        losses.append(new_loss * new_n)
        accus.append(new_accu * new_n)
        ns.append(new_n)
        n_seen += len(batch[0])
        total_n = sum(ns)
        rloss = sum(losses) / total_n
        raccu = sum(accus) / total_n
        if train:
            nsteps = current_step + i + 1
        else:
            nsteps = i
        print('\r%s Epoch %d of %d Step %d Example %d of %d loss = %.4f accu = %.4f'
              % (t, e + 1, epochs, nsteps, n_seen, n_total, rloss, raccu),
              end='')

    if train:
        print('\nTraining complete in ' + str(datetime.now() - chunk_time))
    else:
        print('\nValidation complete in ' + str(datetime.now() - start_time))
    return i, rloss, raccu, preds, tgts


def step(model, batch, train):
    src, tgt = batch
    src = src.to(device)
    tgt = tgt.to(device)
    if args.finetune:
        e = model['embedder'](src)
    else:
        with torch.no_grad():
            e = model['embedder'](src)
    outputs = model['decoder'](e).squeeze()
    loss = loss_func(outputs, tgt)
    accu = torch.mean(((outputs > 0.) == tgt.long()).float())
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    n, ell = src.shape
    return loss.item(), accu.item(), n * ell, outputs.detach().cpu().numpy(), tgt.detach().cpu().numpy()

n_parameters = sum(p.numel() for p in model.parameters())
print('%d model parameters' % n_parameters)
print('%d training tokens' % n_train)
print('%d validation tokens' % n_valid)
total_steps = 0
best_roc = 0
for e in range(epochs):
    new_steps, train_loss, train_accu, _, _ = epoch(model, True, current_step=total_steps)
    total_steps += new_steps
    with torch.no_grad():
        _, valid_loss, valid_accu, preds, tgts = epoch(model, False, current_step=total_steps)
    # print(preds[0].shape, tgts[0].shape)
    preds = np.concatenate(preds).reshape(-1)
    tgts = np.concatenate(tgts).reshape(-1)
    roc = roc_auc_score(tgts, preds)
    print(roc)
    with open(args.out_fpath + 'metrics.csv', 'a') as f:
        f.write(','.join([str(train_loss), str(train_accu), str(valid_loss), str(valid_accu), str(roc)]))
        f.write('\n')
    ds_train.shuffle()
    for _ in range(10):
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.out_fpath + 'checkpoint%d.tar' % e)
            break
        except OSError:
            time.sleep(1)
    else:
        print('Model checkpointing failed!')