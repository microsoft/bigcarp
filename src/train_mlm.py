import argparse
import json
from datetime import datetime
import time
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
from sequence_models.utils import parse_fasta

# Setting up parser
parser = argparse.ArgumentParser(description='Process hyperparameters')

# Defining required arguments for loading data, tokenization, and saving model checkpoints
parser.add_argument('--out_fpath', type=str, required=False, default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--restart', action='store_true')
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--pretrain', action='store_true')

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
train_tokens = [tokens[i] for i in df[df['split'] == 'train'].index]
valid_tokens = [tokens[i] for i in df[df['split'] == 'valid'].index]
test_tokens = [tokens[i] for i in df[df['split'] == 'test'].index]
n_train = len(train_tokens)
n_valid = len(valid_tokens)


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
        return (self.data[item], )

    def __len__(self):
        return len(self.data)

batch_size = args.batch_size
torch.manual_seed(0)
ds_train = ListDataset(train_tokens)
ells = [len(t) for t in train_tokens]
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=mlmcollate)
ds_valid = ListDataset(valid_tokens)
dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=mlmcollate)

# Getting pre-trained embedding
if args.pretrain or args.freeze:
    esm_embeddings = torch.load(data_fpath + 'esm1b_pfam_embs.pt')
    pfams, names = parse_fasta(data_fpath + 'final_pfams.fasta', return_names=True)
    names = [name.split(';')[-2] for name in names]
    new_idx = np.array([domains[name] for name in names]) - min(domain_tokens) - 1
    esm_embeddings = esm_embeddings[new_idx]
    n_frozen_embs = len(esm_embeddings)

# Load model
d_embedding = args.d_embedding
d_model = args.d_model
n_layers = args.n_layers
kernel_size = args.kernel_size
r = args.r
lr = args.lr
opt_level = 'O2'
epochs = args.epochs
torch.cuda.set_device(args.gpu)

device = torch.device('cuda:%d' %args.gpu)
if args.freeze:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                  padding_idx=mask_idx, causal=args.ar, final_ln=True, activation='gelu', n_frozen_embs=n_frozen_embs)
else:
    model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, slim=(not args.wide),
                  padding_idx=mask_idx, causal=args.ar, final_ln=True, activation='gelu')
optimizer = FusedAdam(model.parameters(), lr=lr)
if args.restart:
    checkpoints = os.listdir(args.out_fpath)
    last_epoch = -1
    for chkpt in checkpoints:
        if 'metrics' in chkpt:
            continue
        e = int(chkpt[10:-4])
        if e > last_epoch:
            last_epoch = e
    sd = torch.load(args.out_fpath + 'checkpoint%d.tar' %last_epoch, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
#    msd = {k.split('module.')[1]: v for k, v in msd.items()}
    model.load_state_dict(msd)
    optimizer.load_state_dict(sd['optimizer_state_dict'])
    optimizer.param_groups[0]['lr'] = 0.0
    initial_epoch = last_epoch + 1
    total_steps = sd['nsteps']
else:
    initial_epoch = 0
    total_steps = 0
    if args.freeze:
        with torch.no_grad():
            model.embedder.embedder.frozen.weight = torch.nn.Parameter(torch.tensor(esm_embeddings))
            torch.save({'model_state_dict': model.state_dict()}, args.out_fpath + 'esm_loaded.tar')
    if args.pretrain:
        with torch.no_grad():
            model.embedder.embedder.weight[len(specials) + 1:] = torch.nn.Parameter(torch.tensor(esm_embeddings))
            torch.save({'model_state_dict': model.state_dict()}, args.out_fpath + 'esm_loaded.tar')
model = model.to(device)
optimizer.state = {}
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
if args.restart:
    amp.load_state_dict(sd['amp_state_dict'])
loss_func = MaskedCrossEntropyLoss()
accu_func = MaskedAccuracy()


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
    chunk_time = datetime.now()
    n_seen = 0
    if train:
        n_total = len(ds_train)
    else:
        n_total = len(ds_valid)
    for i, batch in enumerate(loader):
        if train and i == 1 and e == initial_epoch and args.restart:
            optimizer.load_state_dict(sd['optimizer_state_dict'])
        new_loss, new_accu, new_n = step(model, batch, train)
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
        for _ in range(10):
            try:
                torch.save({
                    'step': nsteps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'amp_state_dict': amp.state_dict()
                }, args.out_fpath + 'checkpoint%d.tar' % e)
                break
            except OSError:
                time.sleep(1)
        else:
            print('Model checkpointing failed!')
    else:
        print('\nValidation complete in ' + str(datetime.now() - start_time))
        with open(args.out_fpath + 'metrics.csv', 'a') as f:
            f.write(','.join([str(rloss), str(raccu), str(int(e)), str(current_step)]))
            f.write('\n')
    return i


def step(model, batch, train):
    src, tgt, mask = batch
    src = src.to(device)
    tgt = tgt.to(device)
    mask = mask.to(device)
    input_mask = (src != padding_idx).float()
    outputs = model(src, input_mask=input_mask.unsqueeze(-1))
    loss = loss_func(outputs, tgt, mask)
    accu = accu_func(outputs, tgt, mask)
    if train:
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
    return loss.item(), accu.item(), mask.sum().item()

n_parameters = sum(p.numel() for p in model.parameters())
print('%d model parameters' % n_parameters)
print('%d training sequences' % n_train)
print('%d validation sequences' % n_valid)
for e in range(initial_epoch, epochs):
    total_steps += epoch(model, True, current_step=total_steps)
    with torch.no_grad():
        _ = epoch(model, False, current_step=total_steps)
    
