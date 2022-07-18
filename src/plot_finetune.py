import pathlib
import json

import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.set_style('white')

base_path = '/home/kevyan/amlt/bgc-sup/'
finetune = ['True', 'False']
naive = ['True', 'False']
init = ['finetune', 'naive', 'freeze']
df = []
for f in finetune:
    for n in naive:
        for ini in init:
            m = pd.read_csv(base_path + 'sup-%s-%s-%s/metrics.csv' %(f, n, ini), header=None)
            m.columns = ['train_loss', 'train_accu', 'valid_loss', 'valid_accu', 'valid_auc']
            m['epoch'] = np.arange(len(m))
            m['finetune'] = f
            if n == 'True':
                m['init'] = 'random'
            else:
                m['init'] = ini
            df.append(m)
df = pd.concat(df)

fig, ax = plt.subplots(1, 1)
_ = sns.lineplot(data=df, y='valid_auc', x='epoch', hue='finetune', style='init', ax=ax)
_ = fig.savefig('/home/kevyan/src/bgc/results/supervised.jpg', bbox_inches='tight')

grouped = df.groupby(['init', 'finetune'])['valid_auc'].max()

results = []
for ini in init:
    results.append(softmax(np.load('/home/kevyan/src/bgc/results/%s/mibig.npz' %ini)['first'], axis=-1))
results = np.array(results)
results = results.mean(axis=0)

home = str(pathlib.Path.home())
data_fpath = home + '/data/bgc/'
val_fpath = data_fpath + 'output/MiBIG_1406_dataset.txt'
df = pd.read_csv(val_fpath, header=None)
df.columns = ['name', 'activity', 'domains']
data_fpath += 'dedup/'
with open(data_fpath + 'final_domain_vocab.json') as f:
    tokens = json.load(f)
specials = tokens['specials']
domains = tokens['domains']
domain_tokens = np.array([domains[d] for d in domains])
n_tokens = tokens['size']
padding_idx = specials['-']
mask_idx = specials['#']

classes = ['Alkaloid', 'Saccharide', 'NRP', 'Terpene', 'Polyketide', 'Other', 'RiPP']
class_to_col = {c:i for i, c in enumerate(classes)}
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
predictions = np.zeros((len(results), len(classes)))
for t, c in token_to_columns.items():
    predictions[:, c] += results[:, [t]]
tgts = np.zeros((len(results), len(classes)))
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