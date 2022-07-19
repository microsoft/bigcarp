import argparse
import json
from datetime import datetime
from collections import Counter
import time
import random
from turtle import title

import torch
from Bio import SeqIO


# Imports for PCA/UMAP/t-SNE
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
plt.style.use('ggplot')

# Setting up parser
parser = argparse.ArgumentParser(description='Plotting embeddings')

# Defining required arguments for loading data, tokenization, saved model checkpoints, and clans data
parser.add_argument('token_dir', type=str, help='directory with domain vocabualary')
parser.add_argument('model_dir', type=str, help='directory with saved model checkpoint')
parser.add_argument('clans_dir', type=str, help='file with clan information from pfam')
parser.add_argument('results_dir', type=str, help='directory to save plots')
parser.add_argument('--freeze', action='store_true')

# Optional argument: defines whether pca, tsne, or umap gets called
parser.add_argument('--technique', type=str, default='pca', help='dimensionality reduction techniques such as pca, tsne and umap')
parser.add_argument('--model', type=str, default='0', help='model type or number')

args = parser.parse_args()

# Load data
with open(args.token_dir) as f:
    tokens = json.load(f)
specials = tokens['specials']
domains = tokens['domains']
n_tokens = tokens['size']

# Getting clan data
clans_data = pd.read_csv(args.clans_dir, sep='\t')
# print(clans_data)

# List of domains
domains_list = list(domains)
domains_list = domains_list[1:]
# print(len(domains_list))

# Creating clan list that maps to domain list
clans = [None] * len(domains_list)
palette_clan = {}
for _, row in clans_data.iterrows():
    if row['Pfam ID'] in domains_list:
        index = domains_list.index(row['Pfam ID'])
        clans[index] = row['clan ID']
counter = Counter(clans)
most_common = counter.most_common(12)
# print(most_common)

common_clans = []
for clan in most_common:
    common_clans.append(clan[0])
common_clans = common_clans[2:] # removes first two clans (NaN, None)

# Model's embedding tensor
checkpoint = torch.load(args.model_dir, map_location=torch.device('cpu')) ## load the model checkpoint
# print(checkpoint['model_state_dict'])

if args.freeze:
    embedding_layer = checkpoint['model_state_dict']['embedder.embedder.frozen.weight']
else:
    embedding_layer = checkpoint['model_state_dict']['embedder.embedder.weight']
embedding_layer = embedding_layer.cpu()

# Removing specials from embedding tensor
embedding = pd.DataFrame(embedding_layer).astype("float")
if args.freeze:
    embedding = embedding.iloc[: , :] # DataFrame ready for PCA
else:
    embedding = embedding.iloc[len(specials) + 1: , :] # DataFrame ready for PCA

# Removing domains that don't fall under a common clan
embedding_updated = embedding
embedding_updated['domain'] = domains_list
embedding_updated['clan'] = clans
new_embedding = embedding_updated[embedding_updated.clan.isin(common_clans)]
new_domains = new_embedding['domain'].values.tolist()
new_clans = new_embedding['clan'].values.tolist()
new_embedding = new_embedding.drop(['domain', 'clan'], axis=1)
# print(new_embedding)

# Performs PCA, t-SNE, or UMAP
if args.technique == 't-sne':
    embedding_pca = TSNE(n_components = 2).fit_transform(new_embedding)
    embed_df = pd.DataFrame(data=embedding_pca, columns=[str(args.technique) + '_1', str(args.technique) + '_2'])

if args.technique == 'umap':
    reducer = umap.UMAP(n_components = 2)
    scaled_embedding = StandardScaler().fit_transform(new_embedding)
    embedding_pca = reducer.fit_transform(scaled_embedding)
    embed_df = pd.DataFrame(data=embedding_pca, columns=[str(args.technique) + '_1', str(args.technique) + '_2'])

else:
    scaler = StandardScaler()
    scaler.fit(new_embedding)
    embedding_scaled = scaler.transform(new_embedding)
    pca = PCA(n_components = 2, random_state=2021)
    pca.fit(embedding_scaled)
    embedding_pca = pca.transform(embedding_scaled)
    embed_df = pd.DataFrame(data=embedding_pca, columns=[str(args.technique) + '_1', str(args.technique) + '_2'])

# add columns "domain" and "clan" to embed_df
embed_df['domain'] = new_domains
embed_df['clan'] = new_clans
# print(embed_df)

# Visualize with scatterplot
plt.figure(figsize=(10, 10))
sns.set_style("white")

scatter = sns.scatterplot(x=embed_df[str(args.technique) + '_1'], y=embed_df[str(args.technique) + '_2'], hue=embed_df['clan'], palette='Paired')

plt.xticks([]),plt.yticks([])
# plt.legend(title="Pfam clan", fontsize='23', title_fontsize='23', ncol=1)
plt.savefig(str(args.results_dir) +  str(args.technique) + "_" + str(args.model) + ".png",
           dpi=300,
           bbox_inches='tight')