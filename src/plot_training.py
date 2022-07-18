import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

_ = sns.set_style('white')

dfs = []
for emb, name in zip(['naive', 'finetune', 'freeze'], ['random', 'ESM-1b', 'ESM-1b-freeze']):
    df = pd.read_csv('/home/kevyan/amlt/bgc-mlm/' + emb + '/metrics.csv', header=None)
    df.columns = ['loss', 'accuracy', 'epoch', 'steps']
    df['embedding'] = name
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
_ = sns.lineplot(data=df, x='epoch', y='loss', hue='embedding')
_ = plt.savefig('/home/kevyan/src/bgc/results/retrain-mlm.png', bbox_inches='tight')
df.groupby('embedding')['loss'].min()

dfs = []
for emb, name in zip(['naive', 'finetune', 'freeze'], ['random', 'ESM-1b', 'ESM-1b-freeze']):
    df = pd.read_csv('/home/kevyan/amlt/bgc-eval/test-' + emb + '/test.csv', header=None)
    df.columns = ['dloss', 'daccu', 'closs', 'caccu']
    df['embedding'] = name
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
grouped = df.groupby('embedding')
np.exp(grouped[['dloss', 'daccu', 'closs', 'caccu']].mean())
grouped[['dloss', 'daccu', 'closs', 'caccu']].std()




# a = np.array([[0, 0.8511, 0.17244093220528048],
# [1, 0.9012, 0.15912887823955615],
# [2, 0.8482, 0.1418242403602535]])
# a.mean(axis=0)
# a.std(axis=0)
