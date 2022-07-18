from tqdm import tqdm

import torch
from esm.pretrained import load_model_and_alphabet

from sequence_models.utils import parse_fasta

encoder, alphabet = load_model_and_alphabet("/home/kevyan/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt")
batch_converter = alphabet.get_batch_converter()
encoder = encoder.eval()

data_fpath = '/home/kevyan/data/bgc/dedup/'
seqs, names = parse_fasta(data_fpath + 'final_pfams.fasta', return_names=True)

embs = torch.empty(len(seqs), 1280)
device = torch.device('cuda:1')
encoder = encoder.to(device)
with tqdm(total=len(seqs)) as pbar:
    for i, (s, n) in enumerate(zip(seqs, names)):
        data = [(n, s[:1022])]
        _, _, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = encoder(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        rep = results['representations'][33][0, 1:-1].mean(0).cpu()
        embs[i] = rep
        pbar.update(1)
torch.save(embs, data_fpath + 'esm1b_pfam_embs.pt')
