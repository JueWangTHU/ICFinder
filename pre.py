# -*- coding: utf-8 -*-
# @Time         : 2025/8/24 18:00
# @Author       : Jue Wang
# @Description  : Generate embeddings from sequences by ESM-2

import pickle
import torch
import esm
from tqdm import tqdm

# model loading
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Embedding generating
Name = 'train_ICPermeation'
data_dict = {}
data = open("./Sequence/" + Name + ".txt", 'r').readlines()
for i in tqdm(range(len(data))):
    if data[i].startswith('>'):
        pid = data[i].strip()[1:]
        seq = [(pid, data[i+1].strip())]
        batch_labels, batch_strs, batch_tokens = batch_converter(seq)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        data_dict[pid] = (token_representations.squeeze(0)[1:-1, :], data[i+2].strip())

# saving
pickle.dump(data_dict, open("./Embedding/esm_" + Name + ".pkl", 'wb'))
