import torch
from decoder_models import TransformerDecoder
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import WIKI_dataset
from torch.utils.data import DataLoader
from importlib import reload
reload(WIKI_dataset)
from WIKI_dataset import WikipediaDataset, CustomDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
(train_data, test_data), tokenizer, VOCAB_SIZE = WikipediaDataset()
D_MODEL = 800
N_HEADS = 10
N_BLOCKS = 6
# BATCH = len(train_data)//1000
BATCH = 80
model = TransformerDecoder(VOCAB_SIZE, D_MODEL, N_HEADS, N_BLOCKS).to(DEVICE)
train_dataset = CustomDataset(train_data, tokenizer)
test_dataset = CustomDataset(test_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle = False)

loss_fnc = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
loss_epochs = []
for epoch in tqdm(range(5)):
    loss_epoch = []
    for input_ids, labels_ids, data_attention_mask in tqdm(train_loader):
        input_ids = input_ids.to(DEVICE)
        labels_ids = labels_ids.to(DEVICE)
        # pad_mask = pad_mask.to(DEVICE)
        data_attention_mask = data_attention_mask.to(DEVICE)

        preds = model(input_ids)
        preds = preds[data_attention_mask]
        labels_ids = labels_ids[data_attention_mask]

        optimizer.zero_grad()
        loss = loss_fnc(preds, labels_ids)
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.item())

    print(f'Loss_epoch {epoch + 1}: {np.mean(loss_epoch)}')
    loss_epochs.append(np.mean(loss_epoch))

torch.save(model.state_dict(), './model_alotofparameters.pt')