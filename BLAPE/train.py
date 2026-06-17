# -*- coding: utf-8 -*-
# @Time         : 2025/8/24 18:00
# @Author       : Jue Wang
# @Description  : Training

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score
import logging
from datetime import datetime
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset loading
class SequenceDataset(Dataset):
    def __init__(self, data_dict):
        self.ids = list(data_dict.keys())
        self.data = list(data_dict.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        sample_id = self.ids[idx]
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)
        label = int(label)
        return sample_id, seq, label

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def collate_fn(batch):
    ids = []
    sequences = []
    labels = []
    lengths = []
    for sample_id, seq, label in batch:
        ids.append(sample_id)
        sequences.append(seq)
        lengths.append(seq.size(0))
        labels.append(label)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return ids, sequences_padded, lengths, labels

# model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_layers=1, bidirectional=False, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x, lengths):
        """
        x: shape (batch, seq_len, input_size)
        lengths: 1D tensor of lengths, shape (batch,)
        """
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_x)
        if self.num_directions == 1:
            last_hidden = h_n[-1]  # shape: (batch, hidden_size)
        else:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # shape: (batch, hidden_size*2)
        last_hidden = self.layer_norm(last_hidden)
        logits = self.fc(last_hidden).squeeze(1)  # shape: (batch,)

        return logits


# Train and valid
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        _, sequences, lengths, labels = batch
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * sequences.size(0)
    return epoch_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_ids = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            ids, sequences, lengths, labels = batch
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(sequences, lengths)
            all_ids.extend(ids)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)
    return all_ids, all_labels, preds, probs


def compute_metrics(y_true, y_pred, y_probs):
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auroc = float('nan')
    return mcc, precision, recall, auroc


# log
def setup_logger():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = './train_log/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, f'{current_time}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )

    return logging.getLogger()


def main():
    logger = setup_logger()

    params = {
        "train": '../Embedding/esm_train_ICIdentification.pkl',
        "valid": '../Embedding/esm_valid_ICIdentification.pkl',
        "test": '../Embedding/esm_test_ICIdentification.pkl',
        "input_size": 1280,
        "random_seed": 42,
        "hidden_size": 256, # 1024
        "num_layers": 1,
        "batch_size": 64,
        "num_epochs": 2,
        "learning_rate": 5*1e-5,
        "dropout": 0.3,
        "bidirectional": True,
    }

    logger.info("Model Hyperparameters:")
    for param, value in params.items():
        logger.info(f"{param}: {value}")

    set_seed(params['random_seed'])

    train_pkl = params['train']
    val_pkl = params['valid']
    test_pkl = params['test']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = load_pkl(train_pkl)
    val_data = load_pkl(val_pkl)
    test_data = load_pkl(test_pkl)

    train_dataset = SequenceDataset(train_data)
    val_dataset = SequenceDataset(val_data)
    test_dataset = SequenceDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LSTMClassifier(input_size=params['input_size'], hidden_size=params['hidden_size'], num_layers=params['num_layers'], bidirectional=params['bidirectional'], dropout=params['dropout'])
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_mcc = 0.0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        ids_val, y_val, preds_val, probs_val = evaluate(model, val_loader, device)
        mcc, precision, recall, auroc = compute_metrics(y_val, preds_val, probs_val)

        logger.info(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val MCC = {mcc:.4f} | Precision = {precision:.4f} | Recall = {recall:.4f} | AUROC = {auroc:.4f}")

        if mcc > best_val_mcc:
            best_val_mcc = mcc
            best_model_state = model.state_dict()

    if best_model_state is not None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join('./train_log', f'model_{current_time}.pth')
        torch.save(best_model_state, model_path)
        logger.info(f"Best model saved at {model_path}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    ids_test, y_test, preds_test, probs_test = evaluate(model, test_loader, device)
    mcc, precision, recall, auroc = compute_metrics(y_test, preds_test, probs_test)
    logger.info("\nTest Metrics:")
    logger.info(f"MCC = {mcc:.4f}")
    logger.info(f"Precision = {precision:.4f}")
    logger.info(f"Recall = {recall:.4f}")
    logger.info(f"AUROC = {auroc:.4f}")

    logger.info("\nDetailed Test Predictions:")
    for idx, (true_label, pred, prob) in enumerate(zip(y_test, preds_test, probs_test)):
        logger.info(f"Sample {idx}: Predicted Label = {pred}, Probability = {prob:.4f}, True Label = {true_label}")


if __name__ == "__main__":
    main()
