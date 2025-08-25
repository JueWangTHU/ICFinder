import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
import esm
import random
from tqdm import tqdm
from datetime import datetime

TOP_PERCENTS = [0.05, 0.1, 0.15, 0.2]
DATASET_NAME = "valid_ICPermeation"
NUM_BG = 5
MODEL_PATH = "train_log/model_2025-08-24_23-15-26.pth"

os.makedirs("IG_log", exist_ok=True)
os.makedirs("IG_log/predict", exist_ok=True)


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(self.log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


log_filename = f'IG_log/{DATASET_NAME}_bg{NUM_BG}.log'
logger = Logger(log_filename)
sys.stdout = logger

print(f"\n{'=' * 50}")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: {DATASET_NAME}")
print(f"TOP k: {TOP_PERCENTS}")
print(f"Number of background sequences: {NUM_BG}")
print(f"{'=' * 50}")


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
        # x: (B, L, 1280), lengths: (B,)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_x)
        if self.num_directions == 1:
            last_hidden = h_n[-1]
        else:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        last_hidden = self.layer_norm(last_hidden)
        logits = self.fc(last_hidden).squeeze(1)  # (B,)
        return logits


amino_acids = 'ACDEFGHIKLMNPQRSTVWY'


def generate_random_sequence(length):
    return ''.join(random.choice(amino_acids) for _ in range(length))


def encode_sequence_with_esm2(sequence, model, batch_converter):
    seq_data = [("random_seq", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    return token_representations.squeeze(0)[1:-1, :]  # (L, 1280)


def generate_background(ig_model, esm2_model, batch_converter, seq_length, num_bg=NUM_BG):
    bg_attrs = []
    print(f"Generating background sequences and computing IG (Length: {seq_length})...")
    for _ in tqdm(range(num_bg)):
        random_sequence = generate_random_sequence(seq_length)
        rand_feat = encode_sequence_with_esm2(random_sequence, esm2_model, batch_converter)  # (L,1280)
        rand_feat = rand_feat.unsqueeze(0)  # (1,L,1280)
        rand_feat.requires_grad = True
        lengths = torch.tensor([seq_length], dtype=torch.long)
        bg_attr, _ = ig_model.attribute(
            rand_feat,
            target=None,
            baselines=torch.zeros_like(rand_feat),
            return_convergence_delta=True,
            additional_forward_args=(lengths,)
        )
        bg_attr = bg_attr.squeeze(0).sum(dim=-1).abs().detach().cpu().numpy()  # (L,)
        bg_attrs.append(bg_attr)
    bg_mean = np.mean(np.stack(bg_attrs, axis=0), axis=0)  # (L,)
    return bg_mean


def read_sequences(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            if i + 1 < len(lines):
                pid = lines[i].strip()[1:]
                seq = lines[i + 1].strip()
                sequences[pid] = seq
    return sequences


def generate_predictions(residual_bg_normalized, seq_length, top_percent):
    top_k = max(1, int(seq_length * top_percent))
    top_indices = np.argsort(np.abs(residual_bg_normalized))[-top_k:]
    pred = np.zeros(seq_length, dtype=int)
    pred[top_indices] = 1
    return pred


def save_single_prediction(pid, predictions_for_pid, sequence, top_percents):
    for top_percent in top_percents:
        filename = f"IG_log/predict/{DATASET_NAME}_bg{NUM_BG}_IG_{top_percent:.2f}.txt"
        pred = predictions_for_pid[top_percent]

        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f">{pid}\n")
            f.write(f"{sequence}\n")
            pred_str = ''.join(str(p) for p in pred)
            f.write(f"{pred_str}\n")


def initialize_prediction_files(top_percents):
    for top_percent in top_percents:
        filename = f"IG_log/predict/{DATASET_NAME}_bg{NUM_BG}_IG_{top_percent:.2f}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            pass
        print(f"Initializing: {filename}")


model = LSTMClassifier(1280, 256, 1, bidirectional=True, dropout=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

print("Loading ESM-2...")
esm2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm2_model.eval()

pkl_file = f"../Embedding/esm_{DATASET_NAME}.pkl"
sequence_file = f"../Sequence/{DATASET_NAME}.txt"

print(f"Embedding: {pkl_file}")
print(f"Sequence: {sequence_file}")

with open(pkl_file, "rb") as f:
    data = pickle.load(f)

sequences = read_sequences(sequence_file)

print(f"Loaded embeddings for {len(data)} sequences")
print(f"Loaded sequence information for {len(sequences)} sequences")

initialize_prediction_files(TOP_PERCENTS)

ig = IntegratedGradients(model)
background_cache = {}  # key: seq_length -> bg_mean
print(f"\nStarting to process {len(data)} sequences...")
processed_count = 0

for pid in tqdm(list(data.keys())):
    try:
        feature, _ = data[pid]  # feature: (L, 1280)
        seq_length = feature.shape[0]
        if seq_length == 0:
            print(f"\nSequence {pid} has length 0, skipping")
            continue

        # to tensor
        feature_t = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # (1,L,1280)
        lengths = torch.tensor([seq_length], dtype=torch.long)
        feature_t.requires_grad = True

        # compute IG (relative to zero baseline)
        baseline = torch.zeros_like(feature_t)
        attr, delta = ig.attribute(
            feature_t,
            target=None,
            baselines=baseline,
            return_convergence_delta=True,
            additional_forward_args=(lengths,)
        )
        # attribution strength for each position (sum across channels)
        attr = attr.squeeze(0).sum(dim=-1).detach().cpu().numpy()  # (L,)

        # background mean (cached by length)
        if seq_length not in background_cache:
            print(f"\nGenerating background for length {seq_length}...")
            background_cache[seq_length] = generate_background(
                ig, esm2_model, batch_converter, seq_length, num_bg=NUM_BG
            )
        bg_mean = background_cache[seq_length]

        # background normalization, avoid division by zero
        residual_bg_normalized = attr / (bg_mean + 1e-30)

        # generate predictions for each TOP_PERCENT
        predictions_for_current_seq = {}
        for top_percent in TOP_PERCENTS:
            pred = generate_predictions(residual_bg_normalized, seq_length, top_percent)
            predictions_for_current_seq[top_percent] = pred

        # get sequence information
        sequence = sequences.get(pid, "UNKNOWN_SEQUENCE")

        # save current sequence predictions in real-time
        save_single_prediction(pid, predictions_for_current_seq, sequence, TOP_PERCENTS)

        processed_count += 1

        # show processing progress
        print(f"\nSequence {pid} ({processed_count}/{len(data)}): length={seq_length} [saved]")
        for top_percent in TOP_PERCENTS:
            pred = predictions_for_current_seq[top_percent]
            predicted_sites = int(pred.sum())
            print(
                f"  TOP_{top_percent:.2f}: predicted {predicted_sites} important sites ({predicted_sites / seq_length:.3f})")

    except Exception as e:
        print(f"Error processing sequence {pid}: {e}")
        continue

print(f"\nSuccessfully processed {processed_count} sequences")

# =========================================================
# Overall summary
# =========================================================
print("\n" + "=" * 70)
print(f"Overall summary for dataset {DATASET_NAME}")
print("=" * 70)

for top_percent in TOP_PERCENTS:
    print(f"\nTOP_{top_percent:.2f} results:")
    print(f"Prediction file: IG_log/predict/{DATASET_NAME}_bg{NUM_BG}_IG_{top_percent:.2f}.txt")

# overall statistics
print(f"\n=== Overall Statistics ===")
print(f"Dataset name: {DATASET_NAME}")
print(f"Total sequences processed: {processed_count}")
print(f"Number of background sequences: {NUM_BG}")
print(f"Generated background length types: {len(background_cache)}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# close logger
sys.stdout = logger.terminal
logger.close()
print(f"Results saved to {log_filename}")
