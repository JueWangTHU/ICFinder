import pickle
import torch
import esm
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
import os
import sys

# Model paths - modify these if needed
BLAPE_MODEL_PATH = 'Model/BLAPE-ICIdentification.pth'
CLAPE_MODEL_PATH = 'Model/CLAPE-ICPermeation.pth'


# ==================== ESM-2 MODEL LOADING ====================
def load_esm_model():
    """Load ESM-2 model for embedding generation"""
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet, batch_converter


# ==================== EMBEDDING GENERATION ====================
def generate_embedding(seq, pid, esm_model, batch_converter):
    """Generate embedding for a single sequence"""
    seq_data = [(pid, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)

    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]
    # Remove start and end tokens [CLS] and [SEP]
    embedding = token_representations.squeeze(0)[1:-1, :]
    return embedding.cpu()


# ==================== DATASET CLASS ====================
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
        return sample_id, seq, label


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    ids = []
    sequences = []
    lengths = []

    for sample_id, seq, label in batch:
        ids.append(sample_id)
        sequences.append(seq)
        lengths.append(seq.size(0))

    sequences_padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return ids, sequences_padded, lengths


# ==================== BLAPE MODEL ====================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_layers=1, bidirectional=True, dropout=0.3):
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
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_x)

        if self.num_directions == 1:
            last_hidden = h_n[-1]
        else:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)

        last_hidden = self.layer_norm(last_hidden)
        logits = self.fc(last_hidden).squeeze(1)
        return logits


# ==================== CLAPE MODEL ====================
class ContinueModel(nn.Module):
    def __init__(self):
        super(ContinueModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.layer5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)
        inter = x  # dim 1024
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.layer5(x)), inter


# ==================== MODEL LOADING FUNCTIONS ====================
def load_blape_model(model_path, device):
    """Load BLAPE model"""
    print(f"Loading BLAPE model from {model_path}...")
    model = LSTMClassifier()

    # Load state dict
    if device == 'cpu':
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)

    # Handle potential key differences
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_clape_model(model_path, device):
    """Load CLAPE model"""
    print(f"Loading CLAPE model from {model_path}...")

    model = ContinueModel()

    try:
        if device == 'cpu':
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print("CLAPE model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load CLAPE model properly: {e}")
        print("Please check the model path and checkpoint format.")

    model.to(device)
    model.eval()
    return model


# ==================== SEQUENCE READING ====================
def read_sequences(filepath):
    """Read sequences from FASTA-like format"""
    sequences = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            seq_id = lines[i].strip()[1:]  # Remove '>'
            if i + 1 < len(lines):
                seq = lines[i + 1].strip()
                sequences.append((seq_id, seq))
                i += 2
            else:
                i += 1
        else:
            i += 1

    return sequences


# ==================== PREDICTION FUNCTIONS ====================
def predict_blape(model, data_loader, device):
    """Make predictions using BLAPE model"""
    predictions = {}

    with torch.no_grad():
        for batch in data_loader:
            ids, sequences, lengths = batch
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            logits = model(sequences, lengths)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            for i, seq_id in enumerate(ids):
                predictions[seq_id] = {
                    'prob': probs[i].item(),
                    'pred': preds[i].item()
                }

    return predictions


def predict_clape(model, data_dict, device):
    """Make predictions using CLAPE model"""
    predictions = {}

    with torch.no_grad():
        for seq_id, (embedding, _) in data_dict.items():
            # embedding shape: (seq_len, 1280)
            embedding = embedding.to(device)

            # Process each residue individually
            residue_probs_list = []
            for i in range(embedding.size(0)):
                residue_embedding = embedding[i:i + 1, :]  # (1, 1280)
                probs, _ = model(residue_embedding)  # probs: (1, 2)
                # Take the probability of class 1 (positive class)
                prob_class1 = probs[0, 1].item()
                residue_probs_list.append(prob_class1)

            # Convert probabilities to binary predictions (threshold = 0.5)
            residue_preds = [1 if prob > 0.5 else 0 for prob in residue_probs_list]
            predictions[seq_id] = residue_preds

    return predictions


# ==================== MAIN PROCESSING FUNCTION ====================
def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_file> <output_file>")
        print("Example: python inference.py Sequence/example.txt result_example.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Processing sequences from: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using device: {device}")

    # Load ESM-2 model
    esm_model, alphabet, batch_converter = load_esm_model()

    # Read sequences
    print("Reading sequences...")
    sequences = read_sequences(input_file)
    print(f"Found {len(sequences)} sequences")

    # Generate embeddings
    print("Generating embeddings...")
    data_dict = {}
    for seq_id, seq in tqdm(sequences, desc="Generating embeddings"):
        embedding = generate_embedding(seq, seq_id, esm_model, batch_converter)
        data_dict[seq_id] = (embedding, 0)  # 0 is placeholder label

    # Create dataset and dataloader with batch_size=1
    dataset = SequenceDataset(data_dict)
    data_loader = DataLoader(dataset, batch_size=1,
                             collate_fn=collate_fn, shuffle=False)

    # Load models
    blape_model = load_blape_model(BLAPE_MODEL_PATH, device)
    clape_model = load_clape_model(CLAPE_MODEL_PATH, device)

    # Make predictions
    print("Making BLAPE predictions...")
    blape_predictions = predict_blape(blape_model, data_loader, device)

    print("Making CLAPE predictions...")
    clape_predictions = predict_clape(clape_model, data_dict, device)

    # Write results
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w') as f:
        for seq_id, seq in sequences:
            # Line 1: >ID
            f.write(f">{seq_id}\n")

            # Line 2: sequence
            f.write(f"{seq}\n")

            # Line 3: prob, pred
            blape_result = blape_predictions[seq_id]
            f.write(f"{blape_result['prob']:.6f}, {blape_result['pred']}\n")

            # Line 4: residue-level labels
            clape_result = clape_predictions[seq_id]
            labels_str = ''.join(map(str, clape_result))
            f.write(f"{labels_str}\n")

    print(f"Results saved to {output_file}")
    print("Processing complete!")


if __name__ == "__main__":
    main()
