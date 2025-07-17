# inference.py
import torch
from model import CNNSecondaryStructureV2
from dataset import aa_to_int, structure_to_int, MAX_LEN
from config import DEVICE

int_to_structure = {v: k for k, v in structure_to_int.items()}

def encode_seq(seq):
    seq = seq[:MAX_LEN]
    encoded = [aa_to_int.get(aa, 0) for aa in seq]
    padded = encoded + [0] * (MAX_LEN - len(encoded))
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0)  # [1, L]

def predict_structure(seq, model):
    model.eval()
    with torch.no_grad():
        x = encode_seq(seq).to(DEVICE)
        out = model(x)                     # [1, L, 3]
        preds = out.argmax(dim=2).squeeze(0)  # [L]
        pred_labels = [int_to_structure[int(i)] for i in preds[:len(seq)]]
        return pred_labels

if __name__ == "__main__":
    model = CNNSecondaryStructureV2().to(DEVICE)
    checkpoint = torch.load("model_final.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Paste your amino acid sequence (ACDE...):")
    seq = input().strip().upper()
    pred_structs = predict_structure(seq, model)

    # Print results nicely
    seq_line = 'Sequence:      ' + ' '.join(seq)
    struct_line = 'Structure:     ' + ' '.join(pred_structs)
    print("\n" + seq_line)
    print(struct_line)
