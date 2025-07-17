import torch
from torch.utils.data import Dataset
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}
aa_to_int['X'] = 0

structure_to_int = {'H': 0, 'E': 1, 'C': 2}
MAX_LEN = 552

class ProteinDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        original_len = len(df)
        filtered = df[df['seq'].str.len() <= MAX_LEN]        
        print(f"Filtered out {original_len - len(filtered)} sequences longer than {MAX_LEN}.")
 

        # Filter out sequences longer than MAX_LEN
        filtered = df[df['seq'].str.len() <= MAX_LEN]
        
        self.seqs = filtered['seq'].tolist()
        self.labels = filtered['sst3'].tolist()

    def __len__(self):
        return len(self.seqs)

    def encode_seq(self, seq):
        encoded = [aa_to_int.get(aa, 0) for aa in seq]
        padded = encoded + [0] * (MAX_LEN - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    def encode_sst3(self, sst3):
        encoded = [structure_to_int.get(s, 2) for s in sst3]
        padded = encoded + [2] * (MAX_LEN - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    def __getitem__(self, idx):
        x = self.encode_seq(self.seqs[idx])
        y = self.encode_sst3(self.labels[idx])
        return x, y

