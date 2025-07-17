import torch

# Data
MAX_LEN = 552
NUM_CLASSES = 3
AMINO_ACID_VOCAB_SIZE = 21  # 20 aa + 1 padding

# Training
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-3

# Embedding
EMBEDDING_DIM = 64

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TRAIN_CSV = "data/training_secondary_structure_train.csv"
VALID_CSV = "data/validation_secondary_structure_valid.csv"
TEST_CB513 = "data/test_secondary_structure_cb513.csv"
TEST_TS115 = "data/test_secondary_structure_ts115.csv"
TEST_CASP12 = "data/test_secondary_structure_casp12.csv"
