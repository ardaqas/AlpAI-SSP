# AlpAI-V5: Hybrid Deep Neural Network for Protein Secondary Structure Prediction

## Project Overview
AlpAI-V5 is a deep learning project designed for fast and accurate prediction of protein secondary structure (Q3: Helix - H, Beta strand - E, Coil - C) from sequence-derived features. The model leverages a hybrid architecture (CNN + BiLSTM + Transformer) to robustly capture local, contextual, and global sequence dependencies, aiming to facilitate research and applications in computational biology and bioinformatics.

## Development Timeline
| Version | Architecture & Key Advances |
|---------|-----------------------------|
| v1.0    | **Baseline CNN**: Prototype for local motif detection and sequence tagging. |
| v2.0    | **Deeper CNN + Positional Info**: Expanded CNN depth, introduced positional encoding for richer representations. |
| v3.0    | **CNN + BiLSTM**: Added BiLSTM to capture long-range context, boosting accuracy on global patterns (e.g., beta sheets). |
| v3.1    | **Class Imbalance Handling**: Weighted loss to mitigate label imbalance, addressing biases in coil/helix predictions. |
| v4.0    | **Pure Transformer**: Switched to an encoder-only Transformer, leveraging attention for global interactions. |
| v5.0    | **Hybrid (CNN + BiLSTM + Transformer)**: Combined all previous strengths; CNN for local motif extraction, BiLSTM for medium/long context, Transformer for explicit global modeling and robust final predictions. |

## Model Architecture (v5.0)
- **Input:** Protein sequence features (amino acid one-hot and/or HMM profiles; sequence padded to fixed length)
- **Embedding & Preprocessing:** Features are (optionally) embedded and concatenated
- **CNN Block:** Extracts local motif features using 1D convolutions
- **BiLSTM Block:** Captures medium- and long-range sequence context, enhancing modeling of spatial dependencies
- **Positional Encoding:** Adds explicit or learned positional information
- **Transformer Block:** Models global dependencies via multi-head self-attention and contextualizes all positions jointly
- **Output Layer:** Linear classifier maps representations to Q3 class probabilities for each residue

**Data flow:**

```
Input Features → CNN → BiLSTM → Positional Encoding → Transformer Encoder → Classifier
```

## Dataset & Preparation
- **Format:** Data split into `.npz` files (numpy archives) with fixed-length sequences and HMM/AA features
- **Structure:** Loaded via `NetSurfDataset` (see `dataset.py`). Expected as: `data/CB513_HHblits.npz`, `data/TS115_HHblits.npz`, `data/CASP12_HHblits.npz`, etc.
- **Features Used:** By default, HMM-derived features, optionally amino acid one-hot
- **Labels:** Q3 (Helix/H, Strand/E, Coil/C); auto-converted from Q8 if present
- **Padding/Masking:** All sequences padded to 1632 tokens; masking during training/evaluation to ignore padding or disordered regions

## Usage Instructions
Run training or evaluation in your terminal:

```bash
# Train the model
python train.py

# Evaluate the model and generate metrics/plots
python evaluate.py
```

**Outputs** (by default appear in `Resluts2/`):
- `best_model.pt` / `last_model.pt`: Model checkpoints
- `loss.png`, `metrics.png`: Training/validation loss and metric curves
- Per-dataset directory (`CASP12/`, `CB513/`, `TS115/`) containing:
  - `classification_report.txt`: Precision, recall, F1-score stats, support
  - `confusion_matrix.png`, `confusion_matrix_counts.png`, `confusion_matrix_normalized.png`: Visual analyses

## Results & Benchmarks (v5.0)
### Example Performance (Q3 Prediction)

| Dataset | Accuracy | Macro F1 | H-F1  | E-F1  | C-F1  |
|---------|----------|----------|-------|-------|-------|
| CB513   | 0.8151   | 0.8126   | 0.8570| 0.7823| 0.7986|
| TS115   | 0.8129   | 0.8064   | 0.8643| 0.7804| 0.7744|
| CASP12  | 0.7938   | 0.7937   | 0.8389| 0.7763| 0.7659|

*See `/Resluts2/{CASP12,CB513,TS115}/classification_report.txt` for full per-class stats and support.*

The model provides robust accuracy and consistent F1 across all classes and benchmarks, with especially strong helix recognition.

## Project Structure
```
AlpAI-V5/
├── dataset.py           # Dataset loading/preprocessing
├── model.py             # Model (v5.0 hybrid architecture)
├── train.py             # Training script
├── evaluate.py          # Evaluation and metrics generation
├── Resluts2/            # Results (see below)
│   ├── best_model.pt
│   ├── last_model.pt
│   ├── loss.png / metrics.png
│   ├── CASP12/
│   │   ├── classification_report.txt, confusion_matrix*.png
│   ├── CB513/
│   │   ├── classification_report.txt, confusion_matrix*.png
│   ├── TS115/
│       ├── classification_report.txt, confusion_matrix*.png
├── Results1/            # Legacy/older results
```

## Future Directions
- Extension to Q8 classification (8-class output)
- Enhanced attention mechanisms for broader context modeling
- Data augmentation and expanded datasets for improved robustness
- Deployment tooling (e.g., web server, batch inference scripts)
- Integration with protein structure/modeling pipelines

---
For questions or contributions, open an issue or pull request on GitHub.

