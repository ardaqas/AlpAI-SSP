# AlpAI-SSP  
CNN-BiLSTM based protein secondary structure prediction model

### Project Overview

AlpAI-SSP is a deep learning-based system designed to predict protein secondary structures — specifically, Helix (H), Beta Sheet (E), and Coil (C) — from amino acid sequences. The goal of this project is to assist in computational biology and bioinformatics by providing accurate predictions that can complement or reduce the need for experimental methods such as X-ray crystallography or NMR spectroscopy.

### Motivation

Understanding the secondary structure of proteins is a critical step toward deciphering their function. While experimental determination is accurate, it is often expensive and time-consuming. AlpAI leverages machine learning to offer a faster, scalable alternative for large-scale protein analysis.

### Development Timeline

- **v1.0**: Initial implementation using a basic CNN architecture for sequence tagging.  
- **v2.0**: Integration of positional embeddings and deeper convolutional layers for richer feature extraction.  
- **v3.0**: Introduction of a Bidirectional LSTM (BiLSTM) to capture long-range dependencies and improve prediction of beta sheets.  
- **v3.1**: Added class-balanced loss weighting to address label imbalance.  

### Model Architecture

- **Input**: Protein sequence as integer-encoded amino acid tokens.  
- **Embedding Layer**: Learned amino acid embeddings with positional encoding.  
- **Convolutional Layers**: Multi-scale CNN to capture local structural motifs.  
- **Recurrent Layer**: Bidirectional LSTM for sequence-level context.  
- **Output Layer**: Fully connected layer with softmax activation for class probabilities.  

### Dataset

- Format: CSV files containing protein sequences and corresponding secondary structure labels (H, E, C).  
- Maximum sequence length: 512  
- Padding and masking applied where necessary.  

#### Dataset Download

All required `.csv` files are available here:  
📥 [Download via Google Drive](https://drive.google.com/drive/u/0/folders/1Z7HZPMZOcR_hPhD722Hr4U_xmeUX29aY)

Place the downloaded files inside the `data/` directory.

### Performance

| Class                | Precision | Recall | F1-score |
|----------------------|-----------|--------|----------|
| H                    | 0.66      | 0.85   | 0.75     |
| E                    | 0.44      | 0.86   | 0.58     |
| C                    | 0.99      | 0.83   | 0.90     |
| **Overall Accuracy** |           |        | **0.83** |

*Classification metrics computed on the validation set.*

### Usage

```bash
# Training
python train.py

# Evaluation and visualization
python evaluate.py
```

This script evaluates the model on validation and benchmark datasets, and generates all metrics and plots under the `results/` directory.

```bash
# Inference
python predict.py --sequence "ACDEFGHIKLMNPQRSTVWY"
```

### Repository Structure

```
AlpAI-SSP/
├── README.md
├── requirements.txt
├── model.py
├── dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── model_final.pt
├── data/
├── results/
│   ├── validation_report.txt
│   ├── cb513_report.txt
│   ├── ts115_report.txt
│   ├── casp12_report.txt
│   ├── val_confusion_matrix.png
│   ├── loss_curve.png
│   ├── val_predictions.npy
│   └── val_labels.npy
```

### Future Directions

* Integration of attention mechanisms or transformers for better long-range pattern recognition.
* Experimentation with CRF or structured prediction techniques for output refinement.
* Evaluation on external benchmark datasets and generalization studies.
* Transition from 3-class (Q3: H, E, C) to 8-class (Q8: H, G, I, E, B, T, S, C) secondary structure prediction for more granular modeling.
