# Unified Architecture for Urdu Printed and Handwritten Text Recognition

This repository implements a unified deep learning architecture for Urdu printed and handwritten text recognition. The model combines a custom CNN block and Transformer encoder for image feature extraction, with a pre-trained Transformer decoder for Urdu language modeling.

For detailed methodology, see the paper:  
**[A Unified Architecture for Urdu Printed and Handwritten Text Recognition](https://dl.acm.org/doi/10.1007/978-3-031-41685-9_8)**

## Features

- Unified model for both printed and handwritten Urdu text
- Custom CNN + Transformer encoder for robust image understanding
- Pre-trained Transformer decoder for language modeling
- Modular codebase for easy experimentation

## Repository Structure

- `vocabs/ved/`: Urdu vocabulary files
- `model/`: Model components (CNN, Transformer, etc.)
- `utils/`: Utility scripts (dataset loading, decoding, losses)
- `training_artifacts/`: Training logs and artifacts
- `models.py`: Model implementation

## Requirements

- Python 3.11
- PyTorch
- Transformers
- pandas
- tqdm
- pillow
- opencv-python
- taco-box

Install dependencies with:

```bash
pip install torch transformers pandas tqdm pillow opencv-python taco-box evaluate
```

## Usage

1. Place your datasets in the appropriate directories.
2. Ensure vocabulary files are in `vocabs/ved/`.
3. Pretrain the Transformer decoder (language model) using `Pretraining.ipynb` (see below).
4. Train the full model for printed or handwritten Urdu text using `train_upti.py` or `train_uhwr.py` (see below).

---

## Pretraining the Transformer Decoder

The Urdu language model (Transformer decoder) is pretrained on a large Urdu news dataset using `Pretraining.ipynb`:

- Loads and tokenizes Urdu text data (see `dataloader.py` and `tokeniser.py`).
- Trains a decoder-only Transformer model using HuggingFace's `Trainer` API.
- Stores checkpoints in `decoder_pretrain_tokenizer_bos_eos/`.
- This pretrained decoder is later used for initializing/fine-tuning the full recognition model.

**To run:**

1. Set `ROOT` and `DATA_ROOT` paths in the notebook.
2. Ensure the Urdu news dataset CSV is available.
3. Run all cells to pretrain and save the decoder checkpoints.

---

## Training for Printed Text: `train_upti.py`

This script trains the unified model for Urdu printed text recognition (UPTI dataset):

- Loads printed text images and labels from the UPTI dataset CSVs.
- Uses a CNN + Transformer encoder for image features, and the pretrained Transformer decoder for language modeling.
- Supports CTC and cross-entropy loss, with early stopping and loss curve plotting.
- Saves the best model as `best_model_upti_icdar.pt`.

**To run:**

1. Set `ROOT` and `DATA_ROOT` in the script.
2. Ensure UPTI dataset CSVs are present in the data directory.
3. Run the script:
   ```bash
   python train_upti.py
   ```

---

## Training for Handwritten Text: `train_uhwr.py`

This script trains the unified model for Urdu handwritten text recognition (UHWR dataset), and can also combine printed and handwritten data:

- Loads handwritten (and optionally printed) text images and labels from CSVs.
- Uses the same model architecture as above.
- Supports dynamic loss weighting, early stopping, and loss curve plotting.
- Saves the best model as `best_model_uhwr_icdar.pt`.

**To run:**

1. Set `ROOT` and `DATA_ROOT` in the script.
2. Ensure UHWR and (optionally) UPTI dataset CSVs are present.
3. Run the script:
   ```bash
   python train_uhwr.py
   ```

---

## Citation

If you use this repository, please cite the original paper.
