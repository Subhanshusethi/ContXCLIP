# ContXCLIP

This repository contains the implementation of **ContXCLIP**, a lightweight image captioning model leveraging a dual attention mechanism and a context preservation module (CPM). It builds upon CLIP and GPT-2 to enhance visual-textual alignment and contextual richness in generated captions.

Paper: [ContXCLIP: Contextual Attention for Vision-Language Understanding](https://github.com/Subhanshusethi/ContXCLIP)

---

## 📁 Project Structure

```
├── config.py         # Global configuration for model and training
├── dataset.py        # ClipCocoDataset class for loading and preprocessing data
├── data_split.py     # Splits dataset into train/test using random_split
├── model.py          # Custom activation functions (e.g., ReluSIG)
├── utils.py          # Attention modules, projection layers, utilities
├── main.py           # Defines the full ClipModel, evaluation, and inference logic
├── train.py          # Runs the training loop (used internally)
```

---

## 📄 File Responsibilities

### `config.py`
Defines the configuration class `CFG`:
- Learning rates, batch size, device
- Model architecture parameters (projection dims, transformer layers)
- Paths to pretrained GPT-2 or CLIP models

### `dataset.py`
Implements `ClipCocoDataset`, which:
- Loads pickled CLIP features and captions
- Tokenizes captions using GPT-2 tokenizer
- Applies stopword filtering, padding, and mask generation
- Returns tokenized caption, mask, and visual prefix for training

### `data_split.py`
- Instantiates the `ClipCocoDataset`
- Splits it into train/test subsets using `torch.utils.data.random_split`
- Helps create `train_data`, `test_data` datasets

### `utils.py`
Contains modular components used inside the main model:
- `ImageProjection`, `TextProjection`, `XGLAttentionSeq`, `EFFN`, `MLP`
- Utilities for averaging losses, plotting, etc.

### `model.py`
Contains non-standard activation like `ReluSIG`, optionally used in attention and projection layers.

### `main.py`
Defines the main model `ClipModel` which includes:
- Visual encoder interface with CLIP embedding
- Text encoder/decoder with GPT-2
- Dual Attention mechanism: XGLAttention + CAM
- Context Preservation Module (CPM) with BiLSTM + XGLA
- Evaluation logic including BLEU, CIDEr computation, inference, and visualization

### `train.py`
- Entry point for training (used as backend script)
- Imports dataset, model, and utils
- Initializes optimizer, learning scheduler, loss
- Performs training, validation, checkpoint saving

---

## 📊 Evaluation & Decoding
- Implements Top-k and Top-nσ decoding strategies to avoid repetition
- Evaluation done via BLEU, METEOR, SPICE, CIDEr

---

## 📦 Dataset
- MS COCO (Karpathy Split)
- Data must be preprocessed using CLIP and saved as `.pkl` with fields `clip_embedding` and `captions`

---

## 🧪 Implementation Details
- Uses CLIP (ViT-B/32) for image features and GPT-2 for text
- Dual Attention with XGLA + Logit-based CAM
- BiLSTM + XGLA form the CPM before decoding
- Trained with AdamW + ReduceLROnPlateau

---

## 🧠 Citation
If you use this codebase, please consider citing:

```bibtex
@article{ContXCLIP2024,
  title={ContXCLIP: Contextual Attention for Vision-Language Understanding},
  author={Subhanshu Sethi, Chhavi Dhiman},
  year={2024}
}
```
