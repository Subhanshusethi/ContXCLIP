import os
import re
import sys
import pickle
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from nltk.corpus import stopwords
from typing import Tuple
class ClipCocoDataset(Dataset):
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.stop_words = set(stopwords.words('english'))
        # Load data
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]

        # Tokenized captions file path
        tokens_file_path = f"/kaggle/working/{os.path.basename(data_path).split('.')[0]}_{gpt2_type}_tokensv2.pkl"

        if os.path.isfile(tokens_file_path):
            print("Loading tokenized captions from pickle file...")
            with open(tokens_file_path, 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            print("Tokenizing captions and saving to pickle file...")
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0

            for caption in captions_raw:
                processed_caption = self.preprocess_caption(caption['caption'])
                tokens = torch.tensor(self.tokenizer.encode(processed_caption), dtype=torch.int64)
                self.captions_tokens.append(tokens)
                self.caption2embedding.append(caption["clip_embedding"])  # Should be an index
                max_seq_len = max(max_seq_len, tokens.shape[0])

            # Save tokenized captions to pickle file
            with open(tokens_file_path, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)

        # Validate indices
        valid_indices = [i for i, idx in enumerate(self.caption2embedding) if idx < len(self.prefixes)]
        if len(valid_indices) < len(self.caption2embedding):
            print(f"Found {len(self.caption2embedding) - len(valid_indices)} invalid indices. Filtering out invalid captions.")
            self.captions_tokens = [self.captions_tokens[i] for i in valid_indices]
            self.caption2embedding = [self.caption2embedding[i] for i in valid_indices]
            self.image_ids = [self.image_ids[i] for i in valid_indices]
            self.captions = [self.captions[i] for i in valid_indices]

        # Compute max sequence length based on tokenized data
        all_len = torch.tensor([len(tokens) for tokens in self.captions_tokens]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    def preprocess_caption(self, caption: str) -> str:
        """
        Preprocesses the caption by normalizing case, removing special characters,
        redundant white spaces, and stopwords.
        """
        # Convert to lowercase
        caption = caption.lower()
        # Remove special characters
        caption = re.sub(r"[^\w\s]", "", caption)  # Retain only letters, digits, and spaces
        caption = re.sub(r"[.,@#$%^&*()<>?/|{}~:;\"']", "", caption)
        # Remove redundant white spaces
        caption = re.sub(r"\s+", " ", caption).strip()
        
        # Remove stopwords
        words = caption.split()
        caption = " ".join([word for word in words if word not in self.stop_words])
        return caption
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # Mask is zero where we are out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # Adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]  # Use index to get embedding
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix


