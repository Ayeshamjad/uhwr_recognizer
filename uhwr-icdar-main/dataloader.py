from typing import Optional
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import cv2 
from tacobox import Taco
import random

import chardet


class UrduNewsDataset(Dataset):
    def __init__(self, root, dataset_path: str, tokenizer, max_length: int = 512):
        self.root = root
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df: Optional[pd.DataFrame] = None
        self.load_dataset()

    def __len__(self):
        return len(self.df) if self.df is not None else 0

    def __getitem__(self, idx):
        if self.df is None:
            raise ValueError("Dataset not loaded.")

        text = str(self.df.iloc[idx]['News Text'])

        # Tokenize WITHOUT special tokens first
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,  # room for BOS + EOS
        )

        input_ids = tokenized["input_ids"]

        # Manually add BOS and EOS
        input_ids = (
            [self.tokenizer.bos_token_id] +
            input_ids +
            [self.tokenizer.eos_token_id]
        )

        # Create attention mask BEFORE padding
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Labels: ignore padding
        labels = [
            tok if tok != self.tokenizer.pad_token_id else -100
            for tok in input_ids
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    def load_dataset(self):
        """Load the dataset from the specified CSV file."""
        # self.df = pd.read_csv(os.path.join(
        #     self.root, self.dataset_path), encoding='UTF-8-SIG')

        # Step 1: Detect encoding
        with open(os.path.join(
                self.root, self.dataset_path), 'rb') as f:
            result = chardet.detect(f.read(100000))
            print("Detected encoding:", result['encoding'])
        detected_encoding = result['encoding']
        with open(os.path.join(
                self.root, self.dataset_path), encoding=detected_encoding, errors='ignore') as f:
            urdu = pd.read_csv(f)
            self.df = urdu


class HWRDataset(Dataset):
    def __init__(self, root, df, tokenizer, input_width=1600, 
                 input_height=64,
                 aug = False,
                 taco_aug_frac=0.9,
                 max_length: int = 512):
        self.root = root
        self.df = df
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10
        )
        self.aug=aug
        self.taco_aug_frac=taco_aug_frac

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
            
        image_path = os.path.join(self.root, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Unable to load image:", image_path)
            return None, None
            
        pixel_values = self.preprocess(image, self.aug)

        # Ensure tokenizer has a pad token (GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bos_id = self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Tokenize WITHOUT special tokens
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2  # reserve space for BOS + EOS
        )

        input_ids = tokenized["input_ids"]

        # Manually add BOS and EOS
        input_ids = [bos_id] + input_ids + [eos_id]

        # Attention mask before padding
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Labels: ignore padding in loss
        labels = [
            tok if tok != pad_id else -100
            for tok in input_ids
        ]

        encoding = (
            torch.tensor(pixel_values[None, :, :]).float(),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

        return encoding

    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)
            
        img = img/255.0
        img = img.swapaxes(-2,-1)[...,::-1]
        
        target = np.ones((self.input_width, self.input_height))
        
        new_x = self.input_width/img.shape[0]
        new_y = self.input_height/img.shape[1]
        min_xy = min(new_x, new_y)
        new_x = int(img.shape[0]*min_xy)
        new_y = int(img.shape[1]*min_xy)
        img2 = cv2.resize(img, (new_y,new_x))
        target[:new_x,:new_y] = img2
        return 1 - (target)

    def apply_taco_augmentations(self, input_img):
        random_value = random.random()
        if random_value <= self.taco_aug_frac:
            augmented_img = self.mytaco.apply_vertical_taco(
                input_img, 
                corruption_type='random'
            )
        else:
            augmented_img = input_img
        return augmented_img

def collate_fn(batch):
    src_batch, tgt_batch, attn_mask_batch = [], [], []
    for src_sample, tgt_sample, attn_mask_sample in batch:
        if src_sample is None or tgt_sample is None:
            continue
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        attn_mask_batch.append(attn_mask_sample)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    attn_mask_batch = torch.stack(attn_mask_batch)
    return {"pixel_values": src_batch, "labels": tgt_batch, "attention_mask": attn_mask_batch}