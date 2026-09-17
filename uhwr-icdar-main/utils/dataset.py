import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from tacobox import Taco
import random


class HWRDataset(Dataset):
    def __init__(
        self,
        root,
        df,
        tokenizer,
        input_width=1600,
        input_height=64,
        aug=False,
        taco_aug_frac=0.9,
        max_length: int = 512,
    ):
        self.root = root
        self.df = df.reset_index(drop=True)
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
            min_tw_horizontal=10,
        )

        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

        # Ensure PAD / BOS exist (GPT-2 safety)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]

        image_path = os.path.join(self.root, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # ---- hard guard: NEVER return None ----
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image: {image_path}")

        pixel_values = self.preprocess(image, self.aug)

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Tokenize WITHOUT special tokens
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        )

        input_ids = tokenized["input_ids"]

        # Add BOS / EOS
        input_ids = [bos_id] + input_ids + [eos_id]

        # Attention mask before padding
        attention_mask = [1] * len(input_ids)

        # Pad / truncate
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Labels: PAD → -100 (for CE)
        labels = [
            tok if tok != pad_id else -100
            for tok in input_ids
        ]

        return (
            torch.tensor(pixel_values[None, :, :], dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------
    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)

        # normalize
        img = img.astype(np.float32) / 255.0

        # safety check
        if img.ndim != 2 or img.size == 0:
            raise ValueError("Invalid image after augmentation")

        # (H, W) → (W, H) and flip
        img = img.swapaxes(-2, -1)[..., ::-1]

        target = np.ones(
            (self.input_width, self.input_height), dtype=np.float32
        )

        sx = self.input_width / img.shape[0]
        sy = self.input_height / img.shape[1]
        scale = min(sx, sy)

        new_x = max(1, int(img.shape[0] * scale))
        new_y = max(1, int(img.shape[1] * scale))

        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2

        return 1.0 - target

    # -------------------------------------------------
    # SAFE TACo AUGMENTATION
    # -------------------------------------------------
    def apply_taco_augmentations(self, input_img):
        h, w = input_img.shape[:2]

        # Too small → skip TACo
        if h < 16 or w < 16:
            return input_img

        if random.random() <= self.taco_aug_frac:
            try:
                augmented = self.mytaco.apply_vertical_taco(
                    input_img,
                    corruption_type="random",
                )
                if augmented is None or augmented.size == 0:
                    return input_img
                return augmented
            except Exception:
                return input_img

        return input_img


def collate_fn(batch):
    src_batch, tgt_batch, attn_mask_batch = [], [], []

    for src_sample, tgt_sample, attn_mask_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        attn_mask_batch.append(attn_mask_sample)

    return {
        "pixel_values": torch.stack(src_batch),
        "labels": torch.stack(tgt_batch),
        "attention_mask": torch.stack(attn_mask_batch),
    }
    

class OHWRDataset(Dataset):
    def __init__(
        self,
        root,
        df,
        tokenizer,
        input_width=1600,
        input_height=64,
        aug=False,
        taco_aug_frac=0.9,
        max_length: int = 512,
        # img_list=["img_pressure", "img_thickness", "img_x_tilt", "img_y_tilt", "img_height"],
        img_list=["img_pressure", "img_thickness", "img_x_tilt", "img_y_tilt", "img_height", "img"],
    ):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_list = img_list

        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10,
        )

        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

        # Ensure PAD / BOS exist (GPT-2 safety)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        images = {
            img_name: cv2.imread(
                os.path.join(self.root, self.df[img_name][idx]),
                cv2.IMREAD_GRAYSCALE,
            )
            for img_name in self.img_list
        }
        
        text = self.df["line"][idx]

        # ---- hard guard: NEVER return None ----
        if any(img is None or img.size == 0 for img in images.values()):
            raise ValueError(f"Invalid image in images: {images}")

        pixel_values = {
            img_name: self.preprocess(img, self.aug)
            for img_name, img in images.items()
        }

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Tokenize WITHOUT special tokens
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        )

        input_ids = tokenized["input_ids"]

        # Add BOS / EOS
        input_ids = [bos_id] + input_ids + [eos_id]

        # Attention mask before padding
        attention_mask = [1] * len(input_ids)

        # Pad / truncate
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Labels: PAD → -100 (for CE)
        labels = [
            tok if tok != pad_id else -100
            for tok in input_ids
        ]

        return (
            {img_name: torch.tensor(pixel_values[img_name][None, :, :], dtype=torch.float32) for img_name in pixel_values},
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------
    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)

        # normalize
        img = img.astype(np.float32) / 255.0

        # safety check
        if img.ndim != 2 or img.size == 0:
            raise ValueError("Invalid image after augmentation")

        # (H, W) → (W, H) and flip
        img = img.swapaxes(-2, -1)[..., ::-1]

        target = np.ones(
            (self.input_width, self.input_height), dtype=np.float32
        )

        sx = self.input_width / img.shape[0]
        sy = self.input_height / img.shape[1]
        scale = min(sx, sy)

        new_x = max(1, int(img.shape[0] * scale))
        new_y = max(1, int(img.shape[1] * scale))

        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2

        return 1.0 - target

    # -------------------------------------------------
    # SAFE TACo AUGMENTATION
    # -------------------------------------------------
    def apply_taco_augmentations(self, input_img):
        h, w = input_img.shape[:2]

        # Too small → skip TACo
        if h < 16 or w < 16:
            return input_img

        if random.random() <= self.taco_aug_frac:
            try:
                augmented = self.mytaco.apply_vertical_taco(
                    input_img,
                    corruption_type="random",
                )
                if augmented is None or augmented.size == 0:
                    return input_img
                return augmented
            except Exception:
                return input_img

        return input_img
    

def ocollate_fn(batch):
    src_batch, tgt_batch, attn_mask_batch = [], [], []

    for src_sample, tgt_sample, attn_mask_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        attn_mask_batch.append(attn_mask_sample)

    return {
        "pixel_values": {
            img_name: torch.stack([src[img_name] for src in src_batch])
            for img_name in src_batch[0]
        },
        "labels": torch.stack(tgt_batch),
        "attention_mask": torch.stack(attn_mask_batch),
    }