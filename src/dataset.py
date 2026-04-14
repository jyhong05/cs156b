import hashlib
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


TARGET_PATHOLOGIES: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class CheXpertDataset(Dataset):
    def __init__(self, config: Dict, split: str = "train") -> None:
        if split != "train":
            raise ValueError("Only 'train' split is implemented for Week 1.")

        self.config = config
        self.train_csv = config["train_csv"]
        self.image_base_dir = config["image_base_dir"]
        self.cache_dir = config.get("cache_dir", "cache")
        self.image_size: Tuple[int, int] = tuple(config.get("image_size", [256, 256]))
        self.nan_policy = config.get("nan_policy", "zero").lower()
        self.targets = TARGET_PATHOLOGIES

        os.makedirs(self.cache_dir, exist_ok=True)

        self.df = pd.read_csv(self.train_csv)
        self.path_column = self._find_path_column(self.df)

        missing_targets = [t for t in self.targets if t not in self.df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns in CSV: {missing_targets}")

        self.df[self.targets] = self.df[self.targets].apply(pd.to_numeric, errors="coerce")
        self.df[self.targets] = self._apply_nan_policy(self.df[self.targets])

    @staticmethod
    def _find_path_column(df: pd.DataFrame) -> str:
        candidates = ["Path", "path", "image_path", "Image Index", "image"]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError("Could not find an image path column in training CSV.")

    def _apply_nan_policy(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        if self.nan_policy == "zero":
            return labels_df.fillna(0.0)
        if self.nan_policy == "mean":
            means = labels_df.mean(skipna=True).fillna(0.0)
            return labels_df.fillna(means)
        if self.nan_policy == "ignore":
            return labels_df
        raise ValueError(
            f"Unsupported nan_policy='{self.nan_policy}'. Use one of: zero, mean, ignore."
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_ref: str) -> str:
        if not isinstance(image_ref, str) or not image_ref.strip():
            raise ValueError(f"Invalid image path value: {image_ref}")

        image_ref = image_ref.strip()
        
        # --- THE FIX: Clean the CSV path to match the cluster's modified file system ---
        clean_ref = image_ref.replace("CheXpert-v1.0/", "")
        clean_ref = clean_ref.replace("train/", "").replace("valid/", "")
        clean_ref = clean_ref.replace("patient", "pid")
        
        if os.path.isabs(clean_ref) and os.path.exists(clean_ref):
            return clean_ref

        candidates = [
            os.path.join(self.image_base_dir, clean_ref),
        ]

        for c in candidates:
            if os.path.exists(c):
                return c

        raise FileNotFoundError(
            f"Image not found. Original: '{image_ref}'. Cleaned: '{clean_ref}'. Tried: {candidates}"
        )

    def _cache_path(self, image_ref: str) -> str:
        key = f"{image_ref}|{self.image_size[0]}x{self.image_size[1]}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.png")

    def _load_or_create_cached_image(self, image_ref: str) -> Image.Image:
        cache_path = self._cache_path(image_ref)
        if os.path.exists(cache_path):
            try:
                with Image.open(cache_path) as cached:
                    return cached.convert("L")
            except Exception:
                # If the cache file exists but is corrupted (0 bytes), delete it and recreate
                os.remove(cache_path)

        source_path = self._resolve_image_path(image_ref)
        with Image.open(source_path) as img:
            resized = img.convert("L").resize(self.image_size, Image.BILINEAR)
            resized.save(cache_path)
            return resized

    def __getitem__(self, idx: int):
        try:
            row = self.df.iloc[idx]
            image_ref = row[self.path_column]
            image = self._load_or_create_cached_image(image_ref)

            image_arr = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_arr).unsqueeze(0)

            labels = row[self.targets].to_numpy(dtype=np.float32)
            label_tensor = torch.from_numpy(labels)

            return image_tensor, label_tensor
            
        except FileNotFoundError:
            # The cluster is missing this image file. Grab the next one.
            # The modulo operator (%) ensures we wrap around to 0 if the last image is missing.
            print(f"Warning: Image missing from HPC storage for index {idx}. Skipping...")
            return self.__getitem__((idx + 1) % len(self))
            
        except Exception as e:
            # Catching PIL UnidentifiedImageErrors (corrupted cache) so it doesn't crash
            print(f"Warning: Corrupted file encountered at index {idx} ({e}). Skipping...")
            return self.__getitem__((idx + 1) % len(self))