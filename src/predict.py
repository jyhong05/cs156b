import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset import TARGET_PATHOLOGIES
from model import CheXpertBaseline


TEST_IDS_CSV = "/groups/CS156b/from_central/data/student_labels/test_ids.csv"
TEST_IMAGE_BASE_DIR = "/groups/CS156b/from_central/data/"


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_test_image_path(image_ref: str) -> str:
    image_ref = str(image_ref).strip()
    if os.path.isabs(image_ref):
        return image_ref
    return os.path.join(TEST_IMAGE_BASE_DIR, image_ref.lstrip("/"))


class InferenceDataset(Dataset):
    """A lightweight dataset just for pushing test images through the pipeline."""
    def __init__(self, df: pd.DataFrame, path_col: str, image_size: List[int]):
        self.df = df
        self.path_col = path_col
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_test_image_path(row[self.path_col])
        
        with Image.open(image_path) as img:
            img = img.convert("L").resize(tuple(self.image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            
        # Create a [1, H, W] tensor
        tensor = torch.from_numpy(arr).unsqueeze(0)
        # Repeat to [3, H, W] to match ResNet input
        tensor = tensor.repeat(3, 1, 1)
        
        # Note: We do NOT add the batch dimension here (unsqueeze(0)). 
        # The DataLoader will automatically stack 32 of these together into [32, 3, H, W].
        return tensor


def find_path_column(df: pd.DataFrame) -> str:
    candidates = ["Path", "path", "image_path", "Image Index", "image"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not find an image path column in test_ids.csv")


def main() -> None:
    print("Running prediction")
    config = load_config("configs/week1.json")
    image_size = config.get("image_size", [256, 256])
    batch_size = config.get("batch_size", 32) # Pull from config or default to 32
    print(f"Config loaded. Batch size: {batch_size}. Starting prediction...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(TEST_IDS_CSV)
    path_col = find_path_column(test_df)

    # Initialize Dataset and DataLoader
    dataset = InferenceDataset(test_df, path_col, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = CheXpertBaseline(num_classes=9).to(device)
    state_dict = torch.load("checkpoints/baseline.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded, running fast batched inference on test set...")

    preds = []
    with torch.no_grad():
        # Iterate over batches instead of single rows
        for batch_images in dataloader:
            batch_images = batch_images.to(device)
            
            logits = model(batch_images)
            # Get probabilities and move the whole batch back to the CPU
            probs = torch.sigmoid(logits).cpu().numpy() 
            preds.append(probs)

    print("Inference complete, formatting and saving submission.csv...")

    # Stack all the batch arrays together
    preds_arr = np.vstack(preds)
    
    for i, label in enumerate(TARGET_PATHOLOGIES):
        test_df[label] = preds_arr[:, i]

    test_df.to_csv("submissions/submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()