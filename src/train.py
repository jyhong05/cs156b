import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from dataset import CheXpertDataset
from model import CheXpertBaseline


def load_config(config_path: str):
	with open(config_path, "r", encoding="utf-8") as f:
		return json.load(f)


def main() -> None:
	config_path = "configs/week1.json"
	config = load_config(config_path)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("configs and device loaded")

	dataset = CheXpertDataset(config=config, split="train")
	dataloader = DataLoader(
		dataset,
		batch_size=config.get("batch_size", 16),
		num_workers=config.get("num_workers", 4),
		shuffle=True,
		pin_memory=torch.cuda.is_available(),
	)
	print("dataloader loaded")

	model = CheXpertBaseline(num_classes=9).to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
	epochs = max(1, min(int(config.get("epochs", 1)), 3))
	print(f"running for {epochs} epochs")

	run = wandb.init(
		project=config.get("wandb_project", "cs156b-week1"),
		name=config.get("wandb_run_name", "week1-baseline-train"),
		config=config,
		mode=os.getenv("WANDB_MODE", "offline"),
	)
	print("wandb initialized")

	for epoch in range(epochs):
		print(f"Starting epoch {epoch + 1}/{epochs}...")
		model.train()
		total_loss = 0.0
		n_batches = 0

		for images, labels in dataloader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			# ResNet18 expects 3-channel input; dataset outputs grayscale [B,1,H,W].
			if images.shape[1] == 1:
				images = images.repeat(1, 3, 1, 1)

			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1

		avg_loss = total_loss / max(1, n_batches)
		print(f"Epoch [{epoch + 1}/{epochs}] - loss: {avg_loss:.6f}")
		wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

	os.makedirs("checkpoints", exist_ok=True)
	checkpoint_path = "checkpoints/baseline.pth"
	torch.save(model.state_dict(), checkpoint_path)
	print(f"Saved checkpoint: {checkpoint_path}")

	run.finish()
	print("Baseline training completed successfully.")


if __name__ == "__main__":
	main()
