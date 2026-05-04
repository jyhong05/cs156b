import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from dataset import CheXpertDataset, TARGET_PATHOLOGIES
from models import available_models, get_model


DEFAULT_CONFIG_PATH = "configs/resnet18.json"


def load_config(config_path: str):
	with open(config_path, "r", encoding="utf-8") as f:
		return json.load(f)


def parse_args():
	parser = argparse.ArgumentParser(description="Train a CheXpert model.")
	parser.add_argument(
		"--config",
		default=os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH),
		help="Path to a JSON config file.",
	)
	parser.add_argument(
		"--model",
		choices=available_models(),
		default=None,
		help="Optional model override for the selected config.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	print("running training")
	config = load_config(args.config)
	if args.model is not None:
		config["model_name"] = args.model
		config["checkpoint_path"] = f"checkpoints/{args.model}.pth"
		config["wandb_run_name"] = f"{args.model}-train"
		if args.model in {"customcnn", "densenet264"}:
			config["pretrained"] = False

	model_name = config.get("model_name", "resnet18")
	pretrained = config.get("pretrained")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"configs and device loaded. model={model_name}, device={device}")

	dataset = CheXpertDataset(config=config, split="train")
	dataloader = DataLoader(
		dataset,
		batch_size=config.get("batch_size", 16),
		num_workers=config.get("num_workers", 4),
		shuffle=True,
		pin_memory=torch.cuda.is_available(),
	)
	print("dataloader loaded")

	model = get_model(
		model_name=model_name,
		num_classes=len(TARGET_PATHOLOGIES),
		pretrained=pretrained,
	).to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
	epochs = max(1, int(config.get("epochs", 1)))
	print(f"running for {epochs} epochs")

	run = wandb.init(
		project=config.get("wandb_project", "cs156b-week1"),
		name=config.get("wandb_run_name", f"{model_name}-train"),
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

			# TorchVision CNN backbones expect 3-channel input.
			if images.shape[1] == 1:
				images = images.repeat(1, 3, 1, 1)

			optimizer.zero_grad()
			preds = torch.tanh(model(images))
			loss = criterion(preds, labels)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1

		avg_loss = total_loss / max(1, n_batches)
		print(f"Epoch [{epoch + 1}/{epochs}] - loss: {avg_loss:.6f}")
		wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

	checkpoint_path = config.get("checkpoint_path", f"checkpoints/{model_name}.pth")
	checkpoint_dir = os.path.dirname(checkpoint_path)
	if checkpoint_dir:
		os.makedirs(checkpoint_dir, exist_ok=True)
	torch.save(model.state_dict(), checkpoint_path)
	print(f"Saved checkpoint: {checkpoint_path}")

	run.finish()
	print(f"{model_name} training completed successfully.")


if __name__ == "__main__":
	main()
