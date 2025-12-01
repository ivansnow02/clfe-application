import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import make_dataloader
from .metrics import MetricsTop
from .model import build_model


def load_trained_model(config_path: str, ckpt_path: str, device: torch.device):
    args = load_config(config_path)
    model = build_model(args).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # allow both plain state_dict or dict with state_dict key
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, args


@torch.no_grad()
def run_inference(config_path: str, ckpt_path: str, mode: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = load_trained_model(config_path, ckpt_path, device)
    loader = make_dataloader(args, mode=mode)

    all_preds, all_labels = [], []
    for batch in loader:
        text = batch["text"].to(device)
        vision = batch["vision"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"]["M"].to(device)
        outputs = model(vision, audio, text)
        all_preds.append(outputs.squeeze().cpu().numpy())
        all_labels.append(labels.squeeze().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_labels, axis=0)
    metrics_fn = MetricsTop().get_metrics(args.dataset.datasetName)
    results = metrics_fn(preds, trues)

    print("Inference finished. MOSEI-style metrics:")
    for k, v in results.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="CLFE inference CLI (no training)")
    parser.add_argument(
        "--config", type=str, default="mosei.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best_model.pth",
        help="Path to trained model checkpoint (state_dict or wrapped)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="Dataset split to run on: train/valid/test",
    )
    args = parser.parse_args()

    run_inference(args.config, args.ckpt, mode=args.mode)


if __name__ == "__main__":
    main()
