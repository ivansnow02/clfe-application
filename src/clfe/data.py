import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MMDataset(Dataset):
    """Multimodal dataset for MOSI/MOSEI/SIMS-style pre-extracted features.

    This is identical in spirit to the notebook version but kept minimal for inference.
    """

    def __init__(self, args, mode: str = "test"):
        self.mode = mode
        self.args = args.dataset
        data_path = self.args.dataPath
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.text = data[self.mode]["text_bert"].astype(np.float32)
        self.vision = data[self.mode]["vision"].astype(np.float32)
        self.audio = data[self.mode]["audio"].astype(np.float32)

        self.rawText = data[self.mode]["raw_text"]
        self.ids = data[self.mode]["id"]
        self.labels = {
            "M": data[self.mode][self.args.train_mode + "_labels"].astype(np.float32)
        }
        if self.args.datasetName == "sims":
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode + "_labels_" + m]

        self.audio[self.audio == -np.inf] = 0

    def __len__(self) -> int:
        return len(self.labels["M"])

    def __getitem__(self, index: int) -> Dict:
        sample = {
            "raw_text": self.rawText[index],
            "text": torch.tensor(self.text[index]),
            "audio": torch.tensor(self.audio[index]),
            "vision": torch.tensor(self.vision[index]),
            "index": index,
            "id": self.ids[index],
            "labels": {
                k: torch.tensor(v[index].reshape(-1)) for k, v in self.labels.items()
            },
        }
        return sample


def make_dataloader(args, mode: str = "test") -> DataLoader:
    dataset = MMDataset(args, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=args.base.batch_size,
        num_workers=getattr(args.base, "num_workers", 0),
        shuffle=False,
    )
    return loader
