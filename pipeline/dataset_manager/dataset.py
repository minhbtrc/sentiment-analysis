import json
import torch
from collections import Counter
from itertools import chain
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, config, mode, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data = self.load_dataset(path=f"{self.config.dataset_path}/{mode}.json")
        print(Counter([e[1] for e in self.data]))
        print(f"Load {len(self.data)} examples for {mode} dataset")

    @staticmethod
    def load_dataset(path: str):
        data = json.load(open(path, "r", encoding="utf8"))
        return data["data"]

    def __getitem__(self, idx):
        data_item = self.data[idx]
        labels = torch.zeros(self.config.model_num_labels)
        labels[data_item[-1]] = 1
        segmented_text = " ".join(
            chain.from_iterable(self.config.config.segmentor.word_segment(data_item[0].replace("_", " "))))
        tokenized_text = self.tokenizer(segmented_text, padding="max_length", truncation=True,
                                        max_length=self.config.model_input_max_length, return_tensors="pt")
        return {
            **tokenized_text,
            "labels": labels
        }

    def __len__(self):
        return len(self.data)
