import itertools
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from src.config import HiddenDistillationConfig

class InstructionDataset(Dataset):
    """Combined instruction-response dataset with train/val split.

    Small datasets (alpaca, dolly) are sliced via the HF split string so we only
    materialize what we need. Datasets flagged `dataset_streaming=True` (ultrachat)
    are streamed and truncated with itertools.islice, avoiding a full download of a
    dataset we're only taking a few hundred/thousand rows from.
    """

    def __init__(self, config: HiddenDistillationConfig, tokenizer: AutoTokenizer,
                 max_length: int = 512, split: str = "train", val_size: int = 120, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        all_data = []
        for name, ds_split, size, streaming in zip(
            config.dataset_names, config.dataset_splits, config.dataset_sizes, config.dataset_streaming
        ):
            print(f"Loading {name} (streaming={streaming}, target={size})...")
            try:
                if streaming:
                    ds_iter = load_dataset(name, split=ds_split, streaming=True)
                    dataset = list(itertools.islice(ds_iter, size))
                else:
                    # slice in the split string so we don't materialize more rows than needed
                    dataset = load_dataset(name, split=f"{ds_split}[:{size}]")
                formatted = self._format_dataset(dataset, name)
                all_data.extend(formatted)
                print(f"  Loaded {len(formatted)} examples")
            except Exception as e:
                print(f"  Error loading {name}: {e}")
                continue

        print(f"Total examples loaded: {len(all_data)}")

        np.random.seed(seed)
        indices = np.random.permutation(len(all_data))

        if split == "val":
            val_indices = indices[:val_size]
            self.data = [all_data[i] for i in val_indices]
            print(f"Validation set size: {len(self.data)}")
        else:
            val_indices = set(indices[:val_size].tolist())
            train_indices = [i for i in indices if i not in val_indices]
            self.data = [all_data[i] for i in train_indices]
            print(f"Training set size: {len(self.data)}")

    def _format_dataset(self, dataset, dataset_name: str) -> List[str]:
        examples = []
        for example in dataset:
            if dataset_name == "tatsu-lab/alpaca":
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output = example.get("output", "")
                prompt = (f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse:"
                          if input_text else f"Instruction: {instruction}\n\nResponse:")
                text = f"{prompt} {output}"

            elif dataset_name == "databricks/databricks-dolly-15k":
                instruction = example.get("instruction", "")
                context = example.get("context", "")
                response = example.get("response", "")
                prompt = (f"Instruction: {instruction}\n\nContext: {context}\n\nResponse:"
                          if context else f"Instruction: {instruction}\n\nResponse:")
                text = f"{prompt} {response}"

            elif "ultrachat" in dataset_name:
                messages = example.get("messages", [])
                if not messages:
                    continue
                text_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        text_parts.append(f"User: {content}")
                    elif role == "assistant":
                        text_parts.append(f"Assistant: {content}")
                text = "\n\n".join(text_parts)
            else:
                continue

            if text and len(text) > 50:
                examples.append(text)
        return examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        text = self.data[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                   padding=False, return_tensors=None)
        input_ids = encoding["input_ids"]
        labels = list(input_ids)

        response_tokens = self.tokenizer.encode("Response:", add_special_tokens=False)
        assistant_tokens = self.tokenizer.encode("Assistant:", add_special_tokens=False)

        response_start = len(input_ids)
        for start_pos in range(len(input_ids) - len(response_tokens)):
            if input_ids[start_pos:start_pos + len(response_tokens)] == response_tokens:
                response_start = start_pos + len(response_tokens)
                break

        if response_start == len(input_ids):
            for start_pos in range(len(input_ids) - len(assistant_tokens)):
                if input_ids[start_pos:start_pos + len(assistant_tokens)] == assistant_tokens:
                    response_start = start_pos + len(assistant_tokens)
                    break

        for i in range(response_start):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
