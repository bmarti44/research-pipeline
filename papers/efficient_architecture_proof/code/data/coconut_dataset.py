"""
COCONUT Dataset with Curriculum Training

Implements the exact data format from the official COCONUT repo:
- Special tokens: <bot>, <thought>, <eot>
- Curriculum stages: gradually replace CoT steps with latent tokens
- Proper label masking: only supervise remaining CoT + answer
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class COCONUTTokenizer:
    """
    Simple tokenizer with COCONUT special tokens.
    In practice, you'd extend a real tokenizer (GPT-2, Llama).
    """
    vocab_size: int = 10000
    pad_token_id: int = 0
    eos_token_id: int = 1
    bot_token_id: int = 2  # Beginning of thought
    thought_token_id: int = 3  # Latent thought placeholder
    eot_token_id: int = 4  # End of thought

    def __post_init__(self):
        # Simple character-level tokenization for demo
        # In production, use a real tokenizer
        self.char_to_id = {}
        self.id_to_char = {
            0: "<pad>",
            1: "<eos>",
            2: "<bot>",
            3: "<thought>",
            4: "<eot>",
        }

        # Build vocab from printable ASCII
        next_id = 5
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            self.char_to_id[char] = next_id
            self.id_to_char[next_id] = char
            next_id += 1

        self.actual_vocab_size = next_id

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.char_to_id.get(" ", 5))  # Unknown -> space
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for id in ids:
            if id in self.id_to_char:
                char = self.id_to_char[id]
                if not char.startswith("<"):  # Skip special tokens
                    chars.append(char)
        return "".join(chars)


class COCONUTDataset(Dataset):
    """
    COCONUT-style dataset with curriculum training.

    The key insight from the paper:
    - Stage 0: Full Chain-of-Thought (no latent tokens)
    - Stage k: Replace first k CoT steps with k latent tokens
    - This teaches the model to compress reasoning into continuous space
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: COCONUTTokenizer,
        max_seq_len: int = 256,
        current_stage: int = 0,
        max_stage: int = 5,
        c_thought: int = 1,  # Latent tokens per replaced CoT step
        uniform_prob: float = 0.2,  # Probability of random stage sampling
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.current_stage = current_stage
        self.max_stage = max_stage
        self.c_thought = c_thought
        self.uniform_prob = uniform_prob

        # Load data
        with open(data_path, "r") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples, stage {current_stage}/{max_stage}")

    def set_stage(self, stage: int):
        """Update curriculum stage."""
        self.current_stage = min(stage, self.max_stage)
        print(f"Curriculum stage set to {self.current_stage}")

    def train(self):
        """Set dataset to training mode (enables curriculum sampling)."""
        self._is_training_mode = True

    def eval(self):
        """Set dataset to eval mode (disables curriculum sampling)."""
        self._is_training_mode = False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # FIX: Use idx-based deterministic random for reproducibility
        # This ensures same idx always returns same stage sampling
        if self._use_curriculum_sampling and self.current_stage > 0:
            # Deterministic "random" based on idx and current_stage
            rng = random.Random(idx + self.current_stage * 10000)
            if rng.random() < self.uniform_prob:
                stage = rng.randint(0, self.current_stage)
            else:
                stage = self.current_stage
        else:
            stage = self.current_stage

        return self._create_training_example(sample, stage)

    @property
    def _use_curriculum_sampling(self) -> bool:
        """Whether to use curriculum stage sampling (only during training)."""
        # FIX: Renamed from 'training' to avoid confusion with nn.Module.training
        return getattr(self, '_is_training_mode', True)

    def _create_training_example(
        self, sample: Dict[str, Any], stage: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create training example with proper token structure:

        [Question] <bot> <thought>*n_latent <eot> [Remaining CoT steps] [Answer]

        Labels mask everything before remaining CoT steps.
        """
        # Tokenize components
        question_tokens = self.tokenizer.encode(sample["question"] + " ")
        answer_tokens = self.tokenizer.encode(" " + sample["answer"])

        # Tokenize each step
        steps = sample.get("steps", [])
        step_tokens_list = [
            self.tokenizer.encode(" " + step + ".") for step in steps
        ]

        # Calculate how many steps to replace with latent tokens
        # FIX: Cap n_skip_steps first, then compute n_latent_tokens from actual skipped steps
        # This ensures n_latent_tokens matches the reasoning being compressed
        n_skip_steps = min(stage, len(steps))  # Steps actually replaced
        n_latent_tokens = n_skip_steps * self.c_thought  # Latent tokens = skipped steps * c_thought

        # Remaining CoT steps (not replaced)
        remaining_step_tokens = []
        for i in range(n_skip_steps, len(steps)):
            remaining_step_tokens.extend(step_tokens_list[i])

        # Build input sequence:
        # [Question] <bot> <thought>*n_latent <eot> [Remaining CoT] [Answer] <eos>
        input_ids = (
            question_tokens
            + [self.tokenizer.bot_token_id]
            + [self.tokenizer.thought_token_id] * n_latent_tokens
            + [self.tokenizer.eot_token_id]
            + remaining_step_tokens
            + answer_tokens
            + [self.tokenizer.eos_token_id]
        )

        # Build labels: mask everything up to and including <eot>
        # Only supervise: [Remaining CoT] [Answer] <eos>
        n_masked = len(question_tokens) + 1 + n_latent_tokens + 1  # Q + <bot> + thoughts + <eot>
        labels = (
            [-100] * n_masked
            + remaining_step_tokens
            + answer_tokens
            + [self.tokenizer.eos_token_id]
        )

        # Truncate to max_seq_len
        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len  # Don't supervise padding

        # Create attention mask
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids]

        # FIX R4: Return tensors for all values to avoid DataLoader collate issues
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "n_latent_tokens": torch.tensor(n_latent_tokens, dtype=torch.long),
            "stage": torch.tensor(stage, dtype=torch.long),
        }


def create_coconut_dataloader(
    data_path: str,
    tokenizer: COCONUTTokenizer,
    batch_size: int = 8,
    max_seq_len: int = 256,
    current_stage: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, COCONUTDataset]:
    """Create dataloader with COCONUT dataset."""
    dataset = COCONUTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        current_stage=current_stage,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # For simplicity
        drop_last=True,
    )

    return dataloader, dataset


if __name__ == "__main__":
    # Test the dataset
    from generate_cot_data import generate_synthetic_cot_data

    # Generate test data
    print("Generating test data...")
    samples = generate_synthetic_cot_data(100)
    test_path = Path("data/test_cot_data.json")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w") as f:
        json.dump(samples, f)

    # Create tokenizer and dataset
    tokenizer = COCONUTTokenizer()
    dataset = COCONUTDataset(str(test_path), tokenizer, max_seq_len=128, current_stage=0)

    # Test different stages
    for stage in range(4):
        dataset.set_stage(stage)
        sample = dataset[0]
        print(f"\nStage {stage}:")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  N latent tokens: {sample['n_latent_tokens']}")
        print(f"  Masked labels: {(sample['labels'] == -100).sum().item()}")

        # Decode to show structure
        ids = sample["input_ids"].tolist()
        thought_count = ids.count(tokenizer.thought_token_id)
        print(f"  <thought> tokens in input: {thought_count}")
