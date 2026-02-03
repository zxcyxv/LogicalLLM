import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any


class GSM8KDataset(Dataset):
    """
    GSM8K dataset loader using HuggingFace datasets.
    Grade school math word problems requiring multi-step reasoning.
    """
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 1024,
        cache_tokenization: bool = True,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self._token_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._use_cached = False

        # Load GSM8K dataset
        print(f"Loading GSM8K dataset (split={split})...")
        raw = load_dataset("openai/gsm8k", "main", split=split)
        self.examples = raw
        print(f"Loaded {len(self.examples)} examples")

        # Pre-tokenize if caching enabled
        if cache_tokenization:
            print("Pre-tokenizing examples (batched)...")
            self._pretokenize_all()

    def _pretokenize_all(self):
        """Pre-tokenize all examples using batched processing."""

        def tokenize_batch(batch):
            texts = []
            prompts = []

            for i in range(len(batch['question'])):
                question = batch['question'][i]
                answer = batch['answer'][i]

                # GSM8K format: Question + Answer (with reasoning steps)
                # Answer format: "Step 1...\nStep 2...\n#### 42"
                prompt = f"Question: {question}\nAnswer:"
                full_text = f"{prompt} {answer}"

                texts.append(full_text)
                prompts.append(prompt)

            # Tokenize full texts
            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            # Tokenize prompts to get answer start positions
            prompt_encodings = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            # Answer starts right after prompt
            answer_starts = [len(p) for p in prompt_encodings['input_ids']]

            return {
                'cached_input_ids': encodings['input_ids'],
                'cached_attention_mask': encodings['attention_mask'],
                'answer_start': answer_starts,
            }

        # Batched map with multiprocessing
        self.examples = self.examples.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=4,
            desc="Tokenizing",
        )
        self._use_cached = True
        print(f"Tokenized {len(self.examples)} examples")

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a GSM8K example into a single text string."""
        question = example.get('question', '')
        answer = example.get('answer', '')

        # Format: Question + Answer
        text = f"Question: {question}\nAnswer: {answer}"
        return text

    def _tokenize_and_cache(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize an example and cache it."""
        if idx in self._token_cache:
            return self._token_cache[idx]

        example = self.examples[idx]
        text = self._format_example(example)

        # Tokenize without padding (dynamic padding will be done in collator)
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        result = {
            "input_ids": encodings.input_ids.squeeze(0),
            "attention_mask": encodings.attention_mask.squeeze(0),
        }

        if self.cache_tokenization:
            self._token_cache[idx] = result

        return result

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if hasattr(self, '_use_cached') and self._use_cached:
            example = self.examples[idx]
            return {
                "input_ids": torch.tensor(example['cached_input_ids'], dtype=torch.long),
                "attention_mask": torch.tensor(example['cached_attention_mask'], dtype=torch.long),
                "answer_start": torch.tensor(example.get('answer_start', 0), dtype=torch.long),
            }
        return self._tokenize_and_cache(idx)


class DynamicPaddingCollator:
    """
    Collator that pads to the max length in the batch (rounded to multiple of 8).
    """
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(item["input_ids"].size(0) for item in batch)

        # Round up to multiple of pad_to_multiple_of for efficiency
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        # Pad all sequences
        padded_input_ids = []
        padded_attention_mask = []
        answer_starts = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            padding_len = max_len - seq_len

            # Pad input_ids with pad_token_id
            input_ids = torch.cat([
                item["input_ids"],
                torch.full((padding_len,), self.pad_token_id, dtype=torch.long)
            ])

            # Pad attention_mask with 0s
            attention_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(padding_len, dtype=torch.long)
            ])

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            answer_starts.append(item.get("answer_start", torch.tensor(0)))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_input_ids).clone(),  # Labels = input_ids for CLM
            "answer_start": torch.stack(answer_starts),
        }


def get_gsm8k_loader(
    tokenizer,
    split: str = "train",
    max_length: int = 1024,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_tokenization: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for GSM8K dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ("train", "test")
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading workers
        cache_tokenization: Whether to pre-tokenize and cache examples

    Returns:
        DataLoader for the GSM8K dataset
    """
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        cache_tokenization=cache_tokenization,
    )

    collator = DynamicPaddingCollator(tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Test the loader
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("=" * 50)
    print("Testing GSM8K Loader")
    print("=" * 50)

    # Test train loader
    print("\n[1] Train Loader:")
    train_loader = get_gsm8k_loader(
        tokenizer,
        split="train",
        batch_size=2,
        max_length=1024,
        cache_tokenization=True,
        shuffle=False,
    )
    batch = next(iter(train_loader))
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Answer starts: {batch['answer_start'].tolist()}")

    # Decode first example
    print(f"\nFirst example (truncated):")
    text = tokenizer.decode(batch['input_ids'][0])
    print(text[:500] + "...")

    # Test test loader
    print("\n[2] Test Loader:")
    test_loader = get_gsm8k_loader(
        tokenizer,
        split="test",
        batch_size=2,
        max_length=1024,
        cache_tokenization=True,
        shuffle=False,
    )
    batch = next(iter(test_loader))
    print(f"Test batch loaded: {batch['input_ids'].shape}")

    print("\n" + "=" * 50)
    print("GSM8K Loader Ready!")
    print("=" * 50)
