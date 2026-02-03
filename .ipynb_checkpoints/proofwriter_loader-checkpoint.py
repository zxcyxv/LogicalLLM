import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any


class ProofWriterDataset(Dataset):
    """
    ProofWriter dataset loader using HuggingFace datasets.
    Supports depth filtering for OOD generalization experiments.
    """
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 512,
        cache_tokenization: bool = True,
        depth_filter: Optional[str] = None,  # 'shallow', 'deep', or None
        exact_depth: Optional[int] = None,   # For exact depth filtering (legacy)
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self._token_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._use_cached = False

        # Load ProofWriter dataset
        print(f"Loading ProofWriter dataset (split={split})...")
        raw = load_dataset("tasksource/proofwriter", split=split)

        # Filter by depth
        if depth_filter == 'shallow':
            # Depth 1, 2 only (for training) - exclude depth 0 (trivial lookup)
            print("Filtering for shallow depths (QDep 1-2, excluding 0)...")
            self.examples = raw.filter(lambda x: 1 <= x['QDep'] <= 2)
            print(f"Found {len(self.examples)} examples with depth 1-2")
        elif depth_filter == 'deep':
            # Depth 5 only (for OOD testing)
            print("Filtering for deep depths (QDep == 5)...")
            self.examples = raw.filter(lambda x: x['QDep'] == 5)
            print(f"Found {len(self.examples)} examples with depth == 5")
        elif exact_depth is not None:
            # Legacy: exact depth filtering
            print(f"Filtering for QDep == {exact_depth}...")
            self.examples = raw.filter(lambda x: x['QDep'] == exact_depth)
            print(f"Found {len(self.examples)} examples with depth {exact_depth}")
        else:
            # No filtering - use all depths
            print("Using all depths...")
            self.examples = raw
            print(f"Total {len(self.examples)} examples")

        # Pre-tokenize if caching enabled (using batched map for speed)
        if cache_tokenization:
            print("Pre-tokenizing examples (batched)...")
            self._pretokenize_all()

    def _pretokenize_all(self):
        """Pre-tokenize all examples using batched processing."""
        # Get token id for " Answer:" to find answer start position
        answer_marker_ids = self.tokenizer.encode(" Answer:", add_special_tokens=False)

        def tokenize_batch(batch):
            texts = []
            prompts = []  # Everything before the answer
            for i in range(len(batch['question'])):
                context = batch.get('theory', batch.get('context', [''] * len(batch['question'])))[i] or ''
                question = batch['question'][i]
                answer = batch['answer'][i]

                prompt = f"{context} Question: {question} Answer:"
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
        """Format a ProofWriter example into a single text string."""
        # Format: Context + Question + Answer
        context = example.get('theory', '') or example.get('context', '')
        question = example.get('question', '')
        answer = example.get('answer', '')

        # ProofWriter format: theory contains facts/rules, question is the query
        text = f"{context} Question: {question} Answer: {answer}"
        return text

    def _tokenize_and_cache(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize an example and cache it."""
        if idx in self._token_cache:
            return self._token_cache[idx]

        example = self.examples[idx]
        text = self._format_example(example)
        depth = example.get('QDep', 0)  # Get actual depth from example

        # Tokenize without padding (dynamic padding will be done in collator)
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # No padding here - done in collator
            return_tensors="pt"
        )

        result = {
            "input_ids": encodings.input_ids.squeeze(0),
            "attention_mask": encodings.attention_mask.squeeze(0),
            "depth": torch.tensor(depth, dtype=torch.long),
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
                "depth": torch.tensor(example.get('QDep', 0), dtype=torch.long),
                "answer_start": torch.tensor(example.get('answer_start', 0), dtype=torch.long),
            }
        return self._tokenize_and_cache(idx)


class DynamicPaddingCollator:
    """
    Collator that pads to the max length in the batch (rounded to multiple of 8).
    This is more efficient than padding to a fixed max_length.
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
        depths = []
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
            depths.append(item["depth"])
            answer_starts.append(item.get("answer_start", torch.tensor(0)))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_input_ids).clone(),  # Labels = input_ids for CLM
            "depth": torch.stack(depths),
            "answer_start": torch.stack(answer_starts),
        }


def get_proofwriter_loader(
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_tokenization: bool = True,
    depth_filter: Optional[str] = None,  # 'shallow', 'deep', or None
    exact_depth: Optional[int] = None,   # For exact depth filtering (legacy)
) -> DataLoader:
    """
    Create a DataLoader for ProofWriter dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ("train", "validation", "test")
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading workers
        cache_tokenization: Whether to pre-tokenize and cache examples
        depth_filter: 'shallow' (depth 0-2), 'deep' (depth 5), or None (all)
        exact_depth: Filter for exact QDep value (legacy, use depth_filter instead)

    Returns:
        DataLoader for the ProofWriter dataset
    """
    dataset = ProofWriterDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        cache_tokenization=cache_tokenization,
        depth_filter=depth_filter,
        exact_depth=exact_depth,
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
    # Test the loader with OOD setting
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("=" * 50)
    print("Testing OOD Setting: Train Shallow / Test Deep")
    print("=" * 50)

    # Test shallow (train) loader
    print("\n[1] SHALLOW (Train) Loader:")
    train_loader = get_proofwriter_loader(
        tokenizer,
        split="train",
        batch_size=4,
        depth_filter='shallow',
        cache_tokenization=True
    )
    batch = next(iter(train_loader))
    print(f"Batch depths: {batch['depth'].tolist()}")
    print(f"All depth <= 2: {all(d <= 2 for d in batch['depth'])}")

    # Test deep (test) loader
    print("\n[2] DEEP (Test) Loader:")
    test_loader = get_proofwriter_loader(
        tokenizer,
        split="validation",
        batch_size=4,
        depth_filter='deep',
        cache_tokenization=True
    )
    batch = next(iter(test_loader))
    print(f"Batch depths: {batch['depth'].tolist()}")
    print(f"All depth == 5: {all(d == 5 for d in batch['depth'])}")

    print("\n" + "=" * 50)
    print("OOD Setup Ready!")
    print("=" * 50)
