import torch
from torch.utils.data import Dataset, DataLoader
import os

class BabiDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_and_parse(data_path)
        
    def _load_and_parse(self, path):
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        context = []
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Split ID and text
            space_idx = line.find(" ")
            if space_idx == -1: continue
            
            nid = int(line[:space_idx])
            text_rest = line[space_idx+1:]
            
            if nid == 1:
                context = []
                
            if "?" in text_rest:
                # Question line: "What is the capital? \t Paris \t 1 2"
                parts = text_rest.split("\t")
                question = parts[0]
                answer = parts[1] if len(parts) > 1 else ""
                
                # Extract supporting facts (Chain Length)
                supporting_facts = parts[2].split() if len(parts) > 2 else []
                chain_length = len(supporting_facts)
                
                # Construct full text: Context + Question + Answer
                # Note: bAbI is simple, we can just join sentences.
                full_text = " ".join(context) + " " + question + " " + answer
                examples.append({
                    "text": full_text,
                    "length": chain_length
                })
            else:
                # Context line
                context.append(text_rest)
                
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = item["text"]
        chain_len = item["length"]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # input_ids and attention_mask are [1, SeqLen], squeeze to [SeqLen]
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        
        # Labels for CLM are usually input_ids (shifted inside the model)
        # We return input_ids as labels here.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "chain_length": torch.tensor(chain_len, dtype=torch.long)
        }

def get_dataloader(data_path, tokenizer, batch_size=32, max_length=128):
    dataset = BabiDataset(data_path, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Test block
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    data_path = "data/tasks_1-20_v1-2/en-10k/qa15_basic-deduction_train.txt"
    if os.path.exists(data_path):
        loader = get_dataloader(data_path, tokenizer, batch_size=2)
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print("Input IDs shape:", batch["input_ids"].shape)
        print("First Example decoded:", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
    else:
        print(f"Data file not found at {data_path}")
