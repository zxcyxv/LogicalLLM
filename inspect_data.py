from datasets import load_dataset

def inspect_babi():
    print("Loading bAbI Task 15...")
    # 'babi_qa' dataset, en-10k config, task 15
    dataset = load_dataset("babi_qa", type="en-10k", task_no="15", trust_remote_code=True)
    
    print("\nDataset Structure:")
    print(dataset)
    
    print("\nSample Entry (Train):")
    print(dataset['train'][0])
    
    print("\nSample Entry (Test):")
    print(dataset['test'][0])

if __name__ == "__main__":
    inspect_babi()
