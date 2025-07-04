from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataset(tokenizer, max_length=512):
    dataset = load_dataset("imdb")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return tokenized
