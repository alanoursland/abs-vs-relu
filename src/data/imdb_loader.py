# src/data/imdb_loader.py
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

def collate_batch(batch, tokenizer, max_length=512):
    texts = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    texts = torch.tensor(texts, dtype=torch.int64)
    attention_masks = torch.tensor(attention_masks, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)

    return texts, attention_masks, labels

def load_imdb(batch_size=64, max_length=512, device=torch.device('cpu')):
    """
    Load the IMDB dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - max_length (int): Maximum length of sequences.
    - device (torch.device): Device to load data onto.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    # Load the IMDB dataset from Hugging Face
    dataset = load_dataset('imdb')

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
    
    # Set the format to PyTorch tensors
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create DataLoader instances
    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_batch(batch, tokenizer, max_length))
    test_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False,
                             collate_fn=lambda batch: collate_batch(batch, tokenizer, max_length))

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_imdb(device=device)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
