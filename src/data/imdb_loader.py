# src/data/imdb_loader.py
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)


def collate_batch(batch, tokenizer, max_length=512):
    # texts = [item["input_ids"].clone().detach() for item in batch]
    texts = [item["input_ids"] for item in batch]
    # attention_masks = [torch.tensor(item['attention_mask'], dtype=torch.int64) for item in batch]
    labels = [item["label"] for item in batch]

    # Pad the sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    # attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # max_token_id = max(tensor.max() for tensor in texts)
    # if max_token_id > 2**31 - 1:
    #     raise ValueError("Token IDs exceed the range of 32-bit integers. Cancelling.")

    labels = torch.tensor(labels, dtype=torch.int32)

    # return texts_padded, attention_masks_padded, labels
    return texts_padded, labels.long()

s_tokenizer = None
s_max_length = 0
def s_collate_batch(batch):
    global s_tokenizer
    global s_max_length
    return collate_batch(batch, s_tokenizer, s_max_length)

def load_imdb(batch_size=64, max_length=512, device=torch.device("cpu")):
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
    global s_tokenizer
    global s_max_length

    # Load the IMDB dataset from Hugging Face
    dataset = load_dataset("imdb")

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)

    # Set the format to PyTorch tensors
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    s_tokenizer = tokenizer
    s_max_length = max_length

    # Create DataLoader instances
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        # num_workers=4,
        collate_fn=s_collate_batch,
    )
    test_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        # num_workers=4,
        collate_fn=s_collate_batch,
    )

    return train_loader, test_loader, tokenizer


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_imdb(device=device)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
