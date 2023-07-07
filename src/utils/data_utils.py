from torch.utils.data import DataLoader, TensorDataset
import torch
import tiktoken
import numpy as np
import yaml


def split_data(data: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Split data into training and validation sets."""
    n = len(data)
    train_data = data[:int(n * train_ratio)]
    val_data = data[int(n * train_ratio):]
    return train_data, val_data


def tokenize_and_save(
    train_data: str,
    val_data: str,
    encoder_type: str = "gpt2",
    train_output_path: str = "data/tokens/train_tokens.bin",
    val_output_path: str = "data/tokens/val_tokens.bin",
    dtype: any = np.uint16,
) -> None:
    """Tokenize data and save it to binary files using tiktoken."""
    encoder = tiktoken.get_encoding(encoder_type)
    train_ids = encoder.encode_ordinary(train_data)
    val_ids = encoder.encode_ordinary(val_data)
    print(f"train data has {len(train_ids):,} tokens")
    print(f"validation data has {len(val_ids):,} tokens")
    train_ids = np.array(train_ids, dtype=dtype)
    val_ids = np.array(val_ids, dtype=dtype)
    train_ids.tofile(train_output_path)
    val_ids.tofile(val_output_path)
    print(f"Vocabulary size: {encoder.n_vocab}")


def tokenize(data: str, encoder_type: str = "gpt2") -> np.ndarray:
    """Tokenize data using tiktoken."""
    encoder = tiktoken.get_encoding(encoder_type)
    return np.array(encoder.encode_ordinary(data))


def decode_tokens(token_ids: np.ndarray, encoder_type: str = "gpt2") -> str:
    """Decode a sequence of token IDs back into text."""
    encoder = tiktoken.get_encoding(encoder_type)
    return encoder.decode(token_ids.tolist())


def load_tokens(token_file_path: str, dtype: any = np.uint16) -> np.ndarray:
    """Load tokenized data from a binary file."""
    return np.fromfile(token_file_path, dtype)


def load_config(config_path):
    """
    Loads a configuration file from a specified path. 

    The configuration file should be in YAML format and contain sections for hardware, 
    model, and training configurations. The function uses `yaml.safe_load` to parse 
    the file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file. This should be a YAML file.

    Returns
    -------
    config : dict
        A dictionary containing the parsed configuration file. It includes 
        settings for 'hardware', 'model', and 'training'. Each of these keys map 
        to another dictionary with the appropriate configurations.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data: torch.Tensor, config: dict) -> DataLoader:
    """
    Transforms the raw data into a format that is suitable for training a sequence model,
    and wraps it in a DataLoader for batched processing.

    The function first trims the data to ensure that it is divisible by the sequence length, 
    reshapes it into sequences of the specified length, and then creates input-target pairs 
    by shifting the sequences by one position. The data is then converted to PyTorch tensors 
    and wrapped in a DataLoader.

    Parameters
    ----------
    data : torch.Tensor
        The raw data to be transformed. This is expected to be a 1-dimensional torch.Tensor 
        or similar data structure.
    config : dict
        The configuration dictionary containing training parameters. This function expects 
        that 'sequence_length' and 'batch_size' will be keys in the 'training' sub-dictionary.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        A DataLoader wrapping the input and target data of shape (batch_size, sequence_length), ready for batched processing.
    """
    seq_len = config["training"]["sequence_length"]
    batch_size = config["training"]["batch_size"]

    # Trim data to ensure it is divisible by the sequence length
    num_batches = len(data) // seq_len
    data = data[:num_batches*seq_len]
    data = data.reshape(-1, seq_len) 

    inputs = data[:, :-1] # All but the last position
    targets = data[:, 1:] # Shift by one position

    # Convert to PyTorch tensors
    inputs = inputs.clone().detach().long()
    targets = targets.clone().detach().long()

    # Wrap in a TensorDataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader