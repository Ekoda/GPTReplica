import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tiktoken
from torch.nn import functional as F
from src.components.decoder import Decoder
from src.components.positional_encoding import PositionalEncoding
from torch.utils.data import DataLoader, TensorDataset


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dimensions = config["model"]["model_dimensions"]
        self.layers = config["model"]["layers"]
        self.n_heads = config["model"]["attention_heads"]
        self.dropout_rate = config["training"]["dropout_rate"]
        self.device = config["hardware"]["device"]

        self.tokenizer = tiktoken.get_encoding(config["data"]["tokenizer"])
        self.vocab_size = self.tokenizer.n_vocab

        self.embeddings = nn.Embedding(self.vocab_size, self.model_dimensions)
        self.position_encoder = PositionalEncoding(self.model_dimensions, self.dropout_rate)
        self.layers = nn.ModuleList([Decoder(self.model_dimensions, self.n_heads, self.dropout_rate) for _ in range(self.layers)])
        self.output_layer = nn.Linear(self.model_dimensions, self.vocab_size)

        if config["parameters"]["load"]["should_load"]:
            self.load_state_dict(torch.load(config["parameters"]["load"]["path"]))


    def train_model(self, train_dataloader, val_dataloader, config, grad_clip=None):

        # Load config
        optimizer_class = getattr(optim, config["training"]["optimizer"])
        criterion_class = getattr(nn, config["training"]["loss_function"])
        optimizer = optimizer_class(self.parameters(), lr=float(config["training"]["learning_rate"]))
        criterion = criterion_class()
        grad_clip = config["training"]["grad_clip"]
        n_epochs = config["training"]["n_epochs"]
        should_save = config["parameters"]["save"]["should_save"]
        save_path = config["parameters"]["save"]["path"]

        self.to(self.device)

        for epoch in range(n_epochs):
            start_time = time.time()

            # Training phase
            self.train()
            train_loss = 0 
            for i, batch in enumerate(train_dataloader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad() 
                output = self.forward(inputs)
                loss = criterion(output.view(-1, self.vocab_size), targets.view(-1))
                loss.backward() 
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step() 
                train_loss += loss.item()
            train_loss = train_loss / len(train_dataloader)

            # Evaluation phase
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    output = self.forward(inputs)
                    loss = criterion(output.view(-1, self.vocab_size), targets.view(-1))
                    val_loss += loss.item()
            val_loss = val_loss / len(val_dataloader)

            # Display results and save model
            end_time = time.time()
            epoch_time = end_time - start_time
            print(
                f'Epoch {epoch}:\n'
                    f'\tTrain Loss: {train_loss}\n'
                    f'\tVal Loss: {val_loss}\n'
                    f'\tTime: {epoch_time:.3f}s'
                )
            
            if should_save:
                torch.save(self.state_dict(), save_path)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embeddings(X)
        X = self.position_encoder(X)
        for layer in self.layers:
            X = layer.forward(X)
        X = self.output_layer(X)
        return X


    def predict(self, x: str, temperature: float = 1.0, max_length: int = 50) -> str:
        self.eval()
        self.to(self.device)
        tokens = self.tokenizer.encode_ordinary(x)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.forward(tokens)
                outputs /= temperature
                _, next_token = torch.max(F.softmax(outputs[:, -1, :], dim=-1), dim=-1)
                next_token = next_token.unsqueeze(0)
                tokens = torch.cat((tokens, next_token), dim=-1)
        predicted_text = self.tokenizer.decode(tokens.tolist()[0])
        return predicted_text