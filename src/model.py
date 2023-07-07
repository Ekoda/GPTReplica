import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from src.components.decoder import Decoder
from src.components.positional_encoding import PositionalEncoding


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dimensions = config["model"]["model_dimensions"]
        self.decoder_layers = config["model"]["decoder_layers"]
        self.n_heads = config["model"]["attention_heads"]
        self.vocab_size = config["model"]["vocab_size"]
        self.dropout_rate = config["training"]["dropout_rate"]
        self.device = config["hardware"]["device"]

        self.embeddings = nn.Embedding(self.vocab_size, self.model_dimensions)
        self.position_encoder = PositionalEncoding(self.model_dimensions, self.dropout_rate)
        self.decoder_layers = nn.ModuleList([Decoder(self.model_dimensions, self.n_heads, self.dropout_rate) for _ in range(self.decoder_layers)])
        self.output_layer = nn.Linear(self.model_dimensions, self.vocab_size)

        if config["parameters"]["load"]["should_load"]:
            self.load_state_dict(torch.load(config["parameters"]["load"]["path"]))


    def train_model(self, dataloader, config, grad_clip=None):

        # Load config
        optimizer_class = getattr(optim, config["training"]["optimizer"])
        criterion_class = getattr(nn, config["training"]["loss_function"])
        optimizer = optimizer_class(self.parameters(), lr=config["training"]["learning_rate"])
        criterion = criterion_class()
        grad_clip = config["training"]["grad_clip"]
        n_epochs = config["training"]["n_epochs"]
        should_save = config["parameters"]["save"]["should_save"]
        save_path = config["parameters"]["save"]["path"]

        self.to(self.device)
        for epoch in range(n_epochs):
            start_time = time.time()
            self.train()
            epoch_loss = 0 
            for i, batch in enumerate(dataloader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad() 
                output = self.forward(inputs)
                loss = criterion(output.view(-1, self.vocab_size), targets.view(-1))
                loss.backward() 
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step() 
                epoch_loss += loss.item()
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}, Time = {epoch_time:.3f}s')
            if should_save:
                torch.save(self.state_dict(), save_path)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embeddings(X)
        X = self.position_encoder(X)
        for decoder in self.decoder_layers:
            X = decoder.forward(X)
        X = self.output_layer(X)
        return X
