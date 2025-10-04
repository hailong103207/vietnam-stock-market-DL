import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .model_structure import ModelStructure
from utils.loader import load_config
import yaml

'''
Config: configs/simple_lstm.yaml
Model:
LSTM(input_size=7, hidden_size=32, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
Linear(hidden_size, 1)
'''

class SimpleLSTM(nn.Module, ModelStructure):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.config = load_config("simple_lstm")
        self.model_config = self.config["model"]
        self.init_configs()
        self.init_layers()
        self.init_weights()
        # Load configuration from YAML file

    def init_configs(self):
        self.input_features = self.model_config["input_size"]
        self.hidden_size = self.model_config["hidden_size"]
        self.num_layers = self.model_config["num_layers"]
        self.bias = self.model_config["bias"]
        self.batch_first = self.model_config["batch_first"]
        self.dropout = self.model_config["dropout"]

    def init_layers(self):
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout
        )
        self.fc1 = nn.Linear(self.hidden_size, 1)

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, x):
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (h0, c0) = self.lstm(x, (hidden_states, cell_states))
        print(out.shape)
        print(h0.shape)
        print(c0.shape)
        ans = self.fc1(out)
        return ans

    def save_model(self, save_path):
        save_path = self.train_config["save_path"] if save_path is None else save_path
        torch.save(self.state_dict(), save_path)
    
    def train_by_config(self, train_loader, val_loader):
        criterion = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config["learning_rate"])
        self.to(self.device)
        for epoch in range(self.train_config["num_epochs"]):
            self.train()

            train_losses = []
            #print batch size
            print(f'Epoch {epoch+1}, Batch size: {train_loader.batch_size}')
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # check if different inputs shape and targets shape

                # Forward pass
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                train_losses.append(loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)  # Gradient clipping
                # check nan grad
                for name, param in self.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f'NaN gradient detected in {name} on epoch {epoch+1}, batch {i+1}')
                        break
                optimizer.step()
            avg_train_loss = np.mean(train_losses)

            # Validation
            self.eval()
            val_losses = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)

            print(f'Epoch [{epoch+1}/{self.train_config["num_epochs"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print("Training complete.")