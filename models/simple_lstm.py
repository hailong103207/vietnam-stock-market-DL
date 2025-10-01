import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .model_structure import ModelStructure
from utils.loader import load_config
import yaml

'''
Config: configs/simple_lstm.yaml
Input:
- Sequence: 20 ngày, 7 features: %open, %high, %low, %close, %volume, sma5, volume
- Input type: float32
* feature dưới dạng % biểu thị tỉ lệ thay đổi so với ngày trước đó ví dụ: %close = (close - close_prev)/close_prev
Output:
- %sma5 trong t+8
Model:
LSTM(input_size=7, hidden_size=32, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
Linear(hidden_size, 16)
ReLU()
dropout(0.05)
Linear(16, 1)
'''

class SimpleLSTM(nn.Module, ModelStructure):
    def __init__(self):
        super(SimpleLSTM, self).__init__()

        # Load configuration from YAML file
        self.config = load_config("simple_lstm")
        self.lstm_config = self.config["lstm"]
        self.train_config = self.config["training"]
        self.device = self.config["device"]
        self.lstm = nn.LSTM(input_size=self.lstm_config["input_size"],
                            hidden_size=self.lstm_config["hidden_size"],
                            num_layers=self.lstm_config["num_layers"],
                            bias=self.lstm_config["bias"],
                            batch_first=self.lstm_config["batch_first"],
                            dropout=self.lstm_config["dropout"],
                            bidirectional=self.lstm_config["bidirectional"])
        self.fc1 = nn.Linear(self.lstm_config["hidden_size"], 16)
        self.dp1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #init cell state
        h0 = torch.zeros(self.lstm_config["num_layers"], x.size(1), self.lstm_config["hidden_size"]).to(self.device)
        c0 = torch.zeros(self.lstm_config["num_layers"], x.size(1), self.lstm_config["hidden_size"]).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.dp1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def save_model(self, save_path):
        save_path = self.train_config["save_path"] if save_path is None else save_path
        torch.save(self.state_dict(), save_path)
    
    def trainn(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config["learning_rate"])
        self.to(self.device)
        for epoch in range(self.train_config["num_epochs"]):
            self.train()
            train_losses = []
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Forward pass
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                train_losses.append(loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
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