import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .model_structure import ModelStructure
from utils.loader import load_config
import yaml

'''
Config: configs/simple_lstm.yaml
Input: [%open, %high, %low, %close, %volume, volume, sma_5]
output: [%sma_5_8]
Model:
LSTM(input_size=7, hidden_size=32, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
Linear(hidden_size, 1)
'''

class LSTM_1(nn.Module, ModelStructure):
    def __init__(self):
        super(LSTM_1, self).__init__()
        self.config = load_config("lstm_1")
        self.device = self.config["device"]
        self.init_model_configs()
        self.init_layers()
        self.init_weights()
        self.init_train_configs()
        self.to(self.device)
        # Load configuration from YAML file

    def init_model_configs(self):
        self.model_config = self.config["model"]
        self.lstm = self.model_config["lstm"]
        self.fc1 = self.model_config["fc1"]
        self.dp1 = self.model_config["dp1"]
        self.fc2 = self.model_config["fc2"]
        self.dp2 = self.model_config["dp2"]
        self.fc3 = self.model_config["fc3"]

    def init_train_configs(self):
        optim_dict = {
            "adam" : torch.optim.Adam,
            "sgd" : torch.optim.SGD
        } 
        self.train_config = self.config["train"]
        self.batch_size = self.train_config["batch_size"]
        self.num_epochs = self.train_config["num_epochs"]
        self.learning_rate = self.train_config["learning_rate"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def init_layers(self):
        self.lstm_1 = nn.LSTM(
            input_size=self.lstm["input_size"],
            hidden_size=self.lstm["hidden_size"],
            num_layers=self.lstm["num_layers"],
            bias=self.lstm["bias"],
            batch_first=self.lstm["batch_first"],
            dropout=self.lstm["dropout"]
        )
        self.fc_1 = nn.Linear(self.lstm["hidden_size"], self.fc1["output_size"])
        self.dp_1 = nn.Dropout(self.dp1["dropout"])
        self.fc_2 = nn.Linear(self.fc1["output_size"], self.fc2["output_size"])
        self.dp_2 = nn.Dropout(self.dp2["dropout"])
        self.fc_3 = nn.Linear(self.fc2["output_size"], self.fc3["output_size"])
        self.relu = nn.ReLU()

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, x):
        x.to(self.device)
        hidden_states = torch.zeros(self.lstm["num_layers"], x.size(0), self.lstm["hidden_size"]).to(self.device)
        cell_states = torch.zeros(self.lstm["num_layers"], x.size(0), self.lstm["hidden_size"]).to(self.device)
        out, (h0, c0) = self.lstm_1(x, (hidden_states, cell_states))
        # out: (batch_size, sequence_length, hidden_size)
        # c0, h0: (batch_size, 1, hidden_size)
        out = self.fc_1(out[:,-1,:])
        out = self.relu(out)
        out = self.dp_1(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.dp_2(out)
        out = self.fc_3(out)
        
        return out

    def save_model(self, save_path):
        save_path = self.train_config["save_path"] if save_path is None else save_path
        torch.save(self.state_dict(), save_path)
    
    def train_one_epoch(self, epoch):
        self.train()
        print(f"Epoch: {epoch + 1}")
        running_loss = 0.0
        for batch_index, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            out_batch = self(x_batch)
            # print(out_batch[0], y_batch[0])
            loss_batch = self.loss_function(out_batch, y_batch)
            running_loss += loss_batch
            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()

            if batch_index % 100 == 99:
                avg_loss_across_batches = running_loss/100
                print("Batch {0}, Loss: {1:.4f}".format(batch_index + 1, avg_loss_across_batches))
                running_loss = 0.0
        print()
        
    def validate_one_epoch(self, epoch):
        running_loss = 0.0
        for batch_index, (x_batch, y_batch) in enumerate(self.val_loader):
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            with torch.no_grad():
                out_batch = self(x_batch)
                loss_batch = self.loss_function(out_batch, y_batch)
                running_loss += loss_batch.item()
        avg_loss_across_batches = running_loss / len(self.val_loader)
        print("Val loss: {0:.4f}".format(avg_loss_across_batches))
        print("************************************************************")
        print()

    def train_by_config(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)