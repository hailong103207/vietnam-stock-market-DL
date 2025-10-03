import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .model_structure import ModelStructure
from utils.loader import load_config
import yaml


class SimpleLinear(nn.Module, ModelStructure):
    def __init__(self):
        super(SimpleLinear, self).__init__()

        # Load configuration from YAML file
        self.config = load_config("simple_linear")
        self.model_config = self.config["model"]
        self.train_config = self.config["training"]
        self.device = self.config["device"]
        self.to(self.device)
        self.fc1 = nn.Linear(self.model_config["input_size"], 16)
        self.init_weights()


    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, x):
        out = self.fc1(x)
        return out.to(self.device)
    
    def save_model(self, save_path):
        save_path = self.train_config["save_path"] if save_path is None else save_path
        torch.save(self.state_dict(), save_path)
    
    def trainn(self, train_loader, val_loader):
        criterion = nn.MSELoss(reduction="sum")
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