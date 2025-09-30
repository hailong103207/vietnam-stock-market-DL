import torch
import torch.nn as nn
from model_structure import ModelStructure
from utils.loader import load_config
import yaml

'''
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
        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.dropout = self.config["dropout"]
        self.bias = self.config["bias"]
        self.batch_first = self.config["batch_first"]
        self.bidirectional = self.config["bidirectional"]
        self.device = self.config["device"]

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.hidden_size, 16)
        self.dp1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #init cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.dp1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

