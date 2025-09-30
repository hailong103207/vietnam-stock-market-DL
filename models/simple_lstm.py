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
    
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.linear1 = nn.Linear(self.hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.05)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        

