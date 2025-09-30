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
ReLU(hidden_size, 16)
dropout(0.05)
Linear(16, 1)
'''

class SimpleLSTM(nn.Module, ModelStructure):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.config = load_config("simple_lstm")
        self.model