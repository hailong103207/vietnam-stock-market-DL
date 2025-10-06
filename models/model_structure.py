'''
Abstract bass for models
'''
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class ModelStructure(ABC):

    @abstractmethod
    def forward(self, x): pass

    @abstractmethod
    def init_model_configs(self): pass

    @abstractmethod
    def init_train_configs(self): pass

    @abstractmethod
    def init_layers(self): pass

    @abstractmethod
    def init_weights(self): pass
    
    @abstractmethod
    def train_one_epoch(self): pass

    @abstractmethod
    def train_by_config(self, train_loader, val_loader): pass

    @abstractmethod
    def save_model(self, save_path): pass