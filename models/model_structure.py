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
    def trainn(self, train_loader, val_loader): pass

    @abstractmethod
    def save_model(self, save_path): pass