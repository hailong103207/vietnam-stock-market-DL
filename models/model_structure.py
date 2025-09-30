'''
Abstract bass for models
'''
from abc import ABC, abstractmethod

class ModelStructure(ABC):
    
    @abstractmethod
    def forward(self, x): pass
    
    @abstractmethod
    def train(self, data_loader, epochs): pass

    @abstractmethod
    def save_model(self, save_path): pass