from abc import ABC, abstractmethod

class IPipe(ABC):
    @abstractmethod
    def process(x):
        pass
    
    def adapt(self, x):
        return x

class Pipeline:
    def __init__(self,
                 pipes: list):
        self.pipes = pipes

    def get_batch(self, index):
        x = index
        
        for _, p in enumerate(self.pipes):
            x = p.process(x)
            x = p.adapt(x)
        
        return x