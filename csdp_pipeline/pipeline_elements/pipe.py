from abc import ABC, abstractmethod

class IPipe(ABC):
    @abstractmethod
    def process(x):
        pass

class Pipeline:
    def __init__(self,
                 pipes: list):
        self.pipes = pipes

    def get_batch(self, index):
        x = index
        
        for _, p in enumerate(self.pipes):
            x = p.process(x)
        
        return x