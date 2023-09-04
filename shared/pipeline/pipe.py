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
        
        for i, p in enumerate(self.pipes):
            x = p.process(x)
            x = p.adapt(x)
            
            if (x == None) or (x == -1) or (x == -2):
                return x
        
        return x

class TestPipe(IPipe):
    def process(self, x):
        return x + 1
    
class TestPipe2(IPipe):
    def process(self, x):
        return x * 2    


if __name__ == "__main__":
    pipes = [TestPipe(), TestPipe2()]

    p = Pipeline(pipes, 5)

    print(p.get_batch())

