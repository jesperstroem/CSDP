from abc import ABC, abstractmethod

class Pipeline_Factory(ABC):
    @abstractmethod
    def create_training_pipeline(self):
        pass

    @abstractmethod
    def create_validation_pipeline(self):
        pass
    
    @abstractmethod
    def create_test_pipeline(self):
        pass


