from abc import ABC, abstractmethod

class IDataloader_Factory(ABC):
    @abstractmethod
    def create_training_loader(self):
        pass

    @abstractmethod
    def create_validation_loader(self):
        pass
    
    @abstractmethod
    def create_testing_loader(self):
        pass

class Dataloader_Factory(IDataloader_Factory):
    def create_training_loader(self):
        pass
    
    def create_validation_loader(self):
        pass

    def create_testing_loader(self):
        pass