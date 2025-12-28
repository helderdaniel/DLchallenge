#Model operations 
#
#v0.1 jul 2022, v0.2 aug 2024, v0.3 nov 2024
#hdaniel@ualg.pt
#

from typing import *
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Model(ABC):

    def __init__(self, name:str='') -> None:
        self._name = name       
        self.config()       #run default config
        self._model = None  #Must be defined by subclasses

    def config(self, epochs:int=1, threads:int=1, verbose:int=0) -> None:  
        self._epochs = epochs
        self._threads = threads
        self._verbose = verbose

    def model(self) -> Any:
        return self._model

    def name(self) -> str:
        return self._name

    @abstractmethod
    def fileExtension(self) -> str:
        pass

    @classmethod
    def fromFile(cls, fn:str) -> Any:
        model = cls()
        model.load(fn)
        return model

    @abstractmethod
    def load(self, fn:str) -> None:
        pass
    
    @abstractmethod
    def save(self, fn:str) -> Any:
        pass

    @abstractmethod
    def initialEpoch(self) -> int:
        pass

    @abstractmethod
    def dim(self) -> int:
        pass
    
    @abstractmethod
    def inputShape(self) -> Tuple:
        '''
        The Size of input data:
            for Keras 2D models like CNNs: (samples, width, height, colour maps)
                for 100 images, 20x50 pixels, RGB:  (100, 20, 50, 3)
                for 100 images, 20x50 pixels, Gray: (100, 20, 50, 1)

            for Keras 1D models: (samples, sample_length, maps)
                for 100 samples, 20 points each:    (100, 20, 1)

        The input shape:
            Since the input layer has the same shape for every sample
            first Tuple element is None:
                (None, 20, 50, 3)
                (None, 20, 50, 1)
                (None, 20, 1)

        SKLearn models should follow the same convention.

        https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
        '''
        pass # not needed for @abstractmethod: raise NotImplementedError


    @abstractmethod
    def ouputShape(self) -> Tuple:
        pass

    @abstractmethod
    def modelCountParams(self) -> Tuple:
        pass 

    @abstractmethod
    def __str__(self) -> Tuple:
        pass 

    @abstractmethod
    def valid(self, fn:str) -> bool: 
        pass # not needed for @abstractmethod: raise NotImplementedError
    
    @abstractmethod
    def invalidMsg(self) -> str: 
        pass # not needed for @abstractmethod: raise NotImplementedError

    #@abstractmethod
    #def fit(self, X:NDArray, y:NDArray, batchSize:int=32, shuffle:bool=True) -> Any:
    #    pass # not needed for @abstractmethod: raise NotImplementedError
    
    @abstractmethod
    def score(self, Xe:NDArray, ye:NDArray) -> float:
        pass # not needed for @abstractmethod: raise NotImplementedError
    
    @abstractmethod
    def predict(self, X:NDArray) -> NDArray:
        pass # not needed for @abstractmethod: raise NotImplementedError
