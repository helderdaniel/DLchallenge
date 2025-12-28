#Model operations for ML/DL run challenge
#
#v0.1 nov 2024
#v0.2 jan 2025
#hdaniel@ualg.pt
#

import pickle
from typing import *
from numpy.typing import NDArray
from model import Model
from datasetutil import DatasetUtil

class ModelSKL(Model):

    _extension='pickle'

    def __init__(self, name:str='', mdl:Any=None) -> None:
        super().__init__(name)
        self._model:Any = mdl   #SKlearn have no interface for models

    def fileExtension(self) -> str:
        return self._extension

    def load(self, fn:str) -> None:
        with open(fn, 'rb') as f:
            self._model = pickle.load(f)

    def save(self, fn:str) -> Any:
        with open(fn, 'wb') as f:
            pickle.dump(self._model, f)


    def initialEpoch(self) -> int:
        return 1

    def dim(self) -> int:
        return 1

    def inputShape(self) -> Tuple:
        #Only have n_feature_in_ defined after training
        try:
            f = self._model.n_features_in_
        except:
            f = None
        return (f,)   #1D model, compatible with keras.model.input.shape, but Striping leading 'None'

    def ouputShape(self) -> Tuple:
        return (len(self._model.classes_),)

    def _countWeights(self, bias, coef):
        dimB = bias.shape
        dimC = coef.shape
        return dimC[0]*dimC[1] + dimB[0]


    def modelCountParams(self) -> Tuple:
        #Only have coef_ defined after training
        try:
            f = self._countWeights(self._model.intercept_, self._model.coef_)
        except:
            f = None

        #Try MLP coefs_ if no coef_ found
        if f is None:
            try:
                f = 0
                for b,c in zip(self._model.intercepts_, self._model.coefs_):
                    f += self._countWeights(b,c)
            except:
                f = None
        return f
    

    def valid(self, fn:str)->bool:
        '''
        Try to load a SKLearn model file and and train it
        if fit() fails, it is not an SKLearn model

        Supports only pickle file format
        '''
        
        try:
            model = self.fromFile(fn)
            model._model.n_features_in_
        except:
            return False
        return True
    
    
    def invalidMsg(self)->str: 
        return 'Invalid SKLearn model file, must be in pickle format'


    def __str__(self) -> str: 
        params = self._model.get_params()
        return str(params)



    #def fit(self, X:NDArray, y:NDArray, batchSize:int=32, shuffle:bool=True) -> None:
    #    '''Run default trainer for model'''
    #    if shuffle: Xr, yr = DatasetUtil.shuffle(X, y)
    #    else:       Xr, yr = X, y
    #    yrc = DatasetUtil.toCategorical(yr)
    #
    #    self._model.fit(Xr, yrc)
            
    
    def score(self, Xe:NDArray, ye:NDArray) -> float:
        yec  = DatasetUtil.toCategorical(ye)
        return self._model.score(Xe, yec)

    
    def predict(self, X:NDArray) -> NDArray:
        return self._model.predict(X)
    
