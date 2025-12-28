#Model operations for ML/DL run challenge
#
#v0.1 jul 2022, v0.2 aug 2024, v0.3 nov 2024
#hdaniel@ualg.pt
#

from typing import *
import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from modules.evalprog import EvaluationProgess
#from modules.model import Model
from model import Model
from modules.score import ScoreRank


class ModelRankEval(Model):

    @abstractmethod
    def _rankEval(self, X:NDArray, y:NDArray, evalProg:EvaluationProgess, 
                 rank:ScoreRank, batches:int, batchSize:int=32) -> Tuple[float,float,List[float]]:
        pass # not needed for @abstractmethod: raise NotImplementedError
    
    
    def rankEval(self, X:NDArray, y:NDArray, evalProg:EvaluationProgess, 
                 rank:ScoreRank, batches:int, batchSize:int=32, 
                 shuffle:bool=False, seed:int=None) -> Tuple[float,float,List[float]]:
        if self._model is not None:

            #shuffle different each time before evaluate
            #this will give a different curve every time the same model is evaluated
            #Maybe, it is better not to shuffle to avoid inducing the user in error
            #Or shuffle always with the same seed
            if shuffle:
                rng = np.random.default_rng()    # (seed=0)
                p   = rng.permutation(X.shape[0])
                Xr  = X[p,:]
                yr  = y[p]   
            

            #shuffle evaluation dataset
            if shuffle:
                if seed is None:    # Shuffle different every time the app is restarted
                    rng = np.random.default_rng()    
                else:               # Shuffle with the same seed every time the app is restarted
                    rng = np.random.default_rng(seed=seed)

                p = rng.permutation(X.shape[0])
                Xr  = X[p,:]
                yr  = y[p]
            else:
                Xr = X
                yr = y

            try:
                (loss, acc, accHist) = self._rankEval(Xr, yr, evalProg, rank, batches, batchSize)
                return (loss, acc, accHist)
            except:
                return (0, -1, [])   #acc = -1 signal an error todo: find a better way
