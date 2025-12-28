#Model operations for ML/DL run challenge
#
#v0.1 nov 2024
#v0.2 jan 2025
#hdaniel@ualg.pt
#

import numpy as np
from typing import *
from numpy.typing import NDArray
from modules.evalprog import EvaluationProgess
from modules.evalprogupdate import EvalProgressUpdate
#from modules.modelSKL import ModelSKL
from modelSKL import ModelSKL
from modules.mdlrankeval import ModelRankEval
from modules.score import ScoreRank


class ModelRkEvSKL(ModelSKL, ModelRankEval):
    
    def _rawRankEval(self, X:NDArray, y:NDArray) -> Tuple[float, float]:   
        yc  = self._toCategorical(y)
        acc = self._model.score(X, yc)
        return 0, acc


    def _rankEval(self, X:NDArray, y:NDArray, evalProg:EvaluationProgess, 
                 rank:ScoreRank, batches:int, batchSize:int) -> Tuple[float,float,List[float]]:
        '''
        Subclass dependant evaluation part
        '''
        #Initialize
        batch = 0
        acumAcc = 0
        accHist = []
        samples = X.shape[0]
        params  = self.modelCountParams()    
        evalProgressUpdate = EvalProgressUpdate(evalProg, batches, rank, params)
        
        #Compute accuracy by batch        
        for start in range(0, samples, batchSize):
            end = min(start + batchSize, samples)
            Xn = X[start:end]
            yn = y[start:end]
            loss, curAcc = self._rawRankEval(Xn, yn)

            residue = end-start
            if residue == batchSize:
                acumAcc += curAcc
                acumMean = acumAcc / (batch+1)
            else:
                leadMean = (acumAcc/batch)*(samples-residue)
                lastMean = curAcc * residue
                acumMean = (leadMean + lastMean) / samples

            #Append to accuracy history
            accHist.append(acumMean)

            #Update evaluation progress
            evalProgressUpdate.update(acumMean, batch)
            batch += 1
    
        acc = acumMean
        return 0, acc, accHist
    
    
    #def _toCategorical(self, y:NDArray) -> NDArray:
    #    '''
    #    Convert y to categorical if one_hot encoded
    #    If y has more than one column assumes one_hot encoded
    #    '''
    #    if len(y.shape) > 1:    #do not use AND because y.shape may have only 1 element
    #        if y.shape[1] > 1:
    #            return np.argmax(y, axis=1) 
    #    else:
    #        return y
        