#Model operations for ML/DL run challenge
#
#v0.1 jul 2022, v0.2 aug 2024, v0.3 nov 2024
#v0.4 jan 2025
#hdaniel@ualg.pt
#

from abc import abstractmethod
import keras
from typing import *
from numpy.typing import NDArray
from modules.evalprog import EvaluationProgess
from modules.evalprogupdate import EvalProgressUpdate
#from modules.modelKeras import ModelKeras
from modelKeras import ModelKeras
from modules.mdlrankeval import ModelRankEval
from modules.score import ScoreRank


class ModelRkEvKeras(ModelKeras, ModelRankEval):

    def _rawRankEval(self, X:NDArray, y:NDArray, batchSize:int=32, callbacks:keras.callbacks=[]) -> Tuple[float, float]:
        
        #Ignore metrics defined in model and
        #use just accuracy to evaluate the model
        self._model.compile(loss=self._model.loss, metrics=self._metrics)
    
        loss, acc = self._model.evaluate(X, y, batch_size=batchSize, callbacks=callbacks, verbose=0)
        return loss, acc


    def _rankEval(self, X:NDArray, y:NDArray, evalProg:EvaluationProgess, 
                 rank:ScoreRank, batches:int, batchSize:int=32) -> Tuple[float,float,List[float]]:
        '''
        Subclass dependant evaluation part
        '''
        #Initialize
        accHist = []
        params  = self.modelCountParams()    
        callbacks = [ EvalProgressUpdateCB(accHist, evalProg, batches, rank, params) ]
        
        loss, acc = self._rawRankEval(X, y, batchSize, callbacks)
    
        return loss, acc, accHist


########################
#      Callbacks       #
########################

class EvalProgressUpdateCB(keras.callbacks.Callback):
    '''
    Generate evaluation progress

    update accList with the accuracy of each batch
    '''
    def __init__(self, accList:List[float], evalProg:EvaluationProgess, batches:int, rank:ScoreRank, params:int) -> None:
        self._accList = accList
        self._evalProgressUpdate = EvalProgressUpdate(evalProg, batches, rank, params)

    def on_test_batch_end(self, batch, logs=None):
        curAcc = logs["accuracy"]
        self._accList.append(curAcc)

        self._evalProgressUpdate.update(curAcc, batch)
