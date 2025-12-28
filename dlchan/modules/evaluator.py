#Evaluator for ML/DL run challenge
#
#v0.2 aug 2024, v0.3 nov 2024
#hdaniel@ualg.pt
#

import shutil
import os, time, pickle, gc, math
from typing import *
from numpy.typing import NDArray
from filelock import FileLock
from modules.evalhist import EvalHist
#from modules.model import Model
from model import Model
from modules.mdlrankeval import ModelRankEval
#from modules.modelsel import ModelSelect
from modelsel   import ModelSelect
from modules.runqueue import RunQueue
from modules.evalprog import EvaluationProgess
from modules.score import ScoreTable, ScoreRank
from modules.datastorex import Datastore
from modules.flasklog import FlaskLog

class Evaluator:
    '''
    Evaluate model form RUNQUEUE using dataset X, y
    and store score in SCORETABLE
    '''

    def __init__(self, modelSel:ModelSelect, runQueue:RunQueue, evalProg:EvaluationProgess,
                 scoreTable:ScoreTable, scoreLock:FileLock, evalHist:EvalHist, uploadFolder:str,
                 bestFolder:str, evalDatasetFN:str, classes:int, channels:int, maps:int, 
                 shuffle:bool=False, seed:int=None) -> None:
        self._modelSel     = modelSel
        self._runQueue     = runQueue
        self._evalProg     = evalProg
        self._scoreTable   = scoreTable
        self._scoreLock    = scoreLock
        self._evalHist     = evalHist
        self._uploadFolder = uploadFolder
        self._bestFolder   = bestFolder
        self._channels     = channels
        self._classes      = classes
        self._maps         = maps
        self._shuffle      = shuffle
        self._seed         = seed
        
        
        with open(evalDatasetFN, 'rb') as f:
            
            #Check if dataset has 2 variables X, y 
            #or just one dict with {X, y}
            try:
                self._X : NDArray = pickle.load(f)
                self._y : NDArray = pickle.load(f)
            except:
                self._y = self._X['Y'] # get Y first and X later
                self._X = self._X['X'] # then can rewrite self._X

        #Try to determine number of classes from Y columns, if one_hot encoded.
        #If Y columns is 1 (categorical), then keep value specified in 'dlchan.cfg' file
        if self._y.shape[1] > 1:
            self._classes  = self._y.shape[1]


    def evaluate(self, model:ModelRankEval, X:NDArray, y:NDArray, 
                 rank:ScoreRank, batchSize:int=32) -> Tuple[float,float]:
        '''
        evaluates  model on dataset (X, y) by batches and returns:
        (eval loss, eval final accuracy, eval acuraccy history by batch)
        '''

        #compute number of batches needed
        samples  = X.shape[0]
        nBatches = int(math.ceil(samples/batchSize))

        #Evaluate with subclass specific evaluator
        (loss, acc, accHist) = model.rankEval(X, y, self._evalProg,
                                              rank, nBatches, batchSize,
                                              self._shuffle, self._seed)
        gc.collect()
        return (loss, acc, accHist)



    def evaluatorThread(self, period:int)->None:
        '''
        Periodic Thread that checks RUNQUEUE every period seconds
        IF filled run evaluate()
        '''

        #Shared EvaluationProgress instance
        #Make sure only one model is evaluated at a time
        while True:
            time.sleep(period)   #Wait some time before start evaluating another model: let them see it blinking
                                 #(no timer needed, does not need to be that accurate)
            if self._evalProg.complete(): 
                self._evalProg.setComplete(False) #stop flashing score table
                FlaskLog.warning(f'stop blinking')
            
            with self._scoreLock:
                filename = self._runQueue.get()
                
                if filename is not None:
                    FlaskLog.warning(f'Evaluating model: {filename}')
                    modelFN = os.path.join(self._uploadFolder, filename)
                    modelTag = filename.rsplit('.', 1)[0]
                    self._evalProg.new(modelTag)

                    # Read ScoreTable rank acc/params
                    # this rank is used with each evaluation batch to compute
                    # current position in the score table
                    #
                    # reading here is faster than reading in each batch
                    rank = ScoreRank(self._scoreTable)

                    # get model config 
                    model:Model = self._modelSel.fromFile(modelFN)
                    if model is None:
                        #todo: this happens for valid models, but only sometimes
                        #Why? unsyncing threads?
                        #but views use ad(atomic) to add the model
                        #and this thread use get(atomic) to get model
                        FlaskLog.warning(f'error loading model stored in: {filename}')
                        continue
                    else:
                        FlaskLog.warning(f'model type is: {model.name()}')
                    
                    inLayerShape = model.inputShape()
                    modelDim     = model.dim()
                    params  = model.modelCountParams()    #Model total parameters
                    
                    #Reshape dataset to model input layer
                    evaluate = True
                    if   (modelDim == 1):     # 1D Model
                        X, y = Datastore.shape(self._X, self._y, self._classes, self._channels, int(inLayerShape[0]/self._channels))
                    elif (modelDim == 2):     # 2D Model
                        X, y = Datastore.shape(self._X, self._y, self._classes, self._channels, int(inLayerShape[0]/1))
                        X = Datastore.splitStackChan(X, self._channels, self._maps)
                    else:
                        #todo how to send message to UI?
                        FlaskLog.warning(f'Model {modelTag} input layer is not 1D or 2D')
                        evaluate = False
                        #raise RuntimeError('Model input layer is not 1D or 2D')
                    FlaskLog.warning(f'dataset reshaped to X:{X.shape} y:{y.shape}, for model {modelTag}')

                    if evaluate:
                        loss, acc, accHist = self.evaluate(model, X, y, rank)
                        FlaskLog.warning(f'evaluated accuracy: {acc:.5f}')

                        isUpdated = False
                        if acc >= 0:
                            isUpdated = self._scoreTable.update(modelTag, acc, loss, params, accHist)
                            FlaskLog.warning(f'score table updated: {isUpdated}')
                            self._evalHist.add(modelTag, acc, loss, params, isUpdated)
                            FlaskLog.warning(f'added to evaluation history: {modelTag}')
                        else:
                            FlaskLog.warning(f'error evaluating model stored in: {filename}')
                                                                
                        # If model is better move it to best model folder
                        if isUpdated:
                            bestFN = os.path.join(self._bestFolder, filename)                        
                            shutil.move(modelFN, bestFN)
                            FlaskLog.warning(f'best model moved to: {bestFN}')
                        
                        # If modelTag exists and acc is lower than the one registed 
                        # score table was not updated, but the position is set to 1 below the
                        # current modelTag entry, because it is lower
                        #
                        # In this case the current position of the modelTag is recovered
                        # and used to blink the score table
                        else:
                            curPos = self._scoreTable.findPositionByTag(modelTag)
                            self._evalProg.setPosition(curPos)
                            os.remove(modelFN)
                                                
                        self._evalProg.setComplete(True)  #blink
                        FlaskLog.warning(f'start blinking')

