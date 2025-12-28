# Update EvaluationProgess instance during evaluation
# ML/DL run challenge
#
#v0.1 nov 2022
#hdaniel@ualg.pt
#

from modules.evalprog import EvaluationProgess
from modules.score import ScoreRank


class EvalProgressUpdate:
    '''
    Auxiliary class to update evaluation progress
    '''
    def __init__(self, evalProg:EvaluationProgess, batches:int, rank:ScoreRank, params:int) -> None:
        self._evalProg = evalProg
        self._batches  = batches
        self._rank     = rank
        self._params   = params


    def update(self, curAcc:float, curBatch:int) -> None:
        
        #current progress bar (may need int) or x-axis position
        curProg = (curBatch+1)/self._batches*100 #convert to range 0-100

        #convert current batch accuracy to percentage
        curAccp  = curAcc * 100 #convert to percentage
        
        #set current batch accuracy rank position
        pos = self._rank.findPositionByAccPar(curAccp, self._params)

        #add current batch evaluation to list
        self._evalProg.add((curProg, curAccp), pos, (curBatch, self._batches))
        