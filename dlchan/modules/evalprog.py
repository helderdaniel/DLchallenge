# Used to pass values from Evaluator to routes during evaluation
# ML/DL run challenge
#
#v0.1 jul 2022, v0.2 nov 2024
#hdaniel@ualg.pt
#

from typing import *
import threading


class EvaluationProgess:
    '''
    To pass values to from Evaluator to routes during evaluation
    '''
    def __init__(self, tag:str='', evalAcc:List[Tuple[float, float]]=[], pos:int=-1, 
                       batch:Tuple[int, int]=(0,0),
                       complete:bool=False, hist:List[float]=[]) -> None:
        
        self._lock = threading.Lock()
        self.__update(tag, evalAcc, pos, batch, complete, hist)
                    
    ############################
    #      general update      #
    ############################
    def __update(self, tag:str='', evalAcc:List[Tuple[float, float]]=[], pos:int=-1, 
                      batch:Tuple[int, int]=(0,0),
                      complete:bool=False, hist:List[float]=[]) -> None:
        with self._lock:
            self.__tag      : str = tag
            self.__evalAcc  : List[Tuple[float, float]] = evalAcc # (x=batch, y=acc)
            
            self.__pos      : int = pos       # current position in the score table
            self.__batch    : Tuple[int, int] = batch     # current processed batch
            self.__complete : bool= complete  # evaluation process completed
                                            # all batches processed 
                                            # no more updates on this object
                                            # until the evaluatio of a new model
            self.__topHist : List[float]=hist


    ############################
    #      Write functions     #
    ############################
    def clear(self) -> None:
        #todo: needed to explicitly set clear arguments
        #the default values are not being passed to the evalAcc if omitted
        #it keeps the previous values. WHY??
        self.__update('', [], -1, (0,0), False, [])


    def new(self, modeltag) -> None:
        #todo: needed to explicitly set clear arguments
        #the default values are not being passed to the evalAcc if omitted
        #it keep the previous values. WHY??
        self.__update(modeltag, [], -1, (0,0), False, [])


    def add(self, evalAcc:List[Tuple[float, float]]=None, pos:int=None, 
                  batch:Tuple[int, int]=()) -> None:
        with self._lock:
            self.__evalAcc.append(evalAcc)
            self.__batch = batch
            self.__pos = pos


    def setComplete(self, c:bool) -> None:
        with self._lock:
            self.__complete = c

    def setPosition(self, pos:int) -> int:
        with self._lock:
            self.__pos = pos
    

    ############################
    #      Write functions     #
    ############################
    def position(self) -> int:
        with self._lock:
            return self.__pos
    
    def complete(self) -> bool:
        with self._lock:
            return self.__complete 
    
    def tag(self) -> bool:
        with self._lock:    
            return self.__tag

    def evalAcc(self) -> List[Tuple[float, float]]:
        with self._lock:
            return self.__evalAcc
    
    def batch(self) -> Tuple[int, int]:
        with self._lock:
            return self.__batch
