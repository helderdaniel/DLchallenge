#Score table for ML/DL run challenge
#
#v0.1 jul 2022, 0.2 Nov 2024
#hdaniel@ualg.pt
#

from datetime import datetime
import pickle, os
from filelock import FileLock
from typing import *
import numpy as np


class ScoreTable:

    def __init__(self, tableFN:str, lock:FileLock) -> None:
        self._tableFN = tableFN
        self._lock = lock
        self._table = {}
        self._topHist = []
        self._updateDate  = datetime.now()  #Not needed filled by __read() or __write()

        #read it or create it if does not exist
        try:
            with self._lock:
                self.__read()
        except:
            self.__write()

        #create if it does not exist
        #if not os.path.exists(self._tableFN):
        #    self.__write()



    ############################
    #   public score access    #
    ############################

    def top(self): 
        self.__read() 
        if len(self._table) > 0:
            name = list(self._table.keys())  [0]
            val =  list(self._table.values())[0]
        else:
            name = ''
            val  = ''
        return [name, self._topHist, val]
    

    def update(self, tag:str, acc:float, loss:float, param:int, accHist:List[float]) -> bool:
        '''
        Atomic update file
        '''
        with self._lock:
            return self.__unlockedUpdate(tag, acc, loss, param, accHist)


    def __unlockedUpdate(self, tag:str, acc:float, loss:float, param:int, accHist:List[float]) -> bool:
        save = True
        #get current table from file
        self.__unlockedRead()

        #add new entry or update if it exists
        #Convert acc to percentage
        entry = [acc*100, loss, param, datetime.now()]  #, accHist] #To register history for all submissions

        if tag not in self._table:      #Add if not present
            self._table[tag] = entry
        else:                           #update with best accuracy/ fewer params
            old = self._table[tag]

            #handle param = None
            e2 = entry[2] if entry[2] is not None else 0
            o2 = old[2]   if old[2]   is not None else 0

            if entry[0] > old[0] or \
                (entry[0] == old[0] and e2 < o2):
                    self._table[tag] = entry
            else:
                save = False

        #sort by high acc and less param,
        #update top runner evaluation history and
        #save updated score table file
        if save:
            self.__sort()

            if list(self._table.keys())[0] == tag:  
                #Convert top history accuracy to percentage [0-100]
                topHist = [i*100 for i in accHist]

                #rescale top history x-axis to [0-100]
                x = np.linspace(0, 100, len(topHist))
                self._topHist = [(x,y) for x,y in zip(x,topHist)]

            self._updateDate = datetime.now()
            self.__unlockedWrite()

        return save


    def __sort(self):
        '''sort score table'''
        #self._table = {k: v for k, v in sorted(self._table.items(), key=lambda x:(-x[1][0],x[1][2]))}
        #todo handle params = none as inf? Is it the better way? Consider too complex models if no params reported?
        self._table = {k: v for k, v in sorted(self._table.items(), key=lambda x: (-x[1][0], x[1][2] if x[1][2] is not None else float('inf')))}


    def rank(self) -> List[float]:
        '''
        return sorted list of rank accuracy from the highest to the lowest
        Note that score table is sorted each time it is updated
        operation is atomic
        '''
        with self._lock:
            return self.__unlockedRank()


    def __unlockedRank(self) -> List[Tuple[float,float]]:
        self.__unlockedRead()

        #get only acc and params
        l = [ (value[0], value[2])
                    for key, value in self._table.items() ]
        #no need to sort
        #it is sorted each time it is updated
        #l = sorted(l, key=lambda x:(-x[0],x[1])) 
        return l
        

    def get(self) -> Optional[List|None]:
        '''
        Atomic read table if updated after displayed
        Returns score table as a sorted list by accuracy and then parameters
        Note that score table is sorted each time it is updated
        '''
        with self._lock: 
            return self.__unlockedGet()


    def __unlockedGet(self) -> Optional[List|None]:
        self.__unlockedRead()

        #Select key(name), acc, params and history
        l = [ [key, item[0], item[2]] 
                    for key, item in self._table.items()] 

        #sort not needed, sorted when updated
        #l = sorted(l, key=lambda x:(-x[1],x[2])) 
        return l


    def findPositionByTag(self, tag:str) -> int:
        '''
        return position of tag in score table
        '''
        self.__read()
        try:
            return list(self._table.keys()).index(tag) + 1
        except:
            return -1
        

    def updateDate(self) -> datetime:
        '''
        return update date
        '''
        return self._updateDate

        #This way would read the file.
        #Usefull if some other evalutor updates the same file, but that is not the case
        #with self._lock:
        #    with open(self._tableFN, 'rb') as f:
        #        return pickle.load(f)
        

    ############################
    #   low level file access  #
    ############################
    def __read(self) -> None:
        '''read table with file lock'''
        with self._lock:
            self.__unlockedRead()


    def __unlockedRead(self) -> None:
        '''raw read table, no file lock'''
        with open(self._tableFN, 'rb') as f:
            self._updateDate = pickle.load(f)
            self._topHist    = pickle.load(f)
            self._table      = pickle.load(f)


    def __write(self) -> None:
        '''write table with file lock'''
        with self._lock:
            self.__unlockedWrite()

    def __unlockedWrite(self) -> None:
        '''raw write table, no file lock'''
        self._updateDate  = datetime.now()
        with open(self._tableFN, 'wb') as f:
            pickle.dump(self._updateDate, f)
            pickle.dump(self._topHist, f)
            pickle.dump(self._table, f)



class ScoreRank:
    '''
    Auxiliary class to find rank by accuracy and params
    '''
    def __init__(self, scoreTable:ScoreTable) -> None:
        self._rank = scoreTable.rank()


    def findPositionByAccPar(self, acc:float, params:int) -> int:
        '''
        return position of acc/param in score table
        '''  
        #handle param = None
        params  = params if params is not None else 0

        pos = 1
        for a in self._rank:

            #handle param = None
            a1 = a[1]   if a[1]   is not None else 0
            
            if  a[0] < acc or \
                (a[0] == acc and a1 >= params):
                break
            pos += 1 

        return pos