#Run queue for submited modules, for ML/DL run challenge
#
#v0.1 Aug 2022, v0.2 Nov 2024
#hdaniel@ualg.pt
#

from datetime import datetime
from operator import mod
import pickle
from filelock import FileLock
from typing import *
from modules.flasklog import FlaskLog


class RunQueue:

    def __init__(self, queueFN:str, lock:FileLock) -> None:
        self.__queueFN = queueFN
        self.__lock = lock

        #read it or create it if does not exist
        try:
            with self.__lock:
                self.__read()
        except:
            self.clear()
        
        #create if does not exist
        #if not os.path.exists(self.__queueFN):
        #    self.__write()


    ######################################
    #   low level funlocked file access  #
    ######################################
    def __read(self) -> None:
        with open(self.__queueFN, 'rb') as f:
            self.__queue = pickle.load(f)


    def __write(self) -> None:
        with open(self.__queueFN, 'wb') as f:
            pickle.dump(self.__queue, f)


    ############################
    #   public score access    #
    ############################

    def clear(self) -> None:
        '''
        clear queue
        '''
        with self.__lock:
            self.__queue = []
            self.__write()


    def get(self) -> Optional[str|None]:
        '''
        Atomic get and remove first from queue
        '''
        with self.__lock:
            return self.__unlockedGet()


    def __unlockedGet(self) -> Optional[str|None]:
        '''
        Get and remove first from queue
        '''
        self.__read()
        modelFN = None
        if len(self.__queue) > 0:
            modelFN, date = self.__queue.pop(0)
            self.__write()
        
        return modelFN

    
    def add(self, modelFN:str) -> None:
        '''
        Atomic add to end of queue
        '''
        with self.__lock:
            self.__unlockedAdd(modelFN)

    
    def __unlockedAdd(self, modelFN:str) -> None:
        '''
        Add to end of queue
        '''
        self.__read()
        date = datetime.now()
        self.__queue.append((modelFN, date))
        self.__write()


    def waiting(self, date:bool=False) -> List[str]:
        '''
        return list of waiting models
        operation is atomic
        '''
        with self.__lock:
            return self.__unlockedWaiting(date)


    def __unlockedWaiting(self, date:bool) -> List[str]:
        '''
        return list of waiting models
        '''
        self.__read()
        if date:
            queue = self.__queue
        else:
            queue = [modelFN for modelFN, date in self.__queue]
        return queue