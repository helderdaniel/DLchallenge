#Evaluation history for ML/DL run challenge
#
#0.1 Nov 2024
#hdaniel@ualg.pt
#

from datetime import datetime
import os
from filelock import FileLock
from typing import *


class EvalHist:

    def __init__(self, histFN:str, lock:FileLock) -> None:
        self._histFN = histFN
        self._lock = lock
        
        
    def add(self, tag:str, acc:float, loss:float, param:int, best:bool) -> None:
        '''
        Atomic append to file
        '''
        with self._lock:
            self.__unlockedAdd(tag, acc, loss, param, best)


    def __unlockedAdd(self, tag:str, acc:float, loss:float, param:int, best:bool) -> bool:
        
        #add new entry or update if it exists
        #Convert acc to percentage
        u = '1' if best else '0'
        entry = u +', '+ tag +', '+ str(acc*100) +', '+ str(loss) +', '+ str(param) +', '+ datetime.now().strftime('%Y-%m-%d %H:%M:%S') +'\n'
         
        with open(self._histFN, 'a') as f:
            f.write(entry)


    def read(self) -> str:
        '''
        Atomic read file
        '''
        with self._lock:
            return self.__unlockedRead()


    def __unlockedRead(self) -> str:
        
        with open(self._histFN, 'r') as f:
            return f.read()

#Test it
if __name__ == '__main__':
    fn = 'evalhist.txt'
    lock = 'evalhist.lock'
    
    h = EvalHist(fn, FileLock(lock))
    
    #fill history
    h.add('test', 0.5, 0.5, 100, True)
    h.add('test34', 1.5, 0.5, 100, False)
    h.add('test', 0.5, 2.5, 100, True)

    #show it
    print(h.read())

    #clean temp files
    os.remove(fn)
    os.remove(lock)