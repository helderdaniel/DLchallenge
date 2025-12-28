#Model Itype identification for ML/DL run challenge
#
#v0.1 nov 2024
#hdaniel@ualg.pt
#

from typing import *
#from modules.model import Model
from model import Model

class ModelSelect:

    def __init__(self, models:List[Model]=[]) -> None:
        self._models : List[Model] = models

    def fromFile(self, fn:str) -> Optional[None|Model]:
        for model in self._models:
            if model.valid(fn):
                return model.fromFile(fn)
        return None
    
    def invalidMsg(self) -> str:
        out:str=''
        for model in self._models:
            out += model.invalidMsg() + '!'
        return out  
