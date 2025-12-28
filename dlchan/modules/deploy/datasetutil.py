#Dataset utilities
#
#jan 2025
#hdaniel@ualg.pt
#

from typing import *
import numpy as np
from numpy.typing import NDArray


class DatasetUtil:
    
    @staticmethod
    def toCategorical(y:NDArray) -> NDArray:
        '''
        Convert y to categorical if one_hot encoded
        '''
        if len(y.shape) > 1:    #do not use AND because y.shape may have only 1 element
            if y.shape[1] > 1:
                return np.argmax(y, axis=1) 
        else:
            return y

    @staticmethod
    def shuffle(X:NDArray, y:NDArray) -> Tuple[NDArray, NDArray]:
        '''shuffle dataset'''
        p = np.random.permutation(X.shape[0])
        return X[p], y[p]

