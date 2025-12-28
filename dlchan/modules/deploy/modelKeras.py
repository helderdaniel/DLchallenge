#Model operations for ML/DL run challenge
#
#v0.1 jul 2022, v0.2 aug 2024, v0.3 nov 2024
#v0.4 jan 2025
#hdaniel@ualg.pt
#

import keras, io
import numpy as np
from typing import *
from numpy.typing import NDArray
from model import Model



class ModelKeras(Model):

    _metrics=['accuracy']
    _extension='keras'

    def __init__(self, name:str='', mdl:keras.Model=None) -> None:
        super().__init__(name)
        self._model:keras.Model = mdl                

    def config(self, epochs:int=1, threads:int=1, verbose:int=0,
               callbacks:List=[], lr:float=0.001) -> None:  
        super().config(epochs, threads, verbose)
        self._callbacks = callbacks
        self._initialLR = lr

    def fileExtension(self) -> str:
        return self._extension
    
    def load(self, fn:str) -> None:
        self._model = keras.models.load_model(fn)
    

    def save(self, fn:str) -> Any:
        self._model.save(fn)        
        
    def initialEpoch(self) -> int:
        return self._model._initial_epoch

    def dim(self) -> int:
        '''Infer model dim form, input shape:

            Dense layers: 
                (batch_size, input_size)
            
            1D recurrent layers:
                (batch_size, timesteps, features)
            
            1D convolutional layers:
            https://keras.io/api/layers/convolution_layers/convolution1d/
                
                (batch_size, steps, channels)
                (batch_size, inputLen, channels)

            2D convolutional layers need inputs as:
            https://keras.io/api/layers/convolution_layers/convolution2d/

                if using channels_last, the default:
                    (batch_size, height, width, channels)
                    (batch_size, inputLen[0], inputLen[1], channels)

        '''
        inShape = self.inputShape() #Strips batch size
        if len(inShape) == 3: return 2
        return 1

    def inputShape(self) -> Tuple:
        if self._model is None:
            return None
        else:
            return self._model.input_shape[1:] #Strip leading 'None' which will be replaced later by the batch size

    def ouputShape(self) -> Tuple:
        return self._model.output_shape[1:]    #Strip leading 'None' which will be replaced later by the batch size

    def modelCountParams(self) -> Tuple:
        return self._model.count_params()
    
    def valid(self, fn:str)->bool:
        '''
        Try to load a keras model file
        if it fails it is not a keras model

        Supports *.h5 or *.keras formats
        '''
        
        try:
            #self._rawLoad(fn)
            self.fromFile(fn)
        except:
            return False

        return True
    
    
    def invalidMsg(self)->str: 
        return 'invalid Keras model file, must be in HDF5 *.h5 or *.keras file format'


    def description(self) -> None: 
        #preserve colours
        self._model.summary()
        print(self._modelDetails())
    

    def __str__(self) -> str: 
        out = io.StringIO()
        self._model.summary(line_length=80, print_fn=lambda x: out.write(x + '\n'))
        return out.getvalue() + self._modelDetails()


    def _modelDetails(self) -> str:
        out = ''
        #inShape = self._model.get_layer(name=None, index=0).input.shape
        inShape  = self.inputShape()
        outShape = self.ouputShape()
        out += f'Input  layer shape: {inShape}\n'
        out += f'Output layer shape: {outShape}\n'

        #To use in site to identify loaded model DIM before evaluate
        out += 'Model dim: '
        if    self.dim()==1: out += '1D\n'
        elif  self.dim()==2: out += '2D\n'
        else: raise RuntimeError('Model dim < 1 or > 2: ', self.dim())
        return out


    def shuffle(self) -> None:
        '''randomize weights by permutating them in each layer'''
        weights = self._model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self._model.set_weights(weights)
            

    #def fit(self, X:NDArray, y:NDArray, batchSize:int=32, shuffle:bool=True) -> keras.callbacks.History:
    #    '''Run a simple trainer for model'''
    #    optimizer = keras.optimizers.Adam(learning_rate=self._initialLR) #seems best
    #    #optimizer=keras.optimizers.RMSprop(learning_rate=self._initialLR)
    #    #optimizer=keras.optimizers.Adagrad(learning_rate=self._initialLR) #seems far worst
    #    loss=keras.losses.categorical_crossentropy
    #    self._model.compile(optimizer=optimizer, loss=loss, metrics=self._metrics)
    #
    #    trainHist = self._model.fit(X, y, batchSize, self._epochs, self._verbose, 
    #                                self._callbacks, shuffle=shuffle)
    #    return trainHist


    def score(self, Xe:NDArray, ye:NDArray) -> float:
        loss, acc = self._model.evaluate(Xe, ye, verbose=1)
        return acc


    def predict(self, X:NDArray) -> NDArray:
        return self._model.predict(X, verbose=1)
