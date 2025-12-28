# Dataset reader/writer and manipulation
# version for ML/DL running challendge
# 7 set 2022, hdaniel@ualg.pt 

from scipy.io import loadmat
import numpy as np
from numpy.typing import NDArray
from typing import *
import tensorflow as tf


class Datastore:

    #Read a dataset from file system
    #Optional NOT supported in Colab:
    #   TypeError: unsupported operand type(s) for |: 'type' and '_GenericAlias'
    #
    #def read(cls, fname:Optional[str|List[str]], featuresName:str='X', 
    #             labelsName:str='Y') -> Tuple[NDArray, NDArray]:
    @classmethod
    def read(cls, fname, featuresName:str='X', 
                 labelsName:str='Y') -> Tuple[NDArray, NDArray]:
        '''
        Reads a dataset stored as a file in Matlab *.mat format with:
        a MxN matrix with the features and
        a Mxp matrix with the labels, one hot encoded

        M is the number of samples
        N is the number of features (or data points) in each sample 
        p is the number of classes

        fname is a string with a filename or a List of strings with filenames.
        returns the feature and label as numpy arrays sorted by fault classes
        '''
        if isinstance(fname, str):
            fname = [fname]

        X=None
        y=None
        for f in fname:
            ds=loadmat(f)
            Xc = ds[featuresName] # X is expected to be float32
            yc = ds[labelsName]   # Y is expected to be one hot encoded uint8
            if X is None:
                X = Xc
                y = yc
            else:
                X = np.concatenate((X, Xc), axis=0)
                y = np.concatenate((y, yc), axis=0)

        #Sort by faults
        if len(fname) > 1:
            #Get class number from one_hot
            classNumbers = np.argmax(y, axis=1)
            #get class number sorted indexes
            idxs = np.argsort(classNumbers)
            #Sort by indexes
            X = X[idxs]
            y = y[idxs]

        return X, y


    #Shuffle and split dataset 
    #making sure each class have same number of samples
    #for trainning and testing
    @classmethod
    def split(cls, X:NDArray, y:NDArray, numClasses:int, split:float, 
             xType:str='float32', yType:str='uint8') -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        '''
        Shuffle and split dataset defined in X and y:
        X is MxN matrix with the features and
        y is Mx1 matrix with the labels

        M is the number of samples
        N is the number of features (or data points) in each sample 

        It is assumed that the samples with the labels are ordered and not mixed:
        First are all the samples with the first label
        then are all the samples with the second label
        (...)

        Each set of of class labes are shuffled in place, keeping the order above

        The data set is then split:

        Train samples = Samples * split 
        Test samples  = Samples - TrainSamples

        returns the feature and label train and test sets as numpy arrays:

        xTrain, yTrain, xTest, yTest 
        '''
        #Separate datasets equally by fault classes
        sinalLength  = int(X.shape[1])
        samplesClass = int(X.shape[0] / numClasses)
        splitPoint   = int(samplesClass*split)
        
        xTrain = []
        yTrain = []
        xTest  = []
        yTest  = []

        for i in range(numClasses):       
            #slice by fault class
            st = i*samplesClass
            sp = st + splitPoint
            end = (i+1)*samplesClass
            
            #shuffle in place each fault data before slice train/set
            p = np.random.permutation(samplesClass)
            X[st:end, :] = X[st+p, :]
            y[st:end, :] = y[st+p, :]
                
            xTrain.append(X[st:sp, :])
            yTrain.append(y[st:sp, :])
            xTest.append (X[sp:end, :])
            yTest.append (y[sp:end, :])

        #Reshape matrices to proper sizes and define data types
        xTrain = np.array(xTrain, dtype=xType)
        yTrain = np.array(yTrain, dtype=yType)
        xTest  = np.array(xTest,  dtype=xType)
        yTest  = np.array(yTest,  dtype=yType)
        xTrain = xTrain.reshape(numClasses*splitPoint, sinalLength)
        yTrain = yTrain.reshape(numClasses*splitPoint, numClasses)
        xTest  = xTest.reshape (numClasses*(samplesClass-splitPoint), sinalLength)
        yTest  = yTest.reshape (numClasses*(samplesClass-splitPoint), numClasses)
    
        #Samples are shuflled inside class
        #but classes are in order
        #Need to shuffle in fit (model.fit(shuffle=True))
        #which is better since it shuffle before each epoch

        return xTrain, yTrain, xTest, yTest 



    #Read and shuffle dataset 
    #making sure each class have same number of samples
    #for trainning and testing
    @classmethod
    def readSplit(cls, fname:str, numClasses:int, trainPer:float, 
             featuresName:str='X', labelsName:str='Y',
             xType:str='float32', yType:str='uint8') -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        '''
        Reads shuffle and split a dataset stored as a file in Matlab *.mat format with:
        a MxN matrix with the features and
        a Mx1 matrix with the labels

        M is the number of samples
        N is the number of features (or data points) in each sample 

        It is assumed that the samples with the labels are ordered and not mixed:
        First are all the samples with the first label
        then are all the samples with the second label
        (...)

        Each set of of class labes are shuffled in place, keeping the order above

        The data set is then split:

        Train samples = Samples * trianPer 
        Test samples  = Samples - TrainSamples

        returns the feature and label train and test sets as numpy arrays:

        xTrain, yTrain, xTest, yTest 
        '''
        #read dataset
        X, y = cls.read(fname, featuresName, labelsName)
        #shuffle inside classes and split
        return cls.split(X, y, numClasses, trainPer, xType, yType)



    #Reshape a dataset, number of elements must remain the same 
    @classmethod
    def reShape(cls, X:NDArray, y:NDArray, numChannels:int, 
                     newPoints:int) -> Tuple[NDArray, NDArray]:
        '''
        Reshapes a dataset with numSamples to a new length sample given by newPoints.
        The new number of samples is old(numSamples)/(newPoints/oldPoints) 
        where X are Features and y are Labels one hot encoded

        pre: X and y must be 2D arrays with the same number of rows and
             the number of samples x number of points must remain the same

        Raises:
             Runtime error if the number of samples x number of points 
             do not remains the same after reshaping
        '''
        oldSamples: int   = X.shape[0]
        oldPoints : int   = int(X.shape[1]/numChannels) #Should be int
        factor    : float = newPoints/oldPoints
        newSamples: int   = int(oldSamples/factor)
        if (oldSamples * oldPoints != newSamples * newPoints):
            raise RuntimeError("Number of samples x number of points must be the same after reshaping")
        
        #Reshape from numSamples x oldPoints 
        #to numSamples/factor x newPoints
        newX=None
        for c in range(numChannels):
            #split channel
            Xch = X[:, c*oldPoints:(c+1)*oldPoints]

            #reshape
            Xch = Xch.reshape(newSamples,newPoints)

            #re join channels
            if newX is None:
                newX = Xch
            else:
                newX = np.concatenate((newX, Xch), axis=1)

        yf = int (factor)
        if yf==0:   #extend rows
            newy = np.repeat(y, int(1/factor), axis=0)
        else:       #reduce rows
            newy = y[::yf]

        return newX, newy



    #Shapes a dataset, using the first elements in each class of 
    #the old dataset that fills the new shape
    @classmethod
    def shape(cls, X:NDArray, y:NDArray, numClasses:int, numChannels:int, 
                   newPoints:int) -> Tuple[NDArray, NDArray]:
        '''
        Reshapes a dataset with numSamples to a new length sample given by newPoints.
        The new number of samples is old(numSamples)/(newPoints/oldPoints) 
        where X are Features and y are Labels one hot encoded

        If after reshaping the Number of samples x number of points is not the same,
        it fills the new shape with the first elements in each class of the old dataset.

        pre: X and y must be 2D arrays with the same number of rows
        '''
        numClasses = 7
        oldSamples    :int   = X.shape[0]
        oldClsSamples :int   = int(oldSamples/numClasses)
        oldPoints     :int   = int(X.shape[1]/numChannels) #Should be int
        factor        :float = newPoints/oldPoints
        newClsSamples :int   = int(oldClsSamples/factor)
        newSamples    :int   = newClsSamples*numClasses
       
        #Reshape by class
        newX=None
        newy=None
        for cls in range(numClasses):
            #get class
            clsStart = cls*oldClsSamples
            clsEnd   = clsStart+oldClsSamples
            Xcls = X[clsStart:clsEnd,:]

            newCls=None
            for ch in range(numChannels):
                #get channel
                chStart = ch*oldPoints
                chEnd   = chStart+oldPoints
                Xch = Xcls[:, chStart:chEnd]

                #reshape
                Xch = Xch.flatten()
                Xch = Xch[:(newClsSamples*newPoints)]
                Xch = Xch.reshape(newClsSamples,newPoints)

                #re join channels
                if newCls is None:
                    newCls = Xch
                else:
                    newCls = np.concatenate((newCls, Xch), axis=1)

            #re join classes
            if newX is None:
                newX = newCls
            else:
                newX = np.concatenate((newX, newCls), axis=0)

            #From y
            classTag = y[clsStart,:]
            classTag = classTag.reshape(1,classTag.shape[0])  #to support also categorical encoded
            newTags = np.repeat(classTag, newClsSamples, axis=0)
            if newy is None:
                newy = newTags
            else:
                newy = np.concatenate((newy, newTags), axis=0)

        return newX, newy



    #Extract points from channels specified
    @classmethod
    def extractChan(cls, X:NDArray, chSet:List[int], chPoints:int) -> NDArray:
        '''
        Return x array with only the columns, corresponding to points defined for the channels in chSet
        PRE: All channels have the same number of points
             max(chSet+1)*chPoints <= x.shape[1]            
        '''
        xView = None
        for i in chSet:
            begin = int(i    *chPoints)
            end   = int((i+1)*chPoints)
            xv = X[:, begin:end]
            if xView is None:   xView = xv
            else:               xView = np.concatenate((xView, xv), axis=1)

        return xView



    #IF channel data is on same vector,
    #Split channels in different vectors and stack them
    #This is NOT the case of images: grayscale have only 1 channel
    #                                RGB have 3 channels, already split
    #                                
    @classmethod
    def splitStackChan(cls, X:NDArray, nChan:int, noMaps:int=1) -> NDArray:
        '''
        Split channels in nChan different vectors and stack them
        PRE: All channel points must be in the same vector
            and
             All channels must have the same number of points 
            and
             channels must be in the last axis of the array      
        '''
        curShape = list(X.shape)
        lastShape = curShape[len(curShape)-1]
        splitShape = int(lastShape/nChan)
        newShape  = curShape[:-1]
        newShape.append(splitShape)
        newShape.append(nChan)

        #Todo: not tested, only needed with CNN 2D RGB
        newShape.append(noMaps) # Not needed, since the channels are stacked in a 2D array as a grayscale image
                                # If it was an RGB image this must be 3
        xView = np.reshape(X, tuple(newShape))

        return xView

    

    #Convert to TF Datasets for efficient GPU processing
    #https://www.tensorflow.org/guide/data#using_tfdata_with_tfkeras
    @classmethod
    def toTFDataset(cls, trainSet:Tuple[NDArray,NDArray], 
                         testSet :Tuple[NDArray,NDArray], 
                         evalSet :Tuple[NDArray,NDArray], 
                         batchSize:int) -> Tuple[NDArray,NDArray,NDArray]:

        dsTrain = tf.data.Dataset.from_tensor_slices(trainSet)
        dsTrain = dsTrain.shuffle(trainSet[0].shape[0]).batch(batchSize).cache().prefetch(tf.data.AUTOTUNE)

        #No need to shuffle for testing and evaluation
        dsTest = tf.data.Dataset.from_tensor_slices(testSet)
        dsTest = dsTest.batch(batchSize).cache().prefetch(tf.data.AUTOTUNE)
        dsEval = tf.data.Dataset.from_tensor_slices(evalSet)
        dsEval = dsEval.batch(batchSize).cache().prefetch(tf.data.AUTOTUNE)

        return dsTrain, dsTest, dsEval


    #@classmethod
    #def toTFDataset2(cls, xTrain, yTrain, xTest, yTest, xEval, yEval, batchSize):
    #    cls.toTFDataset((xTrain, yTrain), (xTest, yTest), (xEval, yEval), batchSize)