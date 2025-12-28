#read dataset from reader
#gives info on dataset size
import pickle, numpy
from numpy.typing import NDArray

fname = "data/evalDataset.pickle"

with open(fname, 'rb') as f:
    X : NDArray = pickle.load(f)
    y : NDArray = pickle.load(f)

print("Number of samples:",     X.shape[0]) # or y.shape[0]
print("Points in each sample:", X.shape[1])
print("Fault classes:",         y.shape[1])
