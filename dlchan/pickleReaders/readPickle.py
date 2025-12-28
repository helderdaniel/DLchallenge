#General pickle reader
#gives size of objects down to 2 levels
#no recursion

import pickle

file = "../data/trainDataset.pickle"

objects = []

#read all objects stored in pickle with pickle.dump()
with (open(file, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


#Print length of objects and length of all elements (down to one level, no recursion)
print("Number of objects in file: ", len(objects))
for obj in objects:
    if hasattr(obj, '__len__'): l = len(obj)
    else:                       l=1
    print("object length:", l)
    for o in obj:
        if hasattr(o, '__len__'): l = len(o)
        else:                       l=1
        print("  element length:", l)


'''
#Read just train dataset:

import pickle, numpy
from numpy.typing import NDArray

file = "data/trainDataset.pickle"

with open(file, 'rb') as f:
    X : NDArray = pickle.load(f)
    y : NDArray = pickle.load(f)

print(len(X))
print(len(y))
'''
