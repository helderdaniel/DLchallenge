import keras
from keras import Sequential, layers

# https://keras.io/api/models/model/
# https://keras.io/api/layers/

#=============================================
# 'Sequential' API
#
# model.summary() does not show Input layer
# can be obtained with: model.input_shape
# or 
# shown the input and output shape for every layer, 
# being the model input layer shape the input shape of the first layer shown
#
#=============================================

model:keras.Sequential = Sequential(layers=[
    layers.Input(shape=(64, 64, 3), name='in'),
    layers.Conv2D(32, (3, 3), name='c2d'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='p2d'),
    layers.Dense(units=10, activation='softmax', name='d')
])

model.summary()

print("Model input shape: ", model.input_shape)
for layer in model.layers:
    print(type(layer).__name__, "(",layer.name,")")
    print("Input shape: ", layer.input.shape)
    print("Output shape:", layer.output.shape)
    print("Parameters:  ", layer.count_params())
print('Total params:', model.count_params())

#=============================================
# 'Functional' API
#
# model.summary() does show Input layer
# no need to get it another way
# 
#=============================================

inputs = layers.Input((64, 64, 3), name='in')
c = layers.Conv2D(32, (3, 3), name='c2d')(inputs)
p = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='p2d')(c)
outputs = layers.Dense(units=10, activation='softmax', name='d')(p)
model:keras.Model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

for layer in model.layers:
    print(type(layer).__name__, "(",layer.name,")", layer.output.shape, layer.count_params())
print('Total params:', model.count_params())