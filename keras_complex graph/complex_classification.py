import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dropout

# data
train = pd.read_csv('complex_train.csv')
np.random.shuffle(train.values)
#print(train.head())

# graph
# plt.scatter(train.x.values,train.y.values)
# plt.show()

# sequential
model = keras.Sequential([
    keras.layers.Dense(256,input_shape = (2,),activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')
])

# compiling model
model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

#fitting model
x = np.column_stack((train.x.values,train.y.values))
model.fit(x,train.color.values,batch_size=16,epochs=20)

# EVALUATING
print('Evaluating')
test = pd.read_csv('complex_test.csv')
x = np.column_stack((test.x.values,test.y.values))
model.evaluate(x,test.color.values)

