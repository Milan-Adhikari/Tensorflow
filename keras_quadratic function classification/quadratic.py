from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('quadratic_train.csv')
np.random.shuffle(train_data.values)

# checking the graph
# plt.scatter(data.x,data.y)
# plt.show()
# conclusion: the graph turns out to be quadratic

# sequential / network design
model = keras.Sequential([
    keras.layers.Dense(32,input_shape = (2,),activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')])

# compiling the model
model.compile(optimizer='adam',
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics='accuracy')

# fiting the data to the model

x = np.column_stack((train_data.x.values,train_data.y.values))
model.fit(x,train_data.color.values,batch_size=4,epochs=10)

# EVALUATING MODEL
print('Evaluating Result')
test_data = pd.read_csv('quadratic_test.csv')
test_x = np.column_stack((test_data.x.values,test_data.y.values))
model.evaluate(test_x,test_data.color.values)