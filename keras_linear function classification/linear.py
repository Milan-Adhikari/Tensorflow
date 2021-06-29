import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_math_ops import mod

train_data = pd.read_csv('linear_train.csv')
np.random.shuffle(train_data.values)

#print(train_data.head())

# sequential
model = keras.Sequential([
    keras.layers.Dense(8,input_shape =(2,),activation='relu'),
    keras.layers.Dense(2,activation='softmax')])

# compilation of model
model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# fitting the model
x = np.column_stack((train_data.x.values,train_data.y.values))
model.fit(x,train_data.color.values,batch_size=4,epochs=4)

# testing on test data
test_df = pd.read_csv('linear_test.csv')
test_x = np.column_stack((test_df.x.values,test_df.y.values))
model.evaluate(test_x,test_df.color.values)