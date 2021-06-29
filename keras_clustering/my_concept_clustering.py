from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('clusters_train.csv')
test_data = pd.read_csv('clusters_test.csv')
np.random.shuffle(train_data.values)

# encoding the data i.e label encoding color column
le = LabelEncoder()
train_data['color'] = le.fit_transform(train_data.color)
test_data['color'] = le.fit_transform(test_data.color)
#train_data.pop('color')
#test_data.pop('color')
print(train_data.head())
print(test_data.head())
#print(train_data.encoded_color.unique())
# encoded = pd.get_dummies(to_encode,drop_first=True)
# print(encoded.head())

# checking the graph
# plt.plot(train_data.x,train_data.y)
# plt.show()
# conclusion: the graph turns out to be clustered

# # sequential / network design
model = keras.Sequential([
    keras.layers.Dense(32,input_shape = (2,),activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(6,activation='sigmoid')])

# # compiling the model
model.compile(optimizer='adam',
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics='accuracy')

# # fiting the data to the model

x = np.column_stack((train_data.x.values,train_data.y.values))
model.fit(x,train_data.color.values,batch_size=16,epochs=10)

# EVALUATING MODEL
print('Evaluating Result')
test_x = np.column_stack((test_data.x.values,test_data.y.values))
model.evaluate(test_x,test_data.color.values)

# Prediction
prediction =  model.predict(np.array([[-2,3]]))
print(np.round(prediction))