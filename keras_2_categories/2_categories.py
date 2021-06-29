# here main kura vaneko, output ma acolor ra marker 2 ta predict garna cha,
# so what we do is, color ra marker duitai lai one hot encoding garne, ani merge gardiney,
# then predict both of them as a single output.


from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('2_categories_train.csv')
test_data = pd.read_csv('2_categories_test.csv')

# encoding the data i.e label encoding color column
color_encoded = pd.get_dummies(train_data.color).values
marker_encoded = pd.get_dummies(train_data.marker).values
#print(color_encoded)
#print(marker_encoded)
labels = np.concatenate((color_encoded,marker_encoded),axis = 1)
# print(labels)

test_color_encoded = pd.get_dummies(test_data.color).values
test_marker_encoded = pd.get_dummies(test_data.marker).values
#print(color_encoded)
#print(marker_encoded)
test_labels = np.concatenate((test_color_encoded,test_marker_encoded),axis = 1)
# print(labels)


# checking the graph
# plt.plot(train_data.x,train_data.y)
# plt.show()
# conclusion: the graph turns out to be clustered

# sequential / network design
model = keras.Sequential([
    keras.layers.Dense(32,input_shape = (2,),activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(9,activation='sigmoid'),
    ])

# # # # compiling the model
model.compile(optimizer='adam',loss = keras.losses.BinaryCrossentropy(from_logits=True),metrics='accuracy')

# # # # fiting the data to the model

x = np.column_stack((train_data.x.values,train_data.y.values))
np.random.RandomState(seed=20).shuffle(x)
np.random.RandomState(seed =20).shuffle(labels)
model.fit(x,labels,batch_size=4,epochs=15)

# # EVALUATING MODEL
print('Evaluating Result')
test_x = np.column_stack((test_data.x.values,test_data.y.values))
model.evaluate(test_x,test_labels)

# Prediction
# prediction =  model.predict(np.array([[-2,3]]))
# print(np.round(prediction))