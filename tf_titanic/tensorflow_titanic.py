import pandas as pd
from  sklearn.model_selection import train_test_split
import tensorflow as tf

data1 = pd.read_csv('train.csv')
data1.drop(['Name','Cabin','Ticket'],axis = 1,inplace= True)
# managing data
data1 = data1.dropna()
#print(data1.shape)

#test,train,split
x_train,x_test = train_test_split(data1,test_size=0.10,random_state=42)
y_train = x_train.pop('Survived')
y_test = x_test.pop('Survived')

# columns
categorical_columns = ['Sex','Embarked']
numeric_columns = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']

feature_columns = []

for feature in categorical_columns:
    unique_values = x_train[feature].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature,unique_values))
for feature1 in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature1,dtype=tf.int32))

# making input function
def make_input_function(x,y,num_epoch = 10,shuffle = True,batch_size = 26):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(x),y))
        if shuffle:
            dataset = dataset.shuffle(700)
        dataset =  dataset.batch(batch_size).repeat(num_epoch)
        return dataset
    return input_function


train_input_function = make_input_function(x_train,y_train)
test_input_function = make_input_function(x_test,y_test,num_epoch =1,shuffle = False)

linear_estimator = tf.estimator.LinearClassifier(feature_columns = feature_columns)

linear_estimator.train(train_input_function)
result = linear_estimator.evaluate(test_input_function)

# clear_output()

print(result)

# from __future__ import absolute_import, division, print_function, unicode_literals
#
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# from six.moves import urllib
# import tensorflow as tf