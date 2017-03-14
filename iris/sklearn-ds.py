from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#dataset = load_iris()
iris = load_iris()
#print(data)

target_names = iris.target_names


#X = iris.data
y = iris.target

y_classes = []

for idx,val in enumerate(y):
  cardinal = y[idx]
  y_classes.append(target_names[cardinal])

data_c = np.c_[iris.data, y_classes]

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.DataFrame(data= data_c,
                     columns= names)

dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']] \
  = dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].astype(float)

dataset[['class']] \
  = dataset[['class']].astype(object)


# shape
#print(dataset.shape)

print(dataset.dtypes)

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

#print(len(data))
#print(len(data[0]))
#print(len(data[0][0]))
# print(len(data[1]))
# print(data[1])
# print(data[1])
#target = np.array(data[1])
#print(target)

#dataset = pd.DataFrame(list(data))

# shape
#print(dataset.shape)

# head


#target_names
#print list(dataset.target_names)

#data array( [  [1 2 3 4], [1 2 3 4] ])
#target array([0 1 1 2 1])
# DESCR

#Class Correlation

# shape
#print(dataset.shape)
