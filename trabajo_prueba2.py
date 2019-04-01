
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dataset=pd.read_csv("train.csv")
#print(dataset)

clf = DecisionTreeClassifier()

#entrenamiento
xtrain = dataset.iloc[0:21000, 1:].values
train_label = dataset.iloc[0:21000, 0].values

clf.fit(xtrain, train_label)

#pruebas
xtest = dataset.iloc[21000:, 1].values
actual_label = dataset.iloc[21000:, 0].values

#s
d = xtest[8]
np.reshape(d, (28,28))
#d.reshape(d,(28,28))
plt.imshow(255-d, cmap="gray")
#plt.show()
print("----------------------------")
print(clf.predict([xtest[8]]))

"""
###############
p = clf.predict(xtest)
count = 0

for i in range(0,21000):
    if(p[i] == actual_label[i]):
        count+=1
print("precision: ", (count/21000)*100)"""

