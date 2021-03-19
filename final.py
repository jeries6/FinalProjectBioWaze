# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:41:50 2021

@author: Jeries
"""

import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
from nltk.tokenize import RegexpTokenizer

from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D, Dropout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report

from sklearn import preprocessing, svm

from datas import DataSim





                        #Step 1 - Creating the dataSet
datas = DataSim()                        
num = 1000
PsS = []
PsH = []
PsH, PsS = datas.getPss()


mDataSet = datas.createe_dataset(PsH, PsS, 100)



mDataSet = pd.DataFrame(mDataSet)

mDataSet.to_csv (r'C:\Users\Jeries\Desktop\export_dataframe_v5.csv', index = False, header=True)

                        #Step 2 - Normalizing the data
Xtest = mDataSet.values
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(Xtest)
mDataSet = pd.DataFrame(data_scaled)

  

                #Step 3 - Divide the data into X and Y (Features and Label)
X = mDataSet.iloc[:, 0:7].values
Y = mDataSet.iloc[:, 7:8].values



X = mDataSet[['Food', 'Sport', 'Medical', 'Feelings', 'Smoking', 'Thirsty', 'Pee']]
Y = mDataSet[['Diabetes']]
Y = np.reshape(Y, (num))


X = X.to_numpy()
X = np.reshape(X, (540, 1, 7))
Y = Y.to_numpy()
Y = np.reshape(Y, (540, 1, 1))



print(X.shape)




X = X.astype(np.float32)
Y = Y.astype(np.float32)


mDataSet = mDataSet.values.tolist()


mData = pd.read_csv("C:\\Users\\Jeries\\Desktop\\FinalProject\\mData.csv")


mDataSet = mData  
                #Step 4 - Reshape the Data for the model
mData = mDataSet

xs = []
ys = []
yt = mData[7]
for i in range(len(mData) - 30):
    v = mData.iloc[i: (i+30)].to_numpy()
    xs.append(v)
    ys.append(yt.iloc[i+30])

xs = np.array(xs)
ys = np.array(ys)

X = xs
Y = ys

            #Step 5 - Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)



    #Create model combining LSTM with 1D Convonutional layer and MaxPool layer

lstm_out = 128

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.2))
model.add(LSTM(units=lstm_out, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


plt.scatter(range(594), y_pred, c='g')
plt.scatter(range(594), Y_test, c='r')
plt.show()


#fit model
batch_size = 30
model.fit(X_train, Y_train, epochs=10, verbose=1, batch_size=batch_size, shuffle=False)

#analyze the results
score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size=batch_size)
y_pred = model.predict(X_test)

model.save('CnnLstm.h5')



diabetesData = pd.read_csv('diabetes.csv')
diabetes_Xtrain = diabetesData.iloc[:, 0:8]
diabetes_Ytrain = diabetesData.iloc[:, 8:9]

diabetes_Xtrain, diabetes_Xtest, diabetes_Ytrain, diabetes_Ytest = train_test_split(diabetes_Xtrain, diabetes_Ytrain, test_size= 0.2)

SVMclassifier = svm.SVC(kernel = 'linear', gamma = 'auto', C=3)
SVMclassifier.fit(diabetes_Xtrain, diabetes_Ytrain)
SVM_y_pred = SVMclassifier.predict(diabetes_Xtest)


print(classification_report(diabetes_Ytest, SVM_y_pred))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(diabetes_Xtrain)
diabetes_Xtrain = scaler.transform(diabetes_Xtrain)
diabetes_Xtest = scaler.transform(diabetes_Xtest)


from sklearn.neighbors import KNeighborsClassifier
KNNClassifier = KNeighborsClassifier(n_neighbors=5)
KNNClassifier.fit(diabetes_Xtrain, diabetes_Ytrain)


KNN_y_pred = KNNClassifier.predict(diabetes_Xtest)
print(classification_report(diabetes_Ytest, KNN_y_pred))
print(confusion_matrix(diabetes_Ytest, KNN_y_pred))
