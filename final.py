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
num = 18000
PsS = []
PsH = []
PsH, PsS = datas.getPss()


mDataSet = datas.createe_dataset(PsH, PsS, num)
mDataSet = pd.DataFrame(mDataSet)


newData = datas.convert_data(mDataSet)


mDataSet = pd.read_csv(r'C:\Users\Jeries\Desktop\export_dataframe_v12.csv')
newData = pd.read_csv(r'C:\Users\Jeries\Desktop\bioWaze\FinalProjectBioWaze\SavedDataSets\DataSet_Avg.csv')


mDataSet.to_csv (r'C:\Users\Jeries\Desktop\export_dataframe_v12.csv', index = False, header=True)

                        #Step 2 - Normalizing the data
Xtest = mDataSet.values
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(Xtest)
mDataSet = pd.DataFrame(data_scaled)


Xtest = newData.values
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(Xtest)
newData = pd.DataFrame(data_scaled)

  

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



                #Step 4 EDITED!! - Reshape the Data for the model
mData = mDataSet

xs = []
ys = []
yt = mData[7]
i = 0
while(i < len(mData) - 30):

    v = mData.iloc[i: (i+29)].to_numpy()
    xs.append(v)
    ys.append(yt.iloc[i+29])
    i = i+30

xs = np.array(xs)
ys = np.array(ys)

X = xs
Y = ys




            #Step 5 - Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=42)



    #Create model combining LSTM with 1D Convonutional layer and MaxPool layer

lstm_out = 200

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.25))
#model.add(LSTM(units=lstm_out, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(units=128, return_sequences=True))
#model.add(Dropout(0.25))
model.add(LSTM(units=lstm_out))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())


# plt.scatter(range(5004), y_pred, c='g', label = 'Predictions')
# plt.scatter(range(5004), Y_test, c='r', label = 'YTest')
# plt.title('Predictions vs Tests')
# plt.xlabel('Users')
# plt.ylabel('Label')
# plt.legend()
# plt.show()


#fit model
batch_size = 30
history = model.fit(X_train, Y_train, validation_split = 0.1, epochs=10, verbose=1, batch_size=batch_size, shuffle=False)


            #Plotting the training and validation accuracy and loss at each epoch
            
            
                #loss

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



                #accuracy
            
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





#analyze the results
score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size=batch_size)
y_pred = model.predict(X_test)


cm = confusion_matrix(Y_test, y_pred.round())




model.save('CnnLstm.h5')




diabetesData = mDataSet
diabetes_Xtrain = mDataSet.iloc[:, 0:7].values
diabetes_Ytrain = mDataSet.iloc[:, 7].values

diabetesData = pd.read_csv('diabetes.csv')

diabetesData = newData
diabetes_Xtrain = diabetesData.iloc[:, 0:7]
diabetes_Ytrain = diabetesData.iloc[:, 7]

diabetes_Xtrain, diabetes_Xtest, diabetes_Ytrain, diabetes_Ytest = train_test_split(diabetes_Xtrain, diabetes_Ytrain, test_size= 0.2)

SVMclassifier = svm.SVC(kernel = 'linear', gamma = 'auto', C=2)
SVMclassifier.fit(diabetes_Xtrain, diabetes_Ytrain)
SVM_y_pred = SVMclassifier.predict(diabetes_Xtest)


print(classification_report(diabetes_Ytest, SVM_y_pred))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(diabetes_Xtrain)
diabetes_Xtrain = scaler.transform(diabetes_Xtrain)
diabetes_Xtest = scaler.transform(diabetes_Xtest)


from sklearn.neighbors import KNeighborsClassifier
KNNClassifier = KNeighborsClassifier(n_neighbors=6)
KNNClassifier.fit(diabetes_Xtrain, diabetes_Ytrain)


KNN_y_pred = KNNClassifier.predict(diabetes_Xtest)
print(classification_report(diabetes_Ytest, KNN_y_pred))
print(confusion_matrix(diabetes_Ytest, KNN_y_pred))
