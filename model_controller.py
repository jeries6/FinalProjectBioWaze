# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:14:13 2021

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



    

class DataModel:
    def __init__(self):
        pass
    
                        #Step 1 - Creating the dataSet
    def createDataSet(self,num):
        datas = DataSim()                        
        PsS = []
        PsH = []
        PsH, PsS = datas.getPss()
        
        
        mDataSet = datas.createe_dataset(PsH, PsS, num)
        mDataSet = pd.DataFrame(mDataSet)        
        return mDataSet


                        

    def read_dataSet(self, dataSetPath):
        mDataSet = pd.read_csv(dataSetPath)
        return mDataSet
    
    def saveDataSet(mDataSet):
        mDataSet.to_csv (r'C:\Users\Jeries\Desktop\export_dataframe_v5.csv', index = False, header=True)



                        #Step 2 - Normalizing the data
    def normalize_data(self, mDataSet):
        Xtest = mDataSet.values
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(Xtest)
        mDataSet = pd.DataFrame(data_scaled)
        return mDataSet

  
                #Step 3 - Divide the data into X and Y (Features and Label)
    def divide_x_y(self, mDataSet):
        X = mDataSet.iloc[:, 0:7].values
        Y = mDataSet.iloc[:, 7:8].values
        return X,Y



                #Step 4 - Reshape the Data for the model
    def reshape_data(self, mDataSet):
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
        return X,Y
    
    
    def convert_data(self, mDataSet):
        import pandas as pd
        mData = mDataSet

        xs = []
        i = 0
        while(i < len(mData) - 30):
        
            v = mData.iloc[i: (i+29)]
            vSum = v.sum(axis=0)
            vSum = vSum.tolist()
            xs.append(vSum)
            i = i+30
        
        newData = pd.DataFrame(xs)
        return newData
    
    
                #Step 5 - Split the dataset
    def split_data(self, x, y, size):
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= size, random_state=42)
        return X_train, X_test, Y_train, Y_test



            #Create model combining LSTM with 1D Convonutional layer and MaxPool layer
    def create_cnnLstm_model(self, blocks, filters, poolSize, kernel, dropout):

        lstm_out = blocks
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel, activation='relu', padding='causal'))
        model.add(MaxPooling1D(pool_size=poolSize))
        model.add(Dropout(dropout))
        model.add(LSTM(units=lstm_out, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=32))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


                    #fit model

    def train_cnn_lstm(self, model, X_train, Y_train, epochs, batch_size):
        history = model.fit(X_train, Y_train, validation_split = 0.1, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=False)
        return history


            #Plotting the training and validation accuracy and loss at each epoch
    def plot_loss_epochs(self, history):

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
        


    def plot_acc_epochs(self, history):
                #accuracy
        loss = history.history['loss']        
        epochs = range(1, len(loss) + 1)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'y', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def plot_scatter_results(self, y_pred, Y_test):


        plt.scatter(range(59994), y_pred, c='g')
        plt.scatter(range(59994), Y_test, c='r')
        plt.show()



    def predict(self, model, X_test):
        y_pred = model.predict(X_test)   
        return y_pred
    
    def analyze_results(self, X_test, Y_test, batch_size, model):
        #analyze the results
        score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size=batch_size)
        return score, acc


    def conf_mat(self, Y_test, y_pred):
        cm = confusion_matrix(Y_test, y_pred)
        return cm


    def save_model(self, model):
        model.save('CnnLstm.h5')


    def svm_model(self, fName, mKernel, gamm, c):

        diabetesData = pd.read_csv(fName)
        diabetes_Xtrain = diabetesData.iloc[:, 0:7]
        diabetes_Ytrain = diabetesData.iloc[:, 7]
        
        diabetes_Xtrain, diabetes_Xtest, diabetes_Ytrain, diabetes_Ytest = train_test_split(diabetes_Xtrain, diabetes_Ytrain, test_size= 0.2)
        
        SVMclassifier = svm.SVC(kernel = mKernel, gamma = gamm, C=c)
        SVMclassifier.fit(diabetes_Xtrain, diabetes_Ytrain)
        SVM_y_pred = SVMclassifier.predict(diabetes_Xtest)
        
        print(classification_report(diabetes_Ytest, SVM_y_pred))
        print(confusion_matrix(diabetes_Ytest, SVM_y_pred))
        from sklearn.metrics import accuracy_score
        return accuracy_score(diabetes_Ytest, SVM_y_pred)



    def convert_avg(self, dataSet):
        datas = DataSim() 
        newData = datas.convert_data(dataSet)
        Xtest = newData.values
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(Xtest)
        newData = pd.DataFrame(data_scaled)
        return newData

    def knn_model(self, fName, n):
        
        from sklearn.preprocessing import StandardScaler
        
        diabetesData = pd.read_csv(fName)
        diabetes_Xtrain = diabetesData.iloc[:, 0:7]
        diabetes_Ytrain = diabetesData.iloc[:, 7]
        
        diabetes_Xtrain, diabetes_Xtest, diabetes_Ytrain, diabetes_Ytest = train_test_split(diabetes_Xtrain, diabetes_Ytrain, test_size= 0.2)
        
        scaler = StandardScaler()
        scaler.fit(diabetes_Xtrain)
        diabetes_Xtrain = scaler.transform(diabetes_Xtrain)
        diabetes_Xtest = scaler.transform(diabetes_Xtest)
        
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        KNNClassifier = KNeighborsClassifier(n_neighbors=n)
        KNNClassifier.fit(diabetes_Xtrain, diabetes_Ytrain)
        
        
        KNN_y_pred = KNNClassifier.predict(diabetes_Xtest)
        print(classification_report(diabetes_Ytest, KNN_y_pred))
        print(confusion_matrix(diabetes_Ytest, KNN_y_pred))
        
        return accuracy_score(diabetes_Ytest, KNN_y_pred)
