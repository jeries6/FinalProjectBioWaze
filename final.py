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



#Data = pd.read_excel('DiabetesDataSet.xlsx')

def create_dataset2(num):
    import random
    mList = []

    
    for i in range(1,num+1):
        food=random.randint(1,10)
        sport=random.randint(1,10)
        feelings=random.randint(1,10)
        medical=random.randint(1,10)
        thirsty=random.randint(1,10)
        pee=random.randint(1,10)
        smoking=random.randint(0, 10)
        sick=0
        if(smoking == 1 and food <= 5):
            sick = 1
        elif(smoking > 7  and sport > 6):
            sick = 0
        elif(smoking == 0 and food < 5 and sport < 5):
            sick = 1 
        elif(medical > 5 and (feelings < 5 or sport < 4)):
            sick = 1  
        elif(sport < 8 and (food < 7 or smoking > 2)):
            sick = 1 
        elif(feelings < 5 and (food < 5 or smoking > 5)):
            sick = 1      
        mList = mList + [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
        

    df = pd.DataFrame(mList)
    return df



def mPFunction(num, a):
    import math 
    x = -pow((num - a), 2)
    return math.exp(x)

def getPss():  #get the posibilities of healthy and sick people
    Pss = []
    PsH = []
    PsS = []
    for i in range(10):
        y = mPFunction(i+1, 3)
        Pss.append(y)
    
    Pss.reverse()
    PssSum = sum(Pss)    
    for i in range(10):
        x = Pss.pop()
        newP = x/PssSum
        PsH.append(newP)
        
    Pss = []    
    for i in range(10):
        Pss.append(mPFunction(i+1, 7))
    
    Pss.reverse()
    PssSum = sum(Pss)    
    for i in range(10):
        x = Pss.pop()
        newP = x/PssSum
        PsS.append(newP)
        
    
        
    return PsH, PsS
        

def getRandomNumber(Ps):
    import numpy as np
    flag = 1;
    psSum = 0
    fIndex = 1
    Pss = Ps
    Pss.reverse()
    psSum = Pss.pop()
    feature = np.random.uniform()
    while(flag or (fIndex < 11)):
        if(feature > psSum):
            temp = Pss.pop()
            fIndex += 1
            psSum += temp
        elif(feature <= psSum):
            return fIndex
    

def createDataSetPs(PsH, PsS, num):
    import numpy as np
    
    for i in range(num):
        food=np.random.uniform()
        sport=np.random.uniform()
        feelings=np.random.uniform()
        medical=np.random.uniform()
        thirsty=np.random.uniform()
        pee=np.random.uniform()
        smoking=np.random.uniform()
    

def create_dataset_RDfeatures(pRD, a):
    import numpy as np
    food=np.random.choice(a, p = pRD)
    sport=np.random.choice(a, p = pRD)
    feelings=np.random.choice(a, p = pRD)
    medical=np.random.choice(a, p = pRD)
    thirsty=np.random.choice(a, p = pRD)
    pee=np.random.choice(a, p = pRD)
    smoking=np.random.choice(a, p = pRD)
        
    return food, sport, feelings, medical, thirsty, pee, smoking
    

def createe_dataset(users):
    import numpy as np
    import random
    mList = []
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p_regular = np.array([0.05, 0.08, 0.09, 0.1, 0.1, 0.16000000000000003, 0.2, 0.1, 0.1, 0.02])
    pRD_high = np.array([0.02, 0.02, 0.02, 0.02, 0.2, 0.22, 0.2, 0.1, 0.1, 0.1])   #P's for the Random Days
    pRD_low = np.array([0.15, 0.11, 0.2, 0.2, 0.2, 0.04, 0.03, 0.03, 0.02, 0.02])   #P's for the Random Days for features that the max is the worst
    ps = [p_regular, pRD_high, pRD_low]
    features_p = {'food': 2, 'sport': 2, 'feelings': 2, 'medical': 1, 'thirsty': 1, 'pee': 1, 'smoking': 1}
    
    features = ['food', 'sport', 'feelings', 'medical', 'thirsty', 'pee', 'smoking']  
    feature_list = [['food', 'sport'], ['food', 'smoking'], ['food'], ['smoking'], ['thirsty', 'food'], ['pee'], ['thirsty'], ['feelings', 'pee','food'], ['smoking', 'sport']]
    
    mList2 = []    
    for i in range(users):    

    
        Diabetes = random.randint(0,1)
        if(Diabetes == 1):
            randomDays = random.randint(15,22)
            for i in range(1,randomDays):
                food, sport, feelings, medical, thirsty, pee, smoking = create_dataset_RDfeatures(p_regular, a)
                sick=0 
                mList2 = mList2 + [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
            
            random_features = random.choice(feature_list)
            for i in range(randomDays, 31):
                food, sport, feelings, medical, thirsty, pee, smoking = create_dataset_RDfeatures(p_regular, a)
                sick=1 
                ftrs = [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
                for item in random_features:
                    temp = item
                    ftrs[0][temp.capitalize()] = np.random.choice(a, p = ps[features_p[item]])
    
                mList2 = mList2 + ftrs
                
        else:
            for i in range(1, 31):
                food, sport, feelings, medical, thirsty, pee, smoking = create_dataset_RDfeatures(p_regular, a)
                sick=0 
                mList2 = mList2 + [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
            
    
    
     
    # seq = 0
    # for row in mList2:    
    #     if(row['Food'] > 5):
    #         seq = seq+1
    #         if (seq == 4):
    #             row['Diabetes'] = 1
    #df = pd.DataFrame(mList)
    mList = mList + mList2
    return mList    



PsS = []
PsH = []
PsH, PsS = createMyDataP()


mDataSet = createe_dataset(1000)



mDataSet = pd.DataFrame(mDataSet)

mDataSet.to_csv (r'C:\Users\Jeries\Desktop\export_dataframe_v5.csv', index = False, header=True)

                #Normalizing the data
Xtest = mDataSet.values
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(Xtest)
mDataSet = pd.DataFrame(data_scaled)

  


X = mDataSet.iloc[:, 0:7].values
Y = mDataSet.iloc[:, 7:8].values

X = mDataSet[['Food', 'Sport', 'Medical', 'Feelings', 'Smoking', 'Thirsty', 'Pee']]
Y = mDataSet[['Diabetes']]
Y = np.reshape(Y, (num))
#Y= mDataSet[:, 5]

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

#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)



#create model combining LSTM with 1D Convonutional layer and MaxPool layer

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


plt.scatter(range(5994), y_pred, c='g')
plt.scatter(range(5994), Y_test, c='r')
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
