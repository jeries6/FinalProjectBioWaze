# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:17:02 2021

@author: Jeries
"""

class DataSim:
    
    def __init__(self):
        pass
    
    
    
    def mPFunction(self, num, a):   #calculate the function of the possibility (e^(-(num - a)^2))
        import math 
        x = -pow((num - a), 2)
        return math.exp(x)
    
    def getPss(self):  #get the posibilities lists of healthy and sick people
        Pss = []
        PsH = []
        PsS = []
        for i in range(10):
            y = self.mPFunction(i+1, 3)
            Pss.append(y)
        
        Pss.reverse()
        PssSum = sum(Pss)    
        for i in range(10):
            x = Pss.pop()
            newP = x/PssSum
            PsH.append(newP)
            
        Pss = []    
        for i in range(10):
            Pss.append(self.mPFunction(i+1, 7))
        
        Pss.reverse()
        PssSum = sum(Pss)    
        for i in range(10):
            x = Pss.pop()
            newP = x/PssSum
            PsS.append(newP)
            
        
            
        return PsH, PsS
            
    
    def getRandomNumber(self, Ps):   #given the probabilities list, generate a random number between 1-10
        import numpy as np
        flag = 1;
        psSum = 0
        fIndex = 1
        Pss = Ps.copy()
        Pss.reverse()
        psSum = Pss.pop()
        feature = np.random.uniform()
        while(flag or (fIndex < 10)):
            if(feature > psSum):
                temp = Pss.pop()
                fIndex += 1
                psSum += temp
            elif(feature <= psSum):
                return fIndex
        
        
        
    def createDataSetPs(self, PsH):    #Generate numbers from 1-10 and assign them to the features using the possibility function psH 
        food = self.getRandomNumber(PsH)
        sport= self.getRandomNumber(PsH)
        feelings= self.getRandomNumber(PsH)
        medical= self.getRandomNumber(PsH)
        thirsty= self.getRandomNumber(PsH)
        pee= self.getRandomNumber(PsH)
        smoking= self.getRandomNumber(PsH)
        
        return food, sport, feelings, medical, thirsty, pee, smoking
            
            
    
    
    
    def createe_dataset(self, psh, pss, users):  #given probabilities lists of sick and healthy people, create a dataSet of 30 days per user, returns (users * 30) days
        import random
        mList = []
        mList2 = []    
        
    
        feature_list = [['food', 'sport'], ['food', 'smoking'], ['food'], ['smoking'], ['thirsty', 'food'], ['pee'], ['thirsty'], ['feelings', 'pee','food'], ['smoking', 'sport']]
        
        for i in range(users):    
        
            Diabetes = random.randint(0,1)
            if(Diabetes == 1):
                randomDays = random.randint(15,22)
                for i in range(1,randomDays):
                    food, sport, feelings, medical, thirsty, pee, smoking = self.createDataSetPs(psh)
                    sick=0 
                    mList2 = mList2 + [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
                
                random_features = random.choice(feature_list)
                for i in range(randomDays, 31):
                    food, sport, feelings, medical, thirsty, pee, smoking = self.createDataSetPs(psh)
                    sick=1 
                    ftrs = [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
        
                    for item in random_features:
                        temp = item
                        ftrs[0][temp.capitalize()] = self.getRandomNumber(pss)
                        
                    mList2 = mList2 + ftrs
                    
            else:
                for i in range(1, 31):
                    food, sport, feelings, medical, thirsty, pee, smoking = self.createDataSetPs(psh)
                    sick=0 
                    mList2 = mList2 + [{'Food' : food, 'Sport' : sport, 'Feelings' : feelings, 'Medical' : medical, 'Smoking' : smoking, 'Thirsty' : thirsty, 'Pee' : pee,'Diabetes' : sick}]
                
        
        mList = mList + mList2
        return mList
    
    
    