#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:57:27 2018

@author: abhijay
"""

# import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# os.chdir('/home/abhijay/Documents/ML/hw_2/Q9')

##### Used to get data #####
def readFile(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

##### Used for creating vocabulary #####
def create_vocabulary(data):
    ##### Join all sentences #####
    corpus = (' '.join(map(str, data))).split(' ')
        
    ##### Remove duplicates #####
    corpus = list(set(corpus))
    
    ##### Sort the corpus #####
    corpus.sort()

    ##### Make vocabulary #####
    removeStopWords = (lambda word: word not in stoplist)
    vocabulary = list(filter(removeStopWords,corpus))
    
    return vocabulary

##### Used for pre-processing #####
def pre_processing( data, vocabulary):
    
    ##### Split into sentences #####
    data = [sentence.split(' ') for sentence in data]

    ##### Initialize data matrix to the feature set of vocab and size of data_x and fill with zero #####
    x = np.zeros((len(data), len(vocabulary)))
    
    ##### Fill vocab matrix with 1s of the word exists in  #####
    for i, sentence in enumerate(data):
        for word in sentence:
            if word in vocabulary:
                x[i,vocabulary.index(word)] = 1
    return x

##### Used for fitting model #####
def fit(x,y):
    probY = [0,0]
    
    # P(Y=1)
    probY[1] = (y.sum()+1)/(len(y)+2) # Laplace smoothing applied
    # P(Y=0)
    probY[0] = ((y==0).astype(int).sum()+1)/(len(y)+2) # Laplace smoothing applied
    
    probX = []
    # P(X=u_i|Y=1)
    probX.append(x[y == 0].sum(axis=0)/x[y == 0].shape[0])
    # P(X=u_i|Y=1)
    probX.append(x[y == 1].sum(axis=0)/x[y == 1].shape[0])
    model = [probX,probY]
    return model

##### Used for predicting #####
def classify(x,y,model):
    
    ##### Load model #####
    probX,probY = model[0],model[1] 
    
    ##### Select argmax of values     #####
    ##### P(Y=1) PROD(P(X=u_i|Y=1))   #####
    y_pred = np.array([np.argmax([ \
                                  np.prod(probX[0][x_i == 1])*probY[0], \
                                  np.prod(probX[1][x_i == 1])*probY[1] \
                                  ]) for x_i in x])
    return y_pred

def get_accuracy(y,y_pred):
    ##### Compare predicted with actual values #####
    accuracy = str(round(100*np.sum((y_pred == y).astype(int))/y.shape[0],2))+"%"
    return accuracy
    
if __name__ == "__main__":
    
    ##### Get data #####
    stoplist = readFile("fortunecookiedata/stoplist.txt")
    
    train_data = readFile('fortunecookiedata/traindata.txt')
    train_y = readFile('fortunecookiedata/trainlabels.txt')
    
    test_data = readFile('fortunecookiedata/testdata.txt')
    test_y = readFile('fortunecookiedata/testlabels.txt')
    
    ##### Convert target variable to int #####
    train_y = np.array([int(y) for y in train_y])
    test_y = np.array([int(y) for y in test_y])
    
    ##### Build a vocabulary for pre_processing #####
    vocabulary = create_vocabulary(train_data)
    
    ##### Converting text data to features #####
    train_x = pre_processing( train_data, vocabulary)
    test_x = pre_processing( test_data, vocabulary)
    
    ##### Naive Bayes #####
    nb_model = fit(train_x,train_y) 
    train_pred = classify(train_x,train_y,nb_model)
    print ("Naive Bayes train accuracy:\n",get_accuracy(train_y,train_pred))
    
    test_pred = classify(test_x,test_y,nb_model)
    print ("Naive Bayes test accuracy:\n",get_accuracy(test_y,test_pred))
    
    ##### Scikit-Learn Naive Bayes #####
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)
    
    train_pred = gnb.predict(train_x)
    print ("scikit-learn Naive Bayes train accuracy:\n",get_accuracy(train_y,train_pred))
    
    test_pred = gnb.predict(test_x)
    print ("scikit-learn Naive Bayes test accuracy:\n",get_accuracy(test_y,test_pred))
    
    ##### Scikit-Learn Logistic Regression #####
    logreg = LogisticRegression()
    logreg.fit(train_x, train_y)
    
    y_pred = logreg.predict(train_x)
    print ("scikit-learn Logistic Regression train accuracy:\n",get_accuracy(train_y,train_pred))
    
    y_pred = logreg.predict(test_x)
    print ("scikit-learn Logistic Regression test accuracy:\n",get_accuracy(test_y,test_pred))
    
    
    # 