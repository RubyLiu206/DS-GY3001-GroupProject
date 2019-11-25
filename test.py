# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:30:11 2019

@author: ruby_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import os
# Input data files are available in the "../input/" directory.

#def data_preprocessing():
    

def data_analysis(df):
    
    """
    precentage of the heart disease
    x label is for the count of people 
    """
    if 'target' in df.columns:
        sns.countplot(x="target", data=df, palette="bwr", color = ['lightsteelblue', 'cornflowerblue'])
        plt.show()
        count_No_Disease = len(df[df.target == 0])
        count_Have_Disease = len(df[df.target == 1])
        print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((count_No_Disease / (len(df.target))*100)))
        print("Percentage of Patients Have Heart Disease: {:.2f}%".format((count_Have_Disease / (len(df.target))*100)))

    """
    precentage of female and male
    x label is for the count of people
    """
    if 'sex' in df.columns:
        sns.countplot(x='sex', data=df, palette="mako_r", color = ['blue','darkblue'])
        plt.xlabel("Sex (0 = female, 1= male)")
        plt.show()
        
        count_Female = len(df[df.sex == 0])
        count_Male = len(df[df.sex == 1])
        print("Percentage of Female Patients: {:.2f}%".format((count_Female / (len(df.sex))*100)))
        print("Percentage of Male Patients: {:.2f}%".format((count_Male / (len(df.sex))*100)))

    """
    heart disease with the age
    """
    if 'age' in df.columns and 'target' in df.columns:
        pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6), color = ['mediumpurple','plum'])
        plt.title('Heart Disease Frequency for Ages')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.savefig('heartDiseaseAndAges.png')
        plt.show()


    """
    heart disease with the sex
    seprate the people and the disease 
    """
    if 'sex' in df.columns and 'target' in df.columns:
        pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['limegreen','green' ])
        plt.title('Heart Disease Frequency for Sex')
        plt.xlabel('Sex (0 = Female, 1 = Male)')
        plt.xticks(rotation=0)
        plt.legend(["Haven't Disease", "Have Disease"])
        plt.ylabel('Frequency')
        plt.show()
        
    """ 
    relation with disease and age 
    scatter plot with age range and the different maximum heart rate
    red one is for disease and blue is for non disease
    see the correlation with heart rate and disease 
    """
    if 'sex' in df.columns and 'target' in df.columns:
        plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
        plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
        plt.legend(["Disease", "Not Disease"])
        plt.xlabel("Age")
        plt.ylabel("Maximum Heart Rate")
        plt.show()

    if 'slope' in df.columns and 'target' in df.columns:
        pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['orange','moccasin' ])
        plt.title('Heart Disease Frequency for Slope')
        plt.xlabel('The Slope of The Peak Exercise ST Segment ')
        plt.xticks(rotation = 0)
        plt.ylabel('Frequency')
        plt.show()
    
    
    """
    Fasting Blood Sugar with disease
    to see whether fasting blood suger will influence the people have disease
    """    
    if 'fbs' in df.columns and 'target' in df.columns:
        pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['goldenrod','gold' ])
        plt.title('Heart Disease Frequency According To FBS')
        plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
        plt.xticks(rotation = 0)
        plt.legend(["Haven't Disease", "Have Disease"])
        plt.ylabel('Frequency of Disease or Not')
        plt.show()
        
    """
    Chest Pain Type with disease
    to see whether different chest type will influence the people have disease
    """
    if 'cp' in df.columns and 'target' in df.columns:
        pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['tomato','orangered' ])
        plt.title('Heart Disease Frequency According To Chest Pain Type')
        plt.xlabel('Chest Pain Type')
        plt.xticks(rotation = 0)
        plt.ylabel('Frequency of Disease or Not')
        plt.show()
    
    
    
    
def data_for_model(df):
    """
    change some categorical vars to dummy var
    """
    a = pd.get_dummies(df['cp'], prefix = "cp")
    b = pd.get_dummies(df['thal'], prefix = "thal")
    c = pd.get_dummies(df['slope'], prefix = "slope")
    frames = [df, a, b, c]
    df = pd.concat(frames, axis = 1)
    
    df = df.drop(columns = ['cp', 'thal', 'slope'])
    
    y = df.target.values
    x_data = df.drop(['target'], axis = 1)
    # Normalize
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
    #transpose matrices
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T
    
    print(x_train.shape)
    return x_train,y_train, x_test, y_test


def initialize(dimension):
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias

def sigmoid(z):
    y_head = 1/(1+ np.exp(-z))
    return y_head


def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients


def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    print("iteration:",iteration)
    print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients



def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction



def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
    
def machine_learning_model(x_train,y_train,x_test,y_test):
          
    # All model list here 
    # store the acc into accuracies dictionary
    
    """logistic regression """
    logistic_regression(x_train,y_train,x_test,y_test,1,100)
    
    
    
    """KNN"""
    knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
    knn.fit(x_train.T, y_train.T)
    prediction = knn.predict(x_test.T)
    
    print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
    
    #try ro find best k value
    scoreList = []
    for i in range(1,20):
        knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
        knn2.fit(x_train.T, y_train.T)
        scoreList.append(knn2.score(x_test.T, y_test.T))
    accuracies = {} 
    plt.plot(range(1,20), scoreList)
    plt.xticks(np.arange(1,20,1))
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.show()
    
    acc = max(scoreList)*100
    accuracies['KNN'] = acc
    print("Maximum KNN Score is {:.2f}%".format(acc))
    
    
    """SVM """
    svm = SVC(random_state = 1)
    svm.fit(x_train.T, y_train.T)
    
    acc = svm.score(x_test.T,y_test.T)*100
    accuracies['SVM'] = acc
    print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
    
    
    """Naive Bayes"""
    nb = GaussianNB()
    nb.fit(x_train.T, y_train.T)
    
    acc = nb.score(x_test.T,y_test.T)*100
    accuracies['Naive Bayes'] = acc
    print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
    
    
    """Decision tree"""
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train.T, y_train.T)
    
    acc = dtc.score(x_test.T, y_test.T)*100
    accuracies['Decision Tree'] = acc
    print("Decision Tree Test Accuracy {:.2f}%".format(acc))
    
    
    """Random forest"""
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
    rf.fit(x_train.T, y_train.T)
    
    acc = rf.score(x_test.T,y_test.T)*100
    accuracies['Random Forest'] = acc
    print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
    
    
    
    #compare and plot the acc
    colors = ["#DDA0DD", "#FFD700", "#FF6347", "#6495ED","#9ACD32","#FA8072"]
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(16,5))
    plt.yticks(np.arange(0,100,10))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
    plt.show()
    
    
    
    
    
    """ 
    All Predicted values 
    and get the confuse matrix
    """
    
    lr = LogisticRegression()
    lr.fit(x_train.T,y_train.T)
    y_head_lr = lr.predict(x_test.T)
    knn3 = KNeighborsClassifier(n_neighbors = 3)
    knn3.fit(x_train.T, y_train.T)
    y_head_knn = knn3.predict(x_test.T)
    y_head_svm = svm.predict(x_test.T)
    y_head_nb = nb.predict(x_test.T)
    y_head_dtc = dtc.predict(x_test.T)
    y_head_rf = rf.predict(x_test.T)
    
    
    cm_lr = confusion_matrix(y_test,y_head_lr)
    cm_knn = confusion_matrix(y_test,y_head_knn)
    cm_svm = confusion_matrix(y_test,y_head_svm)
    cm_nb = confusion_matrix(y_test,y_head_nb)
    cm_dtc = confusion_matrix(y_test,y_head_dtc)
    cm_rf = confusion_matrix(y_test,y_head_rf)
    plt.figure(figsize=(24,12))
    
    plt.suptitle("Confusion Matrixes",fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)
    
    plt.subplot(2,3,1)
    plt.title("Logistic Regression Confusion Matrix")
    sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
    plt.subplot(2,3,2)
    plt.title("K Nearest Neighbors Confusion Matrix")
    sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
    plt.subplot(2,3,3)
    plt.title("Support Vector Machine Confusion Matrix")
    sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
    plt.subplot(2,3,4)
    plt.title("Naive Bayes Confusion Matrix")
    sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
    plt.subplot(2,3,5)
    plt.title("Decision Tree Classifier Confusion Matrix")
    sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
    plt.subplot(2,3,6)
    plt.title("Random Forest Confusion Matrix")
    sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
    plt.show()
        
def main():
    # the dataset only from cleverland
    df = pd.read_csv("data/heart.csv")
    df.head(20)
    print(df.target.value_counts())
    data_analysis(df)

    x_train,y_train, x_test, y_test = data_for_model(df)

    #get the mean of total dataset
    df.groupby('target').mean()
    print(df.groupby('target').mean())
    machine_learning_model(x_train,y_train, x_test, y_test)
    
    # the dataset from all three places
    df = pd.read_csv("data/All_data_total.csv")
    df.head(20)
    print(df.target.value_counts())
    data_analysis(df)

    x_train,y_train, x_test, y_test = data_for_model(df)

    #get the mean of total dataset
    df.groupby('target').mean()
    print(df.groupby('target').mean())
    machine_learning_model(x_train,y_train, x_test, y_test)




    
    


