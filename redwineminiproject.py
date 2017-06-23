# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:36:41 2016

@author: RAMA
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
redwine= pd.read_csv("C:/Users/RAMA/Downloads/MSBA/SEMESTER 2 Spring/Dr. Sridhar/final mini pjt/winequality-red.csv",sep=";")
whitewine= pd.read_csv("C:/Users/RAMA/Downloads/MSBA/SEMESTER 2 Spring/Dr. Sridhar/final mini pjt/winequality-white.csv",sep=";")
red_col=redwine.columns
red_col_val=redwine.columns.values
white_col=whitewine.columns.values
red_rows = redwine.shape[0]
red_columns = redwine.shape[1]
white_rows = whitewine.shape[0]
white_columns = whitewine.shape[1]
target_red = redwine[red_col[11:]]
predictors_red= redwine[red_col[0:11]]
rtrain_X, rtest_X, rtrain_Y, rtest_Y =  train_test_split(predictors_red, target_red, test_size = 0.2, random_state = 99)


X_std = StandardScaler().fit_transform(rtrain_X) #PCA
mean_vec = np.mean(X_std, axis=0)#PCA
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)#PCA
print('(THIS IS FOR PCA ANALYSIS (code is not complete)\nCovariance matrix \n%s' %cov_mat)

#classifier_svmrfb = SVC(kernel='rbf', C=0.01,probability=True)
print("\n1.SVM Linear\n2.SVM rbf \n3.Extra Trees")
choice=int(input("\nChoose a machine learning algorithm:"))

def wine_quality(choice):
    if choice==1:
        
        classifier_svmlinear = SVC(kernel='linear', C=0.01,probability=True)
        classifier_svmlinear.fit(rtrain_X, rtrain_Y.values.ravel())
        predicted_red = classifier_svmlinear.predict(rtest_X)
        
        #probabilities_red = classifier_svmlinear.predict_proba(rtest_X)
        print("--Support Vector Classifier--\n1. Accuracy: " + str(metrics.accuracy_score(rtest_Y, predicted_red)))
        print("\n2. The classification report for SVC algorithm:\n")
        print(metrics.classification_report(rtest_Y, predicted_red))
    elif choice==2:
        classifier_svmrbf = SVC(kernel='rbf', C=0.1,probability=True)
        classifier_svmrbf.fit(rtrain_X, rtrain_Y.values.ravel())
        predicted_red = classifier_svmrbf.predict(rtest_X)
        #probabilities_red = classifier_svmrbf.predict_proba(rtest_X)
        print("--Support Vector Classifier--\n1. Accuracy: " + str(metrics.accuracy_score(rtest_Y, predicted_red)))
        print("\n2. The classification report for SVC algorithm:\n")
        print(metrics.classification_report(rtest_Y, predicted_red))
    elif choice==3:
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        forest.fit(rtrain_X, rtrain_Y.values.ravel())
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(rtrain_X.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, redwine.columns[indices[f]], importances[indices[f]]))
        predicted_red = forest.predict(rtest_X)   
        print("--Extra Trees--\n1. Accuracy: " + str(metrics.accuracy_score(rtest_Y, predicted_red)))
        print("\n2. The classification report for extra trees algorithm:\n")
        print(metrics.classification_report(rtest_Y, predicted_red))
        
        #http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py

wine_quality(choice)

