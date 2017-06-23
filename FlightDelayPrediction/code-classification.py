# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:14:00 2016

@author: ramakrishna
"""
from scipy import interp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#pd.options.display.mpl_style = 'default'
#plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,auc
from sklearn.cross_validation import KFold,StratifiedKFold,cross_val_predict,cross_val_score
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
filepath=input("Please enter the location of data file: ")
outpath=input("Please enter the location to store the results :")
#filepath='C:\\UTA Courses\\spring\\Data Science 5378\\project\\final data\\withweather'
#outpath="C:\\UTA Courses\\spring\\Data Science 5378\\project\\final data\\withweather\\new"


print("1 - Logistic Regression\n"+"2 - Naive Bayes\n"+"3 - random forest\n"+"4 - Decision Tree\n")
model=0
while(True):
    model=int(input("Please enter the number(1-4) of model : "))
    if model in [1,2,3,4]:
        break
    else:
        print("please enter a valid number")
data = pd.read_csv(filepath+'\\fromdfwweather2013smote.csv')
data['month']=data['month'].astype('str')
data['dephrs']=data['dephrs'].astype('str')
data['arrhrs']=data['arrhrs'].astype('str')

#Process the file
columns= data.columns
#mapdict stores the original values vs the transformed values for categorical variables
mapdict={}
def convert_to_std():
    count=0
    global mapdict
    for col in data.columns:
        if (data[col].dtypes == object):
            distvalues= set(data[col])
            repldict={}
            for catcol in distvalues:
                count=count+1
                repldict[catcol]='A'+str(count)
                data[col]=data[col].replace(catcol,'A'+str(count))
            mapdict[col]=repldict
predictors=data[columns[0:(len(columns)-1)]]
predictors = pd.get_dummies(predictors)
label_encoder = preprocessing.LabelEncoder()
def convert_cat_num():
    for col in predictors.columns:
        if (predictors[col].dtypes == object):
            encoded_predictors = label_encoder.fit_transform(predictors[col])
            predictors.loc[:,col]= encoded_predictors.tolist()    
#convert_to_std()
#X_scaled = preprocessing.scale(X)            
#convert_cat_num()
corrdf=predictors.corr()
corrdf.to_excel(outpath+"\\corrmatrix.xlsx")
#data.groupby('class').plas.hist()
#data.to_csv("C:\\UTA Courses\\spring\\Data Science 5378\\homework 4 machine learning\\creditDataformatted.csv",index=False)
# logistic regression-------------------------------

target= data[columns[len(columns)-1]]
target = target.replace('No Delay', 0.0)
target = target.replace('Delay', 1.0) 
#0- good 1- bad

predarr=np.array(predictors)
targarr=np.array(target)
#kf_total = KFold(len(targarr), n_folds=10)
skf_total=StratifiedKFold(targarr,n_folds=5)
if model== 1:   
    classifier = LogisticRegression()
elif model == 2:
    classifier = BernoulliNB()
elif model == 3:
#    # depth analysis
#    scores = []
#    depth_values = []
#    for depth in range(1,50):
#        classifier1 = RandomForestClassifier(max_depth = depth, random_state = 99)
#        if classifier1.fit(predictors, target).max_depth < depth:
#            break
#        predictarr1=cross_val_predict(classifier1,predarr,targarr,cv=skf_total)
#        confmat1=confusion_matrix(targarr, predictarr1)
#        score=np.mean(cross_val_score(classifier1, predictors, target, scoring = 'accuracy',
#                                    cv = skf_total))
#
#        depth_values.append(depth)
#        scores.append(score)
#        print('Depth: {} \t accuracy: {}\n'.format(depth, score))
#plt.plot(depth_values, scores)
#plt.xlabel('Depth')
#plt.ylabel('Accuracy')
#plt.savefig("C:\\UTA Courses\\spring\\Data Science 5378\\project\\final data\\withweather\\output\\randdepthacc.png",dpi=1200)
    classifier = RandomForestClassifier(class_weight={0.0:1,1.0:100},max_depth = 40,random_state = 99)
elif model == 4:
    #adjusted the depth to get better costanalysis
    #classifier = DecisionTreeClassifier(class_weight={0.0:1,1.0:7},random_state=99,max_depth=2)
# #depth analysis
#    scores = []
#    depth_values = []
#    for depth in range(1,50):
#        classifier1 = tree.DecisionTreeClassifier(max_depth = depth, random_state = 99)
#        if classifier1.fit(predictors, target).tree_.max_depth < depth:
#            break
#        predictarr1=cross_val_predict(classifier1,predarr,targarr,cv=skf_total)
#        confmat1=confusion_matrix(targarr, predictarr1)
#        score=np.mean(cross_val_score(classifier1, predictors, target, scoring = 'accuracy',
#                                    cv = skf_total))
#
#        depth_values.append(depth)
#        scores.append(score)
#        print('Depth: {} \t accuracy: {}\n'.format(depth, score))
#plt.plot(depth_values, scores)
#plt.xlabel('Depth')
#plt.ylabel('Accuracy')
#plt.savefig(outpath+"\\depthacc.png",dpi=1200)
        classifier = DecisionTreeClassifier(max_depth=21,random_state=99)
    

#train the model
classifier.fit(predictors, target)
predictarr=cross_val_predict(classifier,predictors,target,cv=skf_total)
#classifier.predict()
#get the features
import operator
if model==1 : 
    predfet=classifier.coef_.tolist()
    print("feature importance logistic- descending order of importance")   
    predfet=predfet[0]
    print("intercept : "+str(classifier.intercept_[0]))
    colnames=predictors.columns.tolist()
    fetdict=dict(zip(colnames,predfet))
    sortedfetlist=sorted(fetdict.items(),key=operator.itemgetter(1),reverse=True)
    for item in sortedfetlist:
            print(item[0] + " : " + str(item[1]))
elif model == 4 or model==3:
    predfet=classifier.feature_importances_.tolist()
    colnames=predictors.columns.tolist()
    fetdict=dict(zip(colnames,predfet))
    sortedfetlist=sorted(fetdict.items(),key=operator.itemgetter(1),reverse=True)
    print("feature importance decision tree- descending order of importance")
    for item in sortedfetlist:
        if item[1]>0:
            print(item[0]+ " : " + str(item[1]))

#let us get an accuracy,precision,recall,f1 score
print("\nAccuracy: " + str(metrics.accuracy_score(targarr, predictarr)))
print("\nThe confusion matrix is: ")
confmat=confusion_matrix(targarr, predictarr)
print(confusion_matrix(targarr, predictarr))
print("\n Cost analysis(1-good 5-bad) is: "+ str(confmat[0][1]+((confmat[1][0])*5)))
print("\nThe classification report is: ")
print(classification_report(targarr, predictarr,target_names=["No Delay","Delay"]))
#plotting ROC curves
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(skf_total):
    probas_ = classifier.fit(predarr[train], targarr[train]).predict_proba(predarr[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(targarr[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='base')

mean_tpr /= len(skf_total)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=3)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right",prop={'size':8})
#plt.show()
plt.savefig(outpath+'\\'+str(model)+'-ROC.png',format='png', dpi=1200)
# plot decision tree
from sklearn.externals.six import StringIO  
#x=classifier.estimators_
import pydotplus 
if model == 4:
    dot_data = StringIO() 
    tree.export_graphviz(classifier, out_file=dot_data, feature_names = predictors.columns.tolist(),class_names=['No Delay','Delay'],filled=True) 
    #tree.export_graphviz(classifier, out_file=dot_data) 
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf(outpath+"\\tree.pdf")   
#
#
