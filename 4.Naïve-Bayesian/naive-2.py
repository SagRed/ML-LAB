
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing

dataf = pd.read_csv("./edata.csv")
feature_col_names = ['outlook','temp','humidity','wind']
predicted_class_names = ['play']

def MultiLabelEncoder(columnlist,dataframe):
    for i in columnlist:
        labelencoder_X=preprocessing.LabelEncoder()
        dataframe[i]=labelencoder_X.fit_transform(dataframe[i])
    return dataframe
le = preprocessing.LabelEncoder()
feature_col = ['outlook','temp','humidity','wind','play']

Xdata = MultiLabelEncoder(feature_col,dataf)
X = Xdata[feature_col_names]

yy = dataf[predicted_class_names]


y = Xdata[predicted_class_names]
print(dataf.head)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)
print ('\nThe total number of Training Data:',ytrain.shape)
print ('The total number of Test Data:',ytest.shape)

print(xtrain,ytrain)
classif = GaussianNB().fit(xtrain,ytrain)
print(classif)
predicted = classif.predict(xtest)
pri_enc = le.fit_transform(['sunny','cool','high','strong'])

predictTestData= classif.predict([pri_enc])

print('\nConfusion matrix')
print(metrics.confusion_matrix(ytest,predicted))

print('\nAccuracy of the classifier:',metrics.accuracy_score(ytest,predicted))

print('The value of Precision:', metrics.precision_score(ytest,predicted))

print('The value of Recall:', metrics.recall_score(ytest,predicted))

print("Predicted Value for individual Test Data:", predictTestData)