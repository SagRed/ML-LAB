
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


dataf = pd.read_csv("./data.csv")
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = dataf[feature_col_names].values
y = dataf[predicted_class_names].values

print(dataf.head)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)

print ('\nThe total number of Training Data:',ytrain.shape)
print ('The total number of Test Data:',ytest.shape)


classif = GaussianNB().fit(xtrain,ytrain.ravel())

predicted = classif.predict(xtest)

predictTestData= classif.predict([[5,148,72,35,0,32.6,0.543,50]])

print('\nConfusion matrix')
print(metrics.confusion_matrix(ytest,predicted))

print('\nAccuracy of the classifier:',metrics.accuracy_score(ytest,predicted))

print('The value of Precision:', metrics.precision_score(ytest,predicted))

print('The value of Recall:', metrics.recall_score(ytest,predicted))

print("Predicted Value for individual Test Data:", predictTestData)
