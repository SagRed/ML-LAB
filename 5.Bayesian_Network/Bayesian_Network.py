import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

heart_Disease = pd.read_csv('./heart.csv')
heart_Disease = heart_Disease.replace('?',np.nan)

print('Sample instances from the dataset are given below')
print(heart_Disease.head())

print('\n Attributes and datatypes')
print(heart_Disease.dtypes)

model= BayesianModel([('age','heartdisease'),('sex','heartdisease'),('exang','heartdisease'),('cp','heartdisease'),('heartdisease','restecg'),('heartdisease','chol')])
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heart_Disease,estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')
Heart_Disease_test_infer = VariableElimination(model)

print('\n 1. Probability of HeartDisease given evidence= restecg')
q1=Heart_Disease_test_infer.query(variables=['heartdisease'],evidence={'restecg':1})
print(q1)

print('\n 2. Probability of HeartDisease given evidence= cp ')
q2=Heart_Disease_test_infer.query(variables=['heartdisease'],evidence={'cp':2})
print(q2)