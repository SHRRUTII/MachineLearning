import pandas as pd
training_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/train.csv')
test_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/test.csv')
combined_data=[training_data,test_data]
pd.set_option('display.max_rows',None)
print(training_data)
print(test_data)
submission_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/gender_submission.csv')
print(submission_data)
combined=pd.DataFrame(combined_data)
combined.to_csv('C:/Users/shrut/Documents/MachineLearning/HW1/combined.csv')
pd.set_option('display.max_rows',None)
print(training_data)
print(test_data)
list(training_data.columns)
import numpy as nm
import copy as cp
print(training_data.info())
training_data.select_dtypes(nm.number)
training_data.select_dtypes(nm.number).head()
null_columns=training_data.columns[training_data.isnull().any()]
print(training_data[training_data.isnull().any(axis=1)][null_columns].head())
training_data.isnull().sum()
training_data.select_dtypes(nm.number).count()
training_data.select_dtypes(nm.number).mean()
training_data.select_dtypes(nm.number).median()
training_data.select_dtypes(nm.number).min()
training_data.select_dtypes(nm.number).std()
training_data.select_dtypes(nm.number).quantile(0.25)
training_data.select_dtypes(nm.number).quantile(0.75)
training_data.select_dtypes(nm.number).quantile(0.50)
training_data.select_dtypes(nm.number).max()
pd.value_counts(training_data.Cabin)
pd.value_counts(training_data.Name)
pd.value_counts(training_data.Sex)
pd.value_counts(training_data.Ticket)
pd.value_counts(training_data.Embarked)
training_data.select_dtypes(include=['object']).nunique(axis=0)
training_data.Name.value_counts().idxmax()
training_data.Sex.value_counts().idxmax()
training_data.Ticket.value_counts().idxmax()
training_data.Cabin.value_counts().idxmax()
training_data.Embarked.value_counts().idxmax()
