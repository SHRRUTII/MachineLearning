import pandas as pd
import numpy as nm
import copy as cp
training_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/train.csv')
#nm.corrcoef(training_data.Pclass,training_data.Survived)
training_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
nm.dtype(training_data.Sex)
nm.dtype(training_data.Survived)
pd.get_dummies(training_data['Sex'])
import plotly as plot
import plotly.express as px
import matplotlib.pyplot as plt
#11
survivedbyage = training_data[training_data['Survived'] == 1]
nonsurvivedbyage=training_data[training_data['Survived']==0]
survivedbyage['Age'].hist(bins=range(80), color='blue', label='Survived')
plt.xlabel('Age')
plt.ylabel('No. of Survivors')
plt.title('Survivor Age Distribution')
plt.show()
nonsurvivedbyage['Age'].hist(bins=range(80), color='blue', label='NonSurvived')
plt.title('Non-Survivor Age Distribution')
plt.xlabel('Age')
plt.ylabel
plt.show()
print(training_data.isnull().sum())
age_wrangled = training_data[pd.notnull(training_data['Age'])]
print ('passengers in wrangled data by age: ' + str(len(age_wrangled.index)) + '\n')
#combined
agesurvived = sns.FacetGrid(age_wrangled, row="Survived", margin_titles=True)
bins = nm.linspace(0, 90, 13)
agesurvived.map(plt.hist, "Age", color="blue", bins=bins)
#12
import seaborn as sns
g = sns.FacetGrid(age_wrangled, row="Pclass", col="Survived", margin_titles=True)
bins = nm.linspace(0, 60, 13)
g.map(plt.hist, "Age", color="blue", bins=bins)
plt.show()
#13
embarkedsurvived = sns.FacetGrid(age_wrangled, row="Embarked", col="Survived", margin_titles=True)
bins = nm.linspace(0, 60, 13)
embarkedsurvived.map(plt.bar, "Sex","Fare", color="blue")
plt.show()
#14
training_data.Ticket.describe()
count      891
unique     681
top       1601
freq         7
test_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/test.csv')
training_cabin=training_data.Cabin.describe()
test_cabin=test_data.Cabin.describe()
print(training_cabin)
print(test_cabin)
#Age
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=nm.nan, strategy='mean')
training_data.Age = training_data.Age.replace(0, nm.NaN)
training_data.Age.fillna(training_data.Age.mean(), inplace=True)
print(training_data.Age.isnull().sum())
embarkment=training_data.Embarked.describe()
print(embarkment)
train_embark=training_data["Embarked"].fillna("S", inplace = True)
print(train_embark)
testdata=test_data.Fare
print(testdata.isnull().values.sum())
testdata = testdata.fillna(testdata.value_counts().index[0])
print(testdata.isnull().values.sum())
#training_data['FareBand'] = pd.cut(x=training_data['Fare'], bins=[-0.001, 7.91, 14.454, 31.0,512.329])
#print(training_data.FareBand)
#training_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived', ascending=True)
#print(training_data.FareBand)
fare=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/train.csv')
fare['FareBand'] = pd.qcut(fare['Fare'],4)
print(fare.FareBand)
fare[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived', ascending=True)
