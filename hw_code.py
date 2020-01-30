Python 3.8.1 (tags/v3.8.1:1b293b6, Dec 18 2019, 22:39:24) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> training_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/train.csv')
>>> test_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/test.csv')
>>> combined_data=[training_data,test_data]
>>> pd.set_option('display.max_rows',None)
>>> 
>>> print(training_data)

>>> print(test_data)

>>> submission_data=pd.read_csv('C:/Users/shrut/Documents/MachineLearning/HW1/gender_submission.csv')
>>> 
KeyboardInterrupt
>>> print(submission_data)


>>> combined=pd.DataFrame(combined_data)
>>> combined.to_csv('C:/Users/shrut/Documents/MachineLearning/HW1/combined.csv')
>>> pd.set_option('display.max_rows',None)
>>> print(training_data)

>>> print(test_data)

>>> list(training_data.columns)
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
>>> import numpy as nm
>>> import copy as cp
>>> print(training_data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 66.2+ KB
None
>>> training_data.select_dtypes(nm.number)

>>> training_data.select_dtypes(nm.number).head()
   PassengerId  Survived  Pclass   Age  SibSp  Parch     Fare
0            1         0       3  22.0      1      0   7.2500
1            2         1       1  38.0      1      0  71.2833
2            3         1       3  26.0      0      0   7.9250
3            4         1       1  35.0      1      0  53.1000
4            5         0       3  35.0      0      0   8.0500
>>> null_columns=training_data.columns[training_data.isnull().any()]
>>> print(training_data[training_data.isnull().any(axis=1)][null_columns].head())
    Age Cabin Embarked
0  22.0   NaN        S
2  26.0   NaN        S
4  35.0   NaN        S
5   NaN   NaN        Q
7   2.0   NaN        S
>>> training_data.isnull().sum()
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
>>> training_data.select_dtypes(nm.number).count()
PassengerId    891
Survived       891
Pclass         891
Age            714
SibSp          891
Parch          891
Fare           891
dtype: int64
>>> training_data.select_dtypes(nm.number).mean()
PassengerId    446.000000
Survived         0.383838
Pclass           2.308642
Age             29.699118
SibSp            0.523008
Parch            0.381594
Fare            32.204208
dtype: float64
>>> training_data.select_dtypes(nm.number).median()
PassengerId    446.0000
Survived         0.0000
Pclass           3.0000
Age             28.0000
SibSp            0.0000
Parch            0.0000
Fare            14.4542
dtype: float64
>>> training_data.select_dtypes(nm.number).min()
PassengerId    1.00
Survived       0.00
Pclass         1.00
Age            0.42
SibSp          0.00
Parch          0.00
Fare           0.00
dtype: float64
>>> 
KeyboardInterrupt
>>> training_data.select_dtypes(nm.number).std()
PassengerId    257.353842
Survived         0.486592
Pclass           0.836071
Age             14.526497
SibSp            1.102743
Parch            0.806057
Fare            49.693429
dtype: float64
>>> training_data.select_dtypes(nm.number).quantile(0.25)
PassengerId    223.5000
Survived         0.0000
Pclass           2.0000
Age             20.1250
SibSp            0.0000
Parch            0.0000
Fare             7.9104
Name: 0.25, dtype: float64
>>> training_data.select_dtypes(nm.number).quantile(0.75)
PassengerId    668.5
Survived         1.0
Pclass           3.0
Age             38.0
SibSp            1.0
Parch            0.0
Fare            31.0
Name: 0.75, dtype: float64
>>> training_data.select_dtypes(nm.number).quantile(0.75)

PassengerId    668.5
Survived         1.0
Pclass           3.0
Age             38.0
SibSp            1.0
Parch            0.0
Fare            31.0
Name: 0.75, dtype: float64
>>> training_data.select_dtypes(nm.number).quantile(0.50)
PassengerId    446.0000
Survived         0.0000
Pclass           3.0000
Age             28.0000
SibSp            0.0000
Parch            0.0000
Fare            14.4542
Name: 0.5, dtype: float64
>>> training_data.select_dtypes(nm.number).max()
PassengerId    891.0000
Survived         1.0000
Pclass           3.0000
Age             80.0000
SibSp            8.0000
Parch            6.0000
Fare           512.3292
dtype: float64
>>> pd.value_counts(training_data.Cabin)

>>> pd.value_counts(training_data.Name)

>>> pd.value_counts(training_data.Sex)
male      577
female    314
Name: Sex, dtype: int64
>>> pd.value_counts(training_data.Ticket)

>>>  pd.value_counts(training_data.Embarked)
 
SyntaxError: unexpected indent
>>> pd.value_counts(training_data.Embarked)
S    644
C    168
Q     77
Name: Embarked, dtype: int64
>>> training_data.select_dtypes(include=['object']).nunique(axis=0)
Name        891
Sex           2
Ticket      681
Cabin       147
Embarked      3
dtype: int64
>>> training_data.Name.value_counts().idxmax()
'White, Mr. Percival Wayland'
>>> training_data.Sex.value_counts().idxmax()
'male'
>>> training_data.Ticket.value_counts().idxmax()
'347082'
>>>  training_data.Cabin.value_counts().idxmax()
 
SyntaxError: unexpected indent
>>> training_data.Cabin.value_counts().idxmax()
'C23 C25 C27'
>>> training_data.Embarked.value_counts().idxmax()
'S'
>>> training_data.Name.value_counts().idxmax()
'White, Mr. Percival Wayland'
>>> training_data.Ticket.value_counts().idxmax()
'347082'
>>> training_data.Cabin.value_counts().idxmax()
'C23 C25 C27'
>>> 