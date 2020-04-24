import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor as ETRg
from matplotlib import rcParams
%matplotlib inline
le = preprocessing.LabelEncoder()
##ignore unimportant warnings that is given in console
warnings.filterwarnings('ignore')

############################################################################################################
#FILE LOAD into the kernels
print('Packages are loaded...')
print('Program is started to running...')
print('Reading titanic data files[train.csv, test.csv, gender.submission.csv]...')
##Load data from the folders into the program
data = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')
sampl = pd.read_csv('../input/titanic/gender_submission.csv')
print('Files are loaded successfully[train.csv, test.csv, gender.submission.csv]...')
############################################################################################################

############################################################################################################
#MERGING TRAIN AND TEST DATASETS
#We will do the same cleaning, replacing missing data and feature engineering on both training and testing data.
#So we are combing them together in a data frame named df.
df = data.append(test, sort = False)
############################################################################################################

############################################################################################################
#FIND OUT MISSING VALUES IN COMBINED DATASET
#Check total and percentage of missing values by column (feature).
#total_missing_by_column = df.isnull().sum().sort_values(ascending=False)
#percent_missing_by_column = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total_missing_by_column, percent_missing_by_column], axis=1, keys=['Total', 'Percent'])
#missing_data.head(10)
############################################################################################################

#################################################################################
#NEW FEATURE CREATION===>TicketId
#Creating a new feature TicketId which will bring people who are in the same family or in the same group
df_ticket = pd.DataFrame(df.Ticket.value_counts())
df_ticket.rename(columns = {'Ticket' : 'TicketNum'}, inplace = True)
df_ticket['TicketId'] = pd.Categorical(df_ticket.index).codes
df_ticket.loc[df_ticket.TicketNum < 3, 'TicketId'] = -1
df = pd.merge(left = df, right = df_ticket, left_on = 'Ticket', 
              right_index = True, how = 'left', sort = False)
df = df.drop(['TicketNum'],axis=1)
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>FamilySurv
#Separating the last name in "name" feature to look at statistics for family survivals
df['FamilyName'] = df.Name.apply(lambda x : str.split(x, ',')[0])
#If a passenger has no family member accompanied with him then he is assigned to 0.5.
#If a family member is survived or not survived, then he is assigned to 1.0 and 0.0 respectively.
df['FamilySurv'] = 0.5
for _, grup in df.groupby(['FamilyName','Fare']):
    if len(grup) != 1:
        for index, row in grup.iterrows():
            smax = grup.drop(index).Survived.max()
            smin = grup.drop(index).Survived.min()
            pid = row.PassengerId
            
            if smax == 1:
                df.loc[df.PassengerId == pid, 'FamilySurv'] = 1.0
            elif smin == 0:
                df.loc[df.PassengerId == pid, 'FamilySurv'] = 0.0
for _, grup in df.groupby(['Ticket']):
    if len(grup) != 1:
        for index, row in grup.iterrows():
            if (row.FamilySurv == 0.0 or row.FamilySurv == 0.5):
                smax = grup.drop(index).Survived.max()
                smin = grup.drop(index).Survived.min()
                pid  = row.PassengerId

                if smax == 1:
                    df.loc[df.PassengerId == pid, 'FamilySurv'] = 1.0
                elif smin == 0:
                    df.loc[df.PassengerId == pid, 'FamilySurv'] = 0.0
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>Family_Size
#Creating Family_Size feature since larger families may survive more likely
df['Family_Size'] = 0
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>Alone
#Creating Alone Feature since alone people may not likely to survive
df['Alone'] = 0
df.loc[df['Family_Size'] <= 1, 'Alone'] = 1
df.loc[df['Family_Size'] > 1, 'Alone'] = 0
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>Cabin_Number
#Finding the number of cabins each passenger has in the Titanic
df.Cabin = df.Cabin.fillna('0')
regex = re.compile('\s*(\w+)\s*')
df['Cabin_Number'] = df.Cabin.apply(lambda x : len(regex.findall(x)))
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>Title
#Grouping the different titles under Mr., Mrs., Miss. [Dr and Master titles are exluded since including both gender!]
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
mapping = {
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Lady': 'Mrs',
    'Countess': 'Mrs',
    'Dona': 'Mrs',
    'Major': 'Mr',
    'Col': 'Mr',
    'Sir': 'Mr',
    'Don': 'Mr',
    'Jonkheer': 'Mr',
    'Capt': 'Mr',
    'Rev': 'Mr',
    'General': 'Mr'}
df.replace({'Title': mapping}, inplace=True)
##Histogram of suvrivived against title
#train_set = df[0:891].copy()
#sns.set(style="whitegrid")
#plt.figure(figsize=(15,4))
#ax = sns.barplot(x="Title", y="Survived", data=train_set, ci=None)
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>Fare_Category
#Let's fill the null value for Fare with 7: Least number
df.loc[df['Fare'].isnull(), 'Fare'] = 7
#Creating the Fare_Catefory feature and adjusting the groups
df['Fare_Category'] = 0
df.loc[df['Fare'] < 8, 'Fare_Category'] = 0
df.loc[(df['Fare'] >= 8 ) & (df['Fare'] < 16),'Fare_Category' ] = 1
df.loc[(df['Fare'] >= 16) & (df['Fare'] < 30),'Fare_Category' ] = 2
df.loc[(df['Fare'] >= 30) & (df['Fare'] < 45),'Fare_Category' ] = 3
df.loc[(df['Fare'] >= 45) & (df['Fare'] < 80),'Fare_Category' ] = 4
df.loc[(df['Fare'] >= 80) & (df['Fare'] < 160),'Fare_Category' ] = 5
df.loc[(df['Fare'] >= 160) & (df['Fare'] < 270),'Fare_Category' ] = 6
df.loc[(df['Fare'] >= 270), 'Fare_Category'] = 7
##Histogram of suvrivived against Fare Category
#train_set = df[0:891].copy()
#sns.set(style="whitegrid")
#plt.figure(figsize=(16,4))
#ax = sns.barplot(x="Fare_Category", y="Survived", hue='Title', data=train_set, ci=None)
#################################################################################

#################################################################################
##Filling the null Age values using the data in old&new features
features = ['Pclass','SibSp','Parch','TicketId','Fare','Cabin_Number', 'Alone']
Etr = ETRg(n_estimators = 200, random_state = 2)
AgeX_Train = df[features][df.Age.notnull()]
AgeY_Train = df['Age'][df.Age.notnull()]
AgeX_Test = df[features][df.Age.isnull()]
#Train and predict the missing ages
Etr.fit(AgeX_Train, np.ravel(AgeY_Train))
AgePred = Etr.predict(AgeX_Test)
df.loc[df.Age.isnull(), 'Age'] = AgePred
#################################################################################

#################################################################################
#NEW FEATURE CREATION===>Age_Category
#Grouping the Ages and assigning to Age_Category feature
df['Age_Category'] = 0
df.loc[(df['Age'] <= 5), 'Age_Category'] = 0
df.loc[(df['Age'] <= 12) & (df['Age'] > 5), 'Age_Category'] = 1
df.loc[(df['Age'] <= 18) & (df['Age'] > 12), 'Age_Category'] = 2
df.loc[(df['Age'] <= 22) & (df['Age'] > 18), 'Age_Category'] = 3
df.loc[(df['Age'] <= 32) & (df['Age'] > 22), 'Age_Category'] = 4
df.loc[(df['Age'] <= 45) & (df['Age'] > 32), 'Age_Category'] = 5
df.loc[(df['Age'] <= 60) & (df['Age'] > 45), 'Age_Category'] = 6
df.loc[(df['Age'] <= 70) & (df['Age'] > 60), 'Age_Category'] = 7
df.loc[(df['Age'] > 70), 'Age_Category'] = 8
##Histogram of Survived against Age_Category
#train_set = df[0:891].copy()
#sns.set(style="whitegrid")
#plt.figure(figsize=(14,3.5))
#ax = sns.barplot(x="Age_Category", y="Survived",data=train_set, ci=None)
#################################################################################

#################################################################################
#ENCODING CATEGORICAL DATA
#Encoding Title, Cabin
lsr = {'Title','Cabin'}
for i in lsr:
    le.fit(df[i].astype(str))
    df[i] = le.transform(df[i].astype(str))

#Replace the missing Embarked with value 'C': Most first class passengers came from this port.
df.loc[(df.Embarked.isnull()),'Embarked']= 'C'
#Encoding Embarked, Sex
lst = {'Embarked','Sex'}
for i in lst:
    le.fit(df[i].astype(str))
    df[i] = le.transform(df[i].astype(str))
#################################################################################

#################################################################################
#Scaling up data using StandardScaler
target = data['Survived'].values
select_features = ['Pclass', 'Age','Age_Category','SibSp', 'Parch', 'Fare', 
                   'Embarked', 'TicketId', 'Cabin_Number', 'Title','Cabin',
                   'Fare_Category', 'Family_Size','FamilySurv','Sex', 'Alone']
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[select_features])
train = scaled_df[0:891].copy()
test = scaled_df[891:].copy()
#################################################################################

#################################################################################
#predict over the test data.
#INITIALIZING THE MODEL
random_forest = RandomForestClassifier(criterion = "gini",
                                       min_samples_leaf = 1,
                                       max_features='auto',
                                       max_depth = 5,
                                       min_samples_split = 4,
                                       n_estimators = 500,
                                       oob_score=True,
                                       random_state = 20,
                                       n_jobs = -1)
#Training the model with strandardized train data
random_forest.fit(train, target)
#Predicting the values on Train data to check Accuracy Score
prediction = random_forest.predict(train)
training_accuracy = accuracy_score(target, prediction)
print("Training accuracy: ", training_accuracy)

#Predicting the outcome values on Test data using Random Forest classifier
predict_over_test = random_forest.predict(test)
#Creating the submission.csv file to submit to Kaggle
sampl['Survived'] = pd.DataFrame(predict_over_test)
sampl.to_csv('submission.csv', index=False)
print("Output file has been saved as: submission.csv")