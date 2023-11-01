# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE

## DATA PREPROCESSING BEFORE FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/9ccb1b69-bb46-4190-8c02-111fe1f679eb)


## checking data
```
df.isnull().sum()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/2d9a5e2d-6fb1-43e8-a314-fe6d9514ac5f)


## removing unnecessary data variables
```
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/50a088a5-e625-4394-9cf9-5d5885ead773)


## cleaning data

```
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/32d7c85a-ffd7-41bd-88b5-c97062929e7c)


## removing outliers 
```
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/cf722590-11b5-4324-be9e-5a85787a9984)

```
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/e401084e-5bbf-4ca3-99a2-fc6548b5c4ad)

```
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/6375d649-c208-4421-8ed0-86132c9fabc3)

```
from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/239e0087-b2b8-4f7f-b230-e1d14c1fb299)

```
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/5057da95-3204-4ee3-aa73-a95e228d70b6)

```
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)
df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/080f8ae7-e74a-44d0-9f2c-c5d0f9de32db)

```
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
X = df1.drop("Survived",1) 
y = df1["Survived"]
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/35de38a1-6884-4302-bad1-3bfa2758529e)


##  FEATURE SELECTION:
##  FILTER METHOD:

```
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/b140e557-0a50-418a-bac0-f3f6fce1388e)


## HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/7feef14e-3a9e-4aef-9a48-f9f099aadd42)


## BACKWARD ELIMINATION:

```
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/e9a08c96-fb86-4cfd-b833-b541aa5bd6c7)


## RFE (RECURSIVE FEATURE ELIMINATION):

```
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/928806fd-cc21-42ee-98e6-c5a7a36386e9)


## OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:

```
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/06aaf6e3-3e2b-4cb9-b8cb-effce912f1ea)

## FINAL SET OF FEATURE:
```
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```

![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/d54ebe48-0a1a-4d8c-ac7c-ab7b78b196f6)


## EMBEDDED METHOD:

```
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex-07/assets/103943383/cd9bdd4a-01c7-402c-9211-f6a4c7929e10)


# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
