# EX-01 Introduction to Kaggle and Data preprocessing
### AIM:
To perform Data preprocessing in a data set downloaded from Kaggle. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE:**
### EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
### RELATED THEORETICAL CONCEPT:
**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.<br><br>
**Data Preprocessing:**
Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing.<br><br>
**Need of Data Preprocessing :**
For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
### ALGORITHM:
STEP 1:Importing the libraries.<BR>
STEP 2:Importing the dataset.<BR>
STEP 3:Taking care of missing data.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  **Developed By: SIVARAM R**<br>
STEP 4:Encoding categorical data.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Register No: 212222100050**<br>
STEP 5:Normalizing the data.<BR>
STEP 6:Splitting the data into test and train.<BR>
###  PROGRAM:
```Python
import pandas as pd                                                 # Importing Libraries
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         # Read the dataset from drive
df.head()
df.isnull().sum()                                                   # Finding Missing Values
df.duplicated().sum()                                               # Check For Duplicates
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               # Remove Unnecessary Columns
scaler=StandardScaler()                                             # Normalize the dataset
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     # Split the dataset into input and output
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   # Splitting the data for training & Testing
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                     # X Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   # Y Train and Test
```
### OUTPUT:
**DATASET:** <br>
<img eight=45% idth=34% src="https://github.com/deepikasrinivasans/Ex-1-NN/assets/119393935/345e8e91-5dfb-4c01-9005-73a32e2cb82b"><br>\
<br>
**NULL VALUES:** &emsp;&emsp;&emsp;&emsp;&emsp; **NORMALIZED DATA:** <br>
<img eight=45% idth=34% src="https://github.com/deepikasrinivasans/Ex-1-NN/assets/119393935/4408e509-39ba-4ae1-b8d2-945bd7120fa9">
<img eight=45% idth=34% align=top src="https://github.com/deepikasrinivasans/Ex-1-NN/assets/119393935/3a7054ee-15da-4b4e-8194-4b5524fdc3dc"><br>
**DATA SPLITTING:** <br>
![nn4](https://github.com/deepikasrinivasans/Ex-1-NN/assets/119393935/1d766ecd-e713-4890-a97c-8da11d501cfb)
**TRAIN AND TEST DATA:** <br>
![nn5](https://github.com/deepikasrinivasans/Ex-1-NN/assets/119393935/6cfa9c47-ccf7-415c-98b6-c50ce1ccfa61)
### RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
