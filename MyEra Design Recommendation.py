#!/usr/bin/env python
# coding: utf-8

# In[1]:



#Importing librarires
import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder


# In[2]:


#Reading File
attrdata = pd.read_csv("D:/ANUPAM/study/summer internship/updated dataset/dataset1.csv")
attrdata.tail()
attrdata.shape


# In[3]:



#removing missing rows
attrdata = attrdata.drop(['Link'], axis = 1)
attrdata = attrdata.fillna(attrdata.mean())#counting
print(attrdata.isnull().sum())
#median replacement for null values


# In[4]:


desc = attrdata.describe
attrdata


# In[5]:


encoder = LabelEncoder()
#data cleaning and encoding
# Encoding
attrdata['Size Hierarchy- graphic and text'] = attrdata['Size Hierarchy- graphic and text'].replace(np.nan, 0)
attrdata['Size Hierarchy- graphic and text'] = encoder.fit_transform(attrdata['Size Hierarchy- graphic and text'])
attrdata['Size Hierarchy- typography'] = encoder.fit_transform(attrdata['Size Hierarchy- typography'])
attrdata['Contrast Hierarchy'] = encoder.fit_transform(attrdata['Contrast Hierarchy'])
attrdata['Colour Hierarchy'] = encoder.fit_transform(attrdata['Colour Hierarchy'])
attrdata['Focal point'] = encoder.fit_transform(attrdata['Focal point'])
attrdata['Intrigue'] = encoder.fit_transform(attrdata['Intrigue'])
attrdata['Message'] = encoder.fit_transform(attrdata['Message'])
attrdata['Macro white space'] = encoder.fit_transform(attrdata['Macro white space'])
attrdata['Colour harmony'] = encoder.fit_transform(attrdata['Colour harmony'])
attrdata['type of graphic'] = encoder.fit_transform(attrdata['type of graphic'])
attrdata['Backgroud'] = encoder.fit_transform(attrdata['Backgroud'])
attrdata['Theme'] = encoder.fit_transform(attrdata['Theme'])
attrdata['Promoted'] = encoder.fit_transform(attrdata['Promoted'])
attrdata['Repetition'] = encoder.fit_transform(attrdata['Repetition'])
attrdata['Micro White space'] = encoder.fit_transform(attrdata['Micro White space'])
attrdata


# In[6]:


#from the processed data we have to separate the features and target column again.
X = attrdata.drop(columns=['Likes/followers', 'Creator','Weightage','Text Tone'], axis=1)
Y = attrdata['Likes/followers']


#Splitting data – Train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=12344)

X_train




# In[7]:


Y_test


# In[8]:


from sklearn.linear_model import LinearRegression
#--------------------
lm = LinearRegression()
lm.fit(X_train,Y_train)
print(lm.intercept_,lm.coef_)


# In[9]:


# lets predict on test data
prediction=lm.predict(X_test)
Y_test


# In[15]:




#Model Scores (accuracy)nv
model_scores={'Linear Regression':lm.score(X_test,Y_test)
             }
print(model_scores)


#Linear Regression


lm = LinearRegression()

lm.fit(X_test, Y_test)#We are training the model with RBF'ed data

#Model Scores (accuracy)nv
model_scores_lm={'random forest regression':lm.score(X_test,Y_test)}

# predict on test data
prediction=lm.predict(X_test)


actual = Y_test.tolist()
followers = X_test["Followers"].tolist()
ID = X_test["Image ID"].tolist()
print("##########Linear Regressor:##########")
print("Training accuracy:", model_scores_lm)
for i in  range(0,len(Y_test)):
    print("IMAGE ID:",ID[i],"actual:", round(actual[i],3), "prediction:",round(prediction[i],3), "predicted likes:", round(Decimal(prediction[i]*followers[i]),5))


# In[16]:


#random forest regression
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_test, Y_test)#We are training the model with RBF'ed data

#Model Scores (accuracy)nv
model_scores_rfr={'random forest regression':rfr.score(X_test,Y_test)}
actual_likes = X_test['Likes'].tolist()
# predict on test data
prediction=rfr.predict(X_test)
    
actual = Y_test.tolist()
ID = X_test["Image ID"].tolist()
followers = X_test["Followers"].tolist()
print("##########Random Forest Regressor:##########")
print("Training accuracy:", model_scores_rfr)
for i in  range(0,len(Y_test)):
    print("IMAGE ID:",ID[i], "actual:", round(actual[i],3), "prediction:",round(prediction[i],3),'actual likes:',actual_likes[i],  "predicted likes:", round(Decimal(prediction[i]*followers[i]),5))


# In[18]:


from sklearn.tree import DecisionTreeRegressor
dtr= DecisionTreeRegressor()
dtr.fit(X_train,Y_train)
#Model Scores (accuracy) decision tree regressor
model_scores_dtr={' decision tree regressor':dtr.score(X_test,Y_test)
             }
print(model_scores_dtr)
actual_likes = X_test['Likes'].tolist()
# predict on test data
prediction=dtr.predict(X_test)

actual = Y_test.tolist()
followers = X_test["Followers"].tolist()
print("##########Decision Tree Regressor:##########")    
actual = attrdata["Likes/followers"].tolist()
print("Training accuracy:", model_scores_dtr)
for i in  range(0,len(Y_test)):
    print("actual:", round(actual[i],3), "prediction:",round(prediction[i],3), 'actual likes:',round(actual_likes[i],3),"predicted likes:", round(Decimal(prediction[i]*followers[i]),5))


# In[20]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

rmse_val = [] #to store rmse values for different k
for K in range(50):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(Y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[21]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
#n=10 optimal


# In[29]:


#predicting on the test set and creating submission file
print(model_scores)
prediction = model.predict(X_test)
actual_likes = X_test['Likes'].tolist()
for i in  range(0,len(Y_test)):
    print("actual:", round(actual[i],3), "prediction:",round(prediction[i],3), 'actual likes:',round(actual_likes[i],3),"predicted likes:", round(Decimal(prediction[i]*followers[i]),5))

    
#Model Scores (accuracy)
model_scores={'KNN Regression':model.score(X_test,Y_test)
             }

