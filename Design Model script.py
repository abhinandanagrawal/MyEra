#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

os.getcwd()


# In[2]:


Images=[]
filepath = 'D:/ANUPAM/study/summer internship/image/image 47.png'
img=image.load_img(filepath)
img=img.resize((224,224))
Images.append(img)
Images
Images[0]=image.img_to_array(img)/255.0
Images= np.array(Images)


# In[3]:


Images.shape
VGG_Model = load_model('D:/ANUPAM/study/summer internship/model/VGG_for_preprocessing.hdf5', compile=False)
print(VGG_Model)
features = VGG_Model.predict(np.array(Images))
features.shape


# In[11]:


#1VGG_model_for_featureContrastHierarchy
features = VGG_Model.predict(np.array(Images))


VGG_Model_Contrast = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureContrastHierarchy.hdf5', compile=False)
print(VGG_Model_Contrast.predict(features))
a1 = VGG_Model_Contrast.predict(features)


# In[27]:


#2VGG_model_for_featureFocalPoint
features = VGG_Model.predict(np.array(Images))


VGG_featureFocalPoint = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureFocalPoint.hdf5', compile=False)
print(VGG_featureFocalPoint.predict(features))
a2 = VGG_featureFocalPoint.predict(features)


# In[28]:


#3VGG_model_for_featureIntrigue
features = VGG_Model.predict(np.array(Images))


VGG_featureIntrigue = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureIntrigue.hdf5', compile=False)
print(VGG_featureIntrigue.predict(features))
a3 = VGG_featureIntrigue.predict(features)


# In[29]:


#4VGG_model_for_featureMacroWhiteSpace
features = VGG_Model.predict(np.array(Images))


VGG_featureMacroWhiteSpace = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureMacroWhiteSpace.hdf5', compile=False)
print(VGG_featureMacroWhiteSpace.predict(features))
a4 =VGG_featureMacroWhiteSpace.predict(features)


# In[30]:


#5VGG_model_for_featureMessage
features = VGG_Model.predict(np.array(Images))


VGG_model_for_featureMessage = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureMessage.hdf5', compile=False)
print(VGG_model_for_featureMessage.predict(features))
a5 =VGG_model_for_featureMessage.predict(features)


# In[31]:


#6VGG_model_for_featureSizeHierarchyGraphicAndText
features = VGG_Model.predict(np.array(Images))


VGG_model_for_featureSizeHierarchyGraphicAndText = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureSizeHierarchyGraphicAndText.hdf5', compile=False)
print(VGG_model_for_featureSizeHierarchyGraphicAndText.predict(features))
a6 =VGG_model_for_featureSizeHierarchyGraphicAndText.predict(features)


# In[32]:


#7 VGG_model_for_featureSizeHierarchyTypography
features = VGG_Model.predict(np.array(Images))


VGG_model_for_featureSizeHierarchyTypography = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureSizeHierarchyTypography.hdf5', compile=False)
print(VGG_model_for_featureSizeHierarchyTypography.predict(features))
a7 =VGG_model_for_featureSizeHierarchyTypography.predict(features)


# In[50]:


#8VGG_model_for_featureMicroWhiteSpace
features = VGG_Model.predict(np.array(Images))


VGG_featureMicroWhiteSpace = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureMicroWhiteSpace.hdf5', compile=False)
print(VGG_featureMicroWhiteSpace.predict(features))
a8 =VGG_featureMicroWhiteSpace.predict(features)


# In[53]:


#8VGG_model_for_featureMicroWhiteSpace
'''features = VGG_Model.predict(np.array(Images))


VGG_model_for_featureColourHierarchy = load_model('D:/ANUPAM/study/summer internship/model/VGG_model_for_featureColourHierarchy.hdf5', compile=False)
print(VGG_model_for_featureColourHierarchy.predict(features))
a9 =VGG_model_for_featureColourHierarchy.predict(features)'''


# In[8]:


a1 = VGG_model_for_featureSizeHierarchyTypography.predict(features)
print(a1)


# In[19]:


import pandas as pd
df = pd.DataFrame()
df['Contrast Hierarchy'] = a1[0]
print(df)


# In[51]:


# Creating Empty DataFrame and Storing it in variable df
df = pd.DataFrame()
df['Contrast Hierarchy'] = a1[0]
df['Focal point'] = a2[0]
df['Intrigue'] = a3[0]
df['Macro White space'] = a4[0]
df['Message'] = a5[0]
df['Size Hierarchy- graphic and text'] = a6[0]
df['Size Hierarchy- typography'] = a7[0]
df['Micro White space'] = a8[0]
#df['Colour Hierarchy'] = a9[0]
df


# In[71]:


VGG_Model = load_model('D:/ANUPAM/study/summer internship/model/VGG_for_preprocessing.hdf5', compile=False)
print(VGG_Model)


# In[60]:



#Prediction svn

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


#Reading File
attrdata1 = df
attrdata1.tail()
attrdata1.shape


# In[106]:


attrdata
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
#random forest regression
from sklearn.ensemble import RandomForestRegressor


from sklearn.preprocessing import LabelEncoder


#removing missing rows
attrdata1 = attrdata.fillna(attrdata.mean())#counting
print(attrdata1.isnull().sum())
#median replacement for null values

#Reading File
attrdata = pd.read_csv("D:/ANUPAM/study/summer internship/filtered_data.csv")
attrdata.tail()
attrdata.shape


#removing missing rows
#attrdata = attrdata.drop(['Link'], axis = 1)
attrdata = attrdata.fillna(attrdata.mean())#counting
print(attrdata.isnull().sum())
#median replacement for null values

desc = attrdata.describe
attrdata


encoder = LabelEncoder()
#data cleaning and encoding
# Encoding
attrdata['Size Hierarchy- graphic and text'] = attrdata['Size Hierarchy- graphic and text'].replace(np.nan, 0)

attrdata['Size Hierarchy- graphic and text'] = encoder.fit_transform(attrdata['Size Hierarchy- graphic and text'])
attrdata['Size Hierarchy- typography'] = encoder.fit_transform(attrdata['Size Hierarchy- typography'])
attrdata['Contrast Hierarchy'] = encoder.fit_transform(attrdata['Contrast Hierarchy'])

attrdata['Focal point'] = encoder.fit_transform(attrdata['Focal point'])
attrdata['Intrigue'] = encoder.fit_transform(attrdata['Intrigue'])
attrdata['Message'] = encoder.fit_transform(attrdata['Message'])
attrdata['Macro white space'] = encoder.fit_transform(attrdata['Macro white space'])
attrdata['Micro White space'] = encoder.fit_transform(attrdata['Micro White space'])
attrdata


#from the processed data we have to separate the features and target column again.
X = attrdata.drop(columns=['Likes/followers','Likes'], axis=1)
Y = attrdata['Likes']

print(X)
#Splitting data – Train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=12344)

X_train
X_test

########################################################################################################################
#random forest regression
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_test, Y_test)#We are training the model with RBF'ed data

#Model Scores (accuracy)nv
model_scores_rfr={'random forest regression':rfr.score(X_test,Y_test)}
actual_likes = Y_test.tolist()
# predict on test data
prediction=rfr.predict(X_test)
    
actual = Y_test.tolist()
ID = X_test["Image ID"].tolist()
followers = X_test["Followers"].tolist()
print("##########Random Forest Regressor:##########")
print("Training accuracy:", model_scores_rfr)
for i in  range(0,len(Y_test)):
    print("IMAGE ID:",ID[i], "actual likes:", round(actual[i],3), "prediction likes:",round(prediction[i],3))


X_train


# In[ ]:





# In[107]:


desc = attrdata1.describe
desc


# In[113]:






#testing random forest

#from the processed data we have to separate the features and target column again.
df['Image ID'] = 0
df['Followers'] = 100
df

prediction = rfr.predict(df)
    
actual = prediction.tolist()
ID = test_data["Image ID"].tolist()
followers = test_data["Followers"].tolist()
print("##########Random Forest Regressor:##########")
print("Training accuracy:", model_scores_rfr)
for i in  range(0,len(test_data)):
    print("IMAGE ID:",ID[i], "prediction likes:",round(prediction[i],3))
prediction


# In[ ]:





# In[ ]:




