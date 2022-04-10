#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, preprocessing, model_selection
from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score,ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[42]:


df=pd.read_csv(r"C:\Users\Siddhi\Documents\DataSets\turnover.csv", encoding = "ISO-8859-1")
df.head()


# In[43]:


df.shape


# In[44]:


df.columns


# In[45]:


# Checking Duplicates in df

duplicate = df[df.duplicated()]
duplicate


# In[46]:


# Dropping the 13 duplicated rows

df.drop_duplicates(keep=False,inplace=True)
df


# In[47]:


# Identifying Categorical Variables 

df_cat = df.select_dtypes(include=['object']).copy()
df_cat.columns


# In[48]:


le = LabelEncoder()

df_cat['gender']  = le.fit_transform(df_cat['gender'])
df_cat['industry']  = le.fit_transform(df_cat['industry'])
df_cat['profession']  = le.fit_transform(df_cat['profession'])
df_cat['traffic']  = le.fit_transform(df_cat['traffic'])
df_cat['coach']  = le.fit_transform(df_cat['coach'])
df_cat['head_gender']  = le.fit_transform(df_cat['head_gender'])
df_cat['greywage']  = le.fit_transform(df_cat['greywage'])
df_cat['way']  = le.fit_transform(df_cat['way'])

df_cat.head()


# In[49]:


df_1 = df.drop(['gender', 'industry', 'profession', 'traffic', 'coach', 'head_gender', 'greywage', 'way'], axis = 1)
df_1.head()


# In[50]:


df_4 = pd.merge(df_1,df_cat,left_index=True, right_index=True)
df_4


# In[51]:


df_3 = df_4.drop('event', axis=1)  
X = df_4[['age','extraversion','independ','novator','gender','industry','profession','head_gender','way','greywage']] # Defining X axis
Y = df['event'].values
X.head()


# In[52]:


scaler = preprocessing.StandardScaler()
X2 = scaler.fit_transform(X)


# In[53]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)


# In[54]:


model_rf = RandomForestClassifier(n_estimators= 100, random_state=10, max_depth=13)
model_rf.fit(X_train,Y_train)

print('Precision of Random Forest: {:.3f}'.format(accuracy_score(Y_test, model_rf.predict(X_test))))


# In[55]:


print(classification_report(Y_test, model_rf.predict(X_test)))


# In[56]:


rfc = RandomForestClassifier(n_estimators = 500,criterion = "entropy")
rfc.fit(X_train,Y_train)


# In[57]:


predict_r = rfc.predict(X_test)
predict_r


# In[61]:


pickle.dump(rfc , open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[62]:


prediction = rfc.predict((np.array([[25,9,10,8,1,2,1,2,2,4]]))) # 'age','extraversion','independ','novator','gender','industry','profession','head_gender','way','greywage'
print("The employee will leave (1) or not leave (0): ",prediction)

