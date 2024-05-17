#!/usr/bin/env python
# coding: utf-8

# # ML Task-01

# Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
# 
# Dataset : - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# In[302]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[303]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[304]:


df_train.head()


# In[305]:


df_test.head()


# In[306]:


sns.heatmap(df_train.isnull())


# In[307]:


df_train.columns


# In[308]:


null_columns=[]
for i in df_train.columns.tolist():
    if df_train[i].isnull().sum() >= 500:
        null_columns.append(i)
        print(i,df_train[i].isnull().sum())


# In[309]:


df_train = df_train.drop(columns=null_columns)
df_test = df_test.drop(columns=null_columns)


# In[310]:


df_train_cleaned = df_train.ffill()
df_test_cleaned = df_test.ffill()


# In[311]:


df_train_cleaned.head()


# In[312]:


sns.heatmap(df_train_cleaned.isnull())


# In[313]:


df_train_cleaned.describe()


# In[314]:


plt.title('Salary Distribution Plot')
sns.histplot(df_train_cleaned['SalePrice'])
plt.show()


# # Split data 

# In[315]:


from sklearn.model_selection import train_test_split


# In[316]:


features = ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']

target = 'SalePrice'


# In[317]:


X = df_train_cleaned[features]
y = df_train_cleaned[target]


# In[318]:


df = pd.concat([X, y], axis=1)

# Plotting grid of scatter plots for each feature against the target variable
sns.pairplot(df, x_vars=features, y_vars=[target], kind='scatter', 
             plot_kws={'alpha':0.4}, diag_kws={'alpha':0.55, 'bins':40})

plt.show()


# In[319]:


sns.lmplot(x='GrLivArea', 
           y='SalePrice', 
           data=df_train_cleaned,
           scatter_kws={'alpha':0.3})


# In[320]:


sns.lmplot(x='BedroomAbvGr', 
           y='SalePrice', 
           data=df_train_cleaned,
           scatter_kws={'alpha':0.3})


# In[321]:


sns.lmplot(x='FullBath', 
           y='SalePrice', 
           data=df_train_cleaned,
           scatter_kws={'alpha':0.3})


# In[322]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Train model (Linear Regression)

# In[323]:


model = LinearRegression()


# In[324]:


model.fit(X_train, y_train)


# In[325]:


coefficients = model.coef_


# In[326]:


model.score(X, y)


# In[327]:


for feature, coef in zip(features, coefficients):
    print(f'{feature}: {coef}')


# # Predict results

# In[328]:


predictions = model.predict(X_test)


# In[329]:


predictions


# In[330]:


sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual prices of houses vs. Model Predictions')
plt.show()


# # Evaluation of the model

# In[331]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# In[332]:


print('Mean Absolute Error:',mean_absolute_error(y_test, predictions))
print('Mean Squared Error:',mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:',math.sqrt(mean_squared_error(y_test, predictions)))


# In[333]:


residuals = y_test-predictions
sns.histplot(residuals, bins=30)


# In[334]:


import pylab 
import scipy.stats as stats

stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()


# # Use the trained model on our test data 

# In[335]:


Xt = df_test_cleaned[features]


# In[336]:


test_predictions = model.predict(Xt)


# In[337]:


test_predictions


# In[338]:


plt.title('Salary Distribution Plot')
sns.histplot(test_predictions)
plt.show()


# In[339]:


submission_df = pd.DataFrame({
    'Id': df_test_cleaned['Id'],
    'SalePrice': test_predictions
})

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)


# # Conclusion 

# The current linear regression model provides a basic understanding of house price predictions, but the relatively high error metrics indicate room for improvement.

# In[ ]:




