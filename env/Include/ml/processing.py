# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Datasets processing functions
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def cat2Number(df, cat_cols):
  label = LabelEncoder()
  # transform words to numbers
  for col in cat_cols:
    df[col] = label.fit_transform(df[col]) 
  return df
  
def cat2Dummy(df, cat_cols):
  # Encode Categorical Data
  for col in cat_cols:
    data1_dummy = pd.get_dummies(df[[col]], columns=[col])
    # Avoid Dummy variable trap
    dummyCols = data1_dummy.columns.tolist()[1:]
    df[dummyCols] = data1_dummy[dummyCols]
    df = df.drop(col, axis=1)
  return df

def fScaling(X_train, X_test):
  # Feature Scaling
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  return X_train, X_test