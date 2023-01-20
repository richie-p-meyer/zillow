#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import env

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

# In[7]:


def get_zillow_2017():   
    if os.path.exists('zillow_2017.csv'):
        return pd.read_csv('zillow_2017.csv',index_col=0)
    else:
        url = env.get_connection('zillow')
        query = "select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, lotsizesquarefeet, \
        taxvaluedollarcnt, yearbuilt, fips  from properties_2017 join propertylandusetype \
        using (propertylandusetypeid) join predictions_2017 using (parcelid) where   propertylandusedesc = \
        'Single Family Residential' and transactiondate like '2017%%'"
        df = pd.read_sql(query,url)
        df.to_csv('zillow_2017.csv')
        return df



def wrangle_zillow():

    df = get_zillow_2017()

    # Lose very small amount dropping nan values
    df = df.dropna()
    # Drop all rows with either no bedrooms or bathrooms
    df = df.drop(df[(df.bedroomcnt==0) | (df.bathroomcnt==0)].index)
    # Drop all properties under 250 square feet (this is a small tiny home)
    df = df.drop(df[df.calculatedfinishedsquarefeet<250].index)
    # Drop all properties where value is less than taxes
    df = df.drop(df[df.taxvaluedollarcnt<df.taxamount].index)
    # Drop all properties built before 1850 - suspicious
    df = df.drop(df[df.yearbuilt<1850].index)
    # Drop properties that have more bathrooms than bedrooms
    df = df.drop(df[df.bedroomcnt<df.bathroomcnt].index)
    
    df.columns = ['bed', 'bath', 'squarefeet', 'value', 'yearbuilt', 'fips'] 
    
    
    return df


def split_data(df, target):
    '''
    Splits a df into a train, validate, and test set. 
    target is the feature you will predict
    '''
    full = df
    train_validate, test = train_test_split(df, train_size =.8, random_state = 21)
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 21)
    X_train = train.drop(columns=target)
    y_train = train[target]
    X_val = validate.drop(columns=target)
    y_val = validate[target]
    X_test = test.drop(columns=target)
    y_test = test[target]
    
    X_train,X_val,X_test = scale_minmax(X_train,X_val,X_test)
    
    return train, X_train, y_train, X_val, y_val, X_test, y_test

def scale_minmax(train,validate,test):
    '''
    Takes in train, validate, and test sets and returns the minmax scaled dfs
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)
    train[train.columns] = scaler.transform(train[train.columns])
    validate[validate.columns] = scaler.transform(validate[validate.columns])
    test[test.columns] = scaler.transform(test[test.columns])
    
    return train, validate, test


def rfe(x,y,k):

    
    lm = LinearRegression()
    rfe = RFE(lm,n_features_to_select=k)
    rfe.fit(x,y)
    
    mask = rfe.support_
    
    return x.columns[mask]

def select_kbest(x,y,k):

    f_selector = SelectKBest(f_regression,k=k)
    f_selector.fit(x,y)
    mask = f_selector.get_support()
    return x.columns[mask]

