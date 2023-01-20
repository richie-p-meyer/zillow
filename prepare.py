import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import env

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def prepare_zillow(df):

    # Lose very small amount dropping nan values
    df = df.dropna()
    # Drop all rows with either no bedrooms or bathrooms
    df = df.drop(df[(df.bedroomcnt==0) | (df.bathroomcnt==0)].index)
    # Drop all properties under 250 square feet (this is a small tiny home)
    df = df.drop(df[df.calculatedfinishedsquarefeet<250].index)
    # Drop all properties built before 1850 - suspicious
    df = df.drop(df[df.yearbuilt<1850].index)
    # Drop properties that have more bathrooms than bedrooms
    df = df.drop(df[df.bedroomcnt<df.bathroomcnt].index)
    # Drop rows where lot size is less than house size
    df = df.drop(df[df.lotsizesquarefeet<df.calculatedfinishedsquarefeet].index)
    df.columns = ['bed', 'bath', 'squarefeet', 'lotsquarefeet', 'value', 'yearbuilt', 'fips'] 
    return df

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

def split_data(df, target):
    '''
    Splits a df into a train, validate, and test set. 
    target is the feature you will predict
    '''
    train_validate, test = train_test_split(df, train_size =.8, random_state = 21)
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 21)
    X_train = train.drop(columns=target)
    X_train = pd.get_dummies(X_train, columns=['fips'],drop_first=True)
    y_train = train[target]
    
    X_val = validate.drop(columns=target)
    X_val = pd.get_dummies(X_val, columns=['fips'],drop_first=True)
    y_val = validate[target]
    X_test = test.drop(columns=target)
    X_test = pd.get_dummies(X_test, columns=['fips'],drop_first=True)
    y_test = test[target]
    
    X_train,X_val,X_test = scale_minmax(X_train,X_val,X_test)
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    
    return train, X_train, y_train, X_val, y_val, X_test, y_test

def new_features(df):
    '''
    adds 'bb_sqft' column which is (beds+bath) / squarefeet
    add 'hsf_lsf' column which is squarefeet/lotsquarefeet
    '''
    df['bb_sqft'] = (df['bed']+df['bath'])/df['squarefeet']
    df['hsf_lsf'] = df.squarefeet/df.lotsquarefeet
    return df

def calc_rmse(value,pred):
    '''
    Calculate rmse given two series: actual values and predicted values
    '''
    return mean_squared_error(value,pred)**(1/2)


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
