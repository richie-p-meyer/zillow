#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

import os
import env


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


