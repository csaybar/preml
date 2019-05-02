from __future__ import division, absolute_import, print_function

__all__ = ['go_unique']

import pandas as pd
import numpy as np

def go_unique(df):
  '''
     Object unique
     
  Argument:
    df -> Data.Frame that contains 'object' features
  
  Return:
    A dictionary that contains unique values
  '''
  
  object_columns = (np.where(pd.Series([str(x) for x in df.dtypes.values])
                      .isin(['object']))[0])
                      
  dict_object_uniques = {x:df[x].unique() 
                         for x in df.iloc[:,object_columns].keys()}
  return dict_object_uniques
