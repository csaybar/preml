from __future__ import division, absolute_import, print_function

__all__ = ['go_unique']

import pandas as pd
import numpy as np
from scipy import stats

def re_transform(old_db,new_db,retransform=[]):
  """
  Retransform categorical and object dtypes after Bayesian-Iterative Imputation.

  Parameters
  ----------

  old_db: pd.DataFrame, Dataset before imputation.
  new_db: pd.DataFrame, Dataset after imputation.
  retransform: list, columns name to retransform

  Return
  ------
  pd.DataFrame
  """  
  for x in retransform:
    dfn = pd.concat([old_db[x],new_db[x]],axis = 1)
    dfn.columns = ['old','new']
    
    grouping = dfn.groupby(['old']).agg(lambda x:stats.mode(x)[0]) #get mode
    nan_newvalues = dfn['new'][dfn['old'].isnull()].reset_index(drop=True)

    values_imputate = [(grouping['new'] - z).idxmin() for z in nan_newvalues]
    dfn['old'][dfn['old'].isnull()] = values_imputate
    new_categ = dfn['old']
    new_db.loc[:,x] = new_categ
  return new_db
