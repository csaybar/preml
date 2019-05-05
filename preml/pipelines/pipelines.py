# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__all__ = ['LabelEncoder_NaN', 'TargetEncoder_NaN']

from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import pandas as pd

class LabelEncoder_NaN(LabelEncoder):
  """
  Encode labels with value between 0 and n_classes-1.
  """
  def __init__(self):
    pass
    
  def fit(self, X, y=None):
    return self
    
  def transform(self, X):
    try:
      df = X.copy()
      df['__ref__'] = 1
      func = lambda x:pd.Series(
        data  = self.fit_transform(x[x.notnull()]),
        index = x[x.notnull()].index
      )
      df_encoded = df.apply(func)
      del df_encoded['__ref__']
      return df_encoded  
    except (AttributeError, TypeError):
      raise AssertionError('Input variables should be Pandas.Data.Frame')

class TargetEncoder_NaN(TargetEncoder):
  """
  Target encoding for categorical features.
  
  For the case of categorical target: features are replaced
  with a blend of posterior probability of the target given
  particular categorical value and prior probability of the
  target over all the training data.
    
  For  the  case  of  continuous  target: features are replaced
  with a blend of expected value of the target given particular
  categorical value and expected  value of the target  over all
  the training data.
  """
  def __init__(self,
               verbose = 1,
               cols = None,
               drop_invariant = False,
               return_df = True,
               handle_missing= 'return_nan',
               handle_unknown = 'error',
               min_samples_leaf = 10,
               smoothing = 10):
                 
    super().__init__(verbose = 1,
                     cols = cols,
                     drop_invariant = drop_invariant,
                     return_df = return_df,
                     handle_missing = handle_missing,
                     handle_unknown = handle_unknown,
                     smoothing = smoothing)
  def transform(self, X):
    return super().transform(X)
  def fit(self,X,y):
    return super().fit(X, y)
  def fit_transform(self, X, y):
    self.fit(X, y)
    return self.transform(X)
