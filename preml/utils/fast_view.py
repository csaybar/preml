from __future__ import division, absolute_import, print_function

__all__ = ["fast_view"]

from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import kurtosis 
from scipy.stats import kstest

def fast_view(df, verbose = True):
  """  
  `fast_view` generate descriptive statistics that summarize:
  
    - Central tendency
    - Dispersion
    - Shape
    - `NaN` values
    - Outliers
    - Normality 

  `fast_view` automatically detect both numeric and object series
  
  Parameters
  ----------
  df : pd.DataFrame
  verbose : enable showing shape information
  
  Returns
  -------
  A tuple (x, y) that contains:
    x -> summary statistics for object series
    y -> summary statistics for numeric series

  See Also
  --------
  pd.DataFrame.describe : A more general describe statistics.
  

  Examples
  --------
  >>> from preml.utils import fast_view
  >>> example = pd.DataFrame({'A':['a','c','a'],'B':[1,2,3]})
  >>> obj, num= fast_view(example)
  """
  object_describe, numeric_describe = None,None
  cnt = Counter(df.dtypes)
  if verbose:
    print('number of features %s' % df.shape[1])
    for typo, numb in cnt.items():
      print('type_feature:% s n:%s' % (typo,numb))
      
  features_name = df.columns
  types = pd.Series([str(x) for x in df.dtypes.values])
  
  # Object type analysis
  if sum(types == 'object') is not 0:
    
    object_col_n = types[types.isin(['object'])].index
    object_features = df.iloc[:,object_col_n]
    f_object_name = object_features.columns
    
    # add NaN analysis -- object  
    obj_type_len = {x:(sum(object_features[x].isin([np.NaN]))>=1,
                       np.round(sum(object_features[x].isin([np.NaN]))/df.shape[0],3)
                    ) for x in f_object_name}
    resumen_pd_obj = pd.DataFrame(obj_type_len, index = ['NaN_exist?','%perc_NA'])
    other_info_obj = df.describe(include=np.object)
    object_describe = pd.concat([other_info_obj, resumen_pd_obj], sort = True)
    
  # Numeric type analysis
  if sum(types == 'int64') is not 0 or sum(types == 'float64') is not 0:
    numeric_col_n = types[~types.isin(['object'])].index # index of numeric columns
    numeric_features = df.iloc[:, numeric_col_n] # numeric columns
    f_numeric_name = numeric_features.columns # numeric columns name
    
    # 1. add NaN analysis -- numeric                    
    NaN_dict = {x:(sum(numeric_features[x].isin([np.NaN]))>=1,
                       np.round(sum(numeric_features[x].isin([np.NaN]))/df.shape[0],3)
                      ) for x in f_numeric_name}
    
    NaN_pd = pd.DataFrame(NaN_dict, index = ['NaN_exist?','%perc_NA'])
    
    # 2. add outlier
    outlier_dict = {x:(np.sum(__tukey_outlier(numeric_features[x])),
                       np.sum(__zscore_outlier(numeric_features[x]))
                      ) for x in f_numeric_name}
    outlier_pd = pd.DataFrame(outlier_dict, index = ['tukey_outlier_1.5','zscore_outlier_7.5'])
                             
    # 3. add normality information
    normality_dict = {x:(__shapiro_ps(numeric_features[x]),
                         __agostino_ps(numeric_features[x]),
                         __kstest_ps(numeric_features[x]),
                         numeric_features[x].skew(skipna = True),
                         kurtosis(numeric_features[x],nan_policy='omit')
                         ) for x in f_numeric_name}
    normality_pd = pd.DataFrame(normality_dict, index = ['shapiro_pvalue','DAgostino_pvalue','kstest_pvalue','Skew','Kurtosis'])
    
    # 4. Simple describe
    other_info_num = df.describe()
    
    
    numeric_describe = pd.concat([other_info_num, NaN_pd, outlier_pd, normality_pd], sort = True)
    
  return object_describe, numeric_describe

def __kstest_ps(df):
  stat, p= kstest(df.dropna(), 'norm')
  return p

def __shapiro_ps(df):
  stat, p = shapiro(df.dropna())
  return p

def __agostino_ps(df):
  stat, p = normaltest(df.dropna())
  return p

def __tukey_outlier(ys, threshold = 1.5):
    quartile_1, quartile_3 = np.percentile(ys, [5, 95])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * threshold)
    upper_bound = quartile_3 + (iqr * threshold)
    return (ys > upper_bound) | (ys < lower_bound)

def __zscore_outlier(ys, threshold = 8):    
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.abs(z_scores) > threshold
