from __future__ import division, absolute_import, print_function

__all__ = ['fast_view']

from collections import Counter
import pandas as pd
import numpy as np

def fast_view(df):
  '''
    Fast View features
  '''
  object_describe, numeric_describe = None,None
  cnt = Counter(df.dtypes)
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
    numeric_col_n = types[~types.isin(['object'])].index
    numeric_features = df.iloc[:, numeric_col_n]
    f_numeric_name = numeric_features.columns
    
    # add NaN analysis -- numeric                    
    num_type_len = {x:(sum(numeric_features[x].isin([np.NaN]))>=1,
                       np.round(sum(numeric_features[x].isin([np.NaN]))/df.shape[0],3)
                      ) for x in f_numeric_name}
                    
    resumen_pd_num = pd.DataFrame(num_type_len, index = ['NaN_exist?','%perc_NA'])
    other_info_num = df.describe()
    numeric_describe = pd.concat([other_info_num, resumen_pd_num], sort = True)
    
  return object_describe, numeric_describe
