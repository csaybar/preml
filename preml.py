# -*- coding: utf-8 -*-

"""Main module."""
def get_object_unique(df):
  '''
    Object unique
  '''
  object_columns = (np.where(pd.Series([str(x) for x in df.dtypes.values])
                      .isin(['object']))[0])
                      
  dict_object_uniques = {x:df[x].unique() 
                         for x in df.iloc[:,object_columns].keys()}
  return dict_object_uniques

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


def optim_f1_score(model,data,target,iterations = 100, rgn = [0,1]):
  """
    optim f1-score
  arguments:
    - model      : Put your model.
    - data       : X_values
    - target     : y_values
    - iterations : Number of iteration (Force brute)
    - rgn        : Range searching for f1-threshold.
  return:
    Best f1-score found
  """
  def optim_model(x):
    thershold = x
    y_pred = (model.predict(data) > thershold)*1
    return f1_score(target, y_pred)
  rango = np.linspace(rgn[0], rgn[1], iterations)
  f1_scores = pd.Series([optim_model(z) for z in rango])
  optim_model(0.2)
  return rango[f1_scores.idxmax()]


def lr_decayp(current_iter = 0.001,lr = 0.05,dcp = 0.99):
  '''
    learning rate with decay power
    
  Arguments:
    -current_iter = Current iteration
    -lr = learning rate 
    -dcp = decay power
  return 
    function with only 'current_iter' as a argument
  '''
  def func_lr_dcp(current_iter = current_iter):
    nlr = lr  * np.power(dcp, current_iter)
    return nlr if nlr > 1e-3 else 1e-3
  return func_lr_dcp

def create_fitparams(**kwargs):
  fit_params={"early_stopping_rounds":30, 
              "eval_metric" : 'auc', 
              "eval_set" : [(None,None)],
              'eval_names': ['valid'],
              'callbacks': [lgbm.reset_parameter(learning_rate=lr_decayp())],
              'verbose': 100,
              'categorical_feature': 'auto'}
  for key,value in kwargs.items():
    fit_params[key] = value
  return fit_params
