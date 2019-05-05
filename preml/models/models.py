# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__all__ = ['show_parameter', 'lr_decayp', 'lightgbm_fitparams', 
           'lightgbm_SS', 'optim_f1_score']

import lightgbm as lgbm

def show_parameter(pipeline, models = None):
  '''
  Show all parameters of your pipeline
  '''
  for x in pipeline.get_params().keys():
    if models == None:
      print(x)
    elif bool(re.search(str(models), x)):
      print(x)
    else:
      pass
      
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
  def c(x):
    pred_df = pd.DataFrame(model.predict_proba(data))
    y_pred = (pred_df.iloc[:,1].values > x)*1
    return f1_score(target, y_pred)
  rango = np.linspace(rgn[0], rgn[1], iterations)
  f1_scores = pd.Series([optim_model(z) for z in rango])
  return rango[f1_scores.idxmax()]

def lightgbm_fitparams(**kwargs):
  fit_params = {
              "early_stopping_rounds":30, 
              "eval_metric" : 'auc',
              "eval_set" : [(None,None)],
              'eval_names': ['valid'],
              'callbacks': [lgbm.reset_parameter(learning_rate=lr_decayp())],
              'verbose': 100,
              'categorical_feature': 'auto'
              }
  for key,value in kwargs.items():
    fit_params[key] = value
  return fit_params

def lightgbm_SS(**kwargs):
  search_spaces = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'num_leaves': (20, 100),      
    'max_depth': (10, 50),
    'min_child_samples': (5, 50),
    'max_bin': (100, 1000),
    'subsample': (0.01, 1.0, 'uniform'),
    'subsample_freq': (2, 10),
    'colsample_bytree': (0.01, 1.0, 'uniform'),
    'min_child_weight': (1, 10),
    'subsample_for_bin': (100000, 500000),
    'reg_lambda': (1e-9, 1000, 'log-uniform'),
    'reg_alpha': (1e-9, 1.0, 'log-uniform'),
    'scale_pos_weight': (1e-6, 500, 'log-uniform'),
    'n_estimators': (50, 100)}
    
  for key,value in kwargs.items():
    search_spaces[key] = value
  return search_spaces
