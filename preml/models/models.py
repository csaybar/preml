# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__all__ = ['bayesian_optimization','show_parameter', 'lr_decayp', 'lightgbm_fitparams', 
           'lightgbm_SS', 'optim_f1_score']

import lightgbm as lgbm
from sklearn.base import BaseEstimator


class bayesian_optimization(BaseEstimator):
  '''
        Hyperopt - Tree-structured Parzen Estimator model
  Bayesian search  over specified parameter values for an estimator.
  
  Parameters
  ----------
  params: A list of Parameter Expressions [('Parameter Expressions',**kwargs,type),(...),...].
          A "Parameter Expressions" is the form that Hyperopt define the search space, 
          for an extensive description of this parameter, see:
          https://github.com/hyperopt/hyperopt/wiki/FMin
          type: A String. Specify the Python data type of the hyperparameter.
          
  max_evals: An Integer. Allow up to this many function evaluations before returning. 
  
  cv_params: A dictionary that contains sklearn.model_selection.cross_val_score **kwargs.
  
  cv_stat: Statistical description function used to aggregate the 
           sklearn.model_selection.cross_val_score values.
  
  Example
  --------
  Comming soon!
  '''
  def __init__(self,params,max_evals, cv_params, cv_stat = np.mean):
    self.params = params
    self.max_evals = max_evals
    self.cv_params = cv_params
    self.cv_stat = cv_stat
    self.objective_model = self.__crt_objmodel()
  
  def create_params(self):
    params_dict = dict()   
    for hparams, param, _ in self.params:
      params_dict[param['label']]= eval(hparams)(**param)
    return params_dict
    
  def __crt_objmodel(self):            
    def obj_model(parameters):
      params = {param['label']:(eval(htype)(parameters[param['label']])) for _,param, htype in self.params}
      self.cv_params['estimator'] = self.cv_params['estimator'].set_params(**params)
      cv_values = cross_val_score(**self.cv_params)
      score = cv_stat(cv_values)
      print("Score: {:.3f} || params {}".format(score, params))      
      return score      
    return obj_model
  
  def fit(self):
    best = fmin(fn=self.__crt_objmodel(),
              space=self.create_params(),
              algo=tpe.suggest,
              max_evals=self.max_evals)
    #Changing the Python data type of dict values
    datatypes = [x[2] for x in self.params]
    count = 0    
    for key,value in best.items():
      best[key] = eval(datatypes[count])(value)
      count += 1    
    return self.cv_params['estimator'].set_params(**best)

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
  def optim_model(x):
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

