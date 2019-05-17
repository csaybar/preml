from __future__ import division, absolute_import, print_function
    
__all__ = ["DUplots"]

import missingno as msno
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
    
import itertools
import warnings

class DUplots(object):
  """
  Create Beautiful and helpful plots for fast Data Understanding!
  
  Parameters
  ----------
  db_train: pd.DataFrame, Training Dataset  
  db_test: pd.DataFrame, Testing Dataset
  precision: int, default 3, parameter for the _GetGroupedData method indicate 
             the precision at which to store and display the bins labels.
  threshold: float, default 0.3, minimum % difference required to count as trend change.
  """
  def __init__(self, 
               db_train,
               target = 'Target',
               db_test = None,
               bins = 10,
               features = [],               
               precision = 5,
               threshold = 0.3):
    self.db_train = db_train
    self.db_test = db_test
    self.target = target
    self.bins = bins
    self.features = features    
    self.precision = precision
    self.threshold = threshold
  
  # -----------------------------
  @property
  def target(self):
    return self.__target
  
  @property
  def bins(self):
    return self.__bins
    
  @property 
  def features(self):
    return self.__features
  
  @property
  def precision(self):
    return self.__precision
  
  @property
  def threshold(self):
    return self.__threshold
  
  # Setter functions -----------------------------
  @target.setter
  def target(self,target):
    if isinstance(target, str):
      self.__target = target
    else:
      raise Exception('target must be a string')  
      
  @bins.setter
  def bins(self, bins):
    if isinstance(bins, int):
      self.__bins = bins
    elif isinstance(bins, list):
      self.__bins = bins
    else:
      raise Exception('bins must be a int or list')
      
  @features.setter
  def features(self, features):
    if isinstance(features, list):
      self.__features = features
    else:
      raise Exception('features must be a list')

  @precision.setter
  def precision(self, precision):
    if isinstance(precision, int):
      self.__precision = precision
    else:
      raise Exception('precision must be a int')
    
  @threshold.setter
  def threshold(self, threshold):
    if isinstance(threshold, float):
      self.__threshold = threshold
    else:
      raise Exception('threshold must be a float')    
      
  # Methods -----------------------------------
  def _GetGroupedData(self, feature, target):
    """  
    Bins continuous features into equal sample size buckets and returns 
    the target mean in each bucket. Separates out nulls into another bucket.

    Parameters
    ----------  
    feature: PandasSeries. Categorical or Numeric feature
    target: PandasSeries. Target binary classification
    bins: Number bins required  

    Returns
    -------
    A pd.DataFrame  

    Examples
    --------
    >>> Comming Soon!
    """    
    
    if sum(target.isnull()) > 0:
      raise Exception('Target has NaN values!')
            
    df = pd.DataFrame({'feature': feature, 'target': target})
    df_base = df.copy()
    
    # "feature": categorical or object dtype.
    if str(feature.dtype) == 'object' or str(feature.dtype) == 'category':
      df = df.groupby('feature').agg({
          'feature': ['count'],
          'target': ['mean']
      }).reset_index()
      df.columns = ['Groups', 'Samples_in_bin', 'Target_mean']
      gr_unique = df['Groups'].tolist()
    # "feature": numeric dtype
    else:
      if isinstance(self.bins, int):
        df['groups'] = pd.qcut(df['feature'], self.bins, precision=self.precision)
      elif isinstance(self.bins, list):
        df['groups'] = pd.cut(df['feature'], self.bins, precision=self.precision)
      else:
        raise Exception('"bin" argument only supports either integer or list')
      df = df.groupby('groups').agg({
          'feature': ['count', 'mean'],
          'target': ['mean']
      }).reset_index()
      df.columns = ['Groups', 'Samples_in_bin', 'Feature_mean', 'Target_mean']
      # Extracting categories
      group_range = list(
          itertools.chain(*[(x.left, x.right)
                            for x in df['Groups'].cat.categories]))
      gr_unique = pd.Series(group_range).unique().tolist()
      
    # Adding nan info if this exists
    if sum(feature.isnull()) > 0:
      add_nan = df_base[df_base['feature'].isnull()].agg({
          'target': ['count', 'mean']
      }).values
      nan_info = pd.DataFrame({
          'Groups': np.NaN,
          'Samples_in_bin': add_nan[0],
          'Feature_mean': np.NaN,
          'Target_mean': add_nan[1]
      })
      return gr_unique, pd.concat([nan_info, df]).reset_index(drop=True)
    else:
      return gr_unique, df
      
  def _GetTrendCorrelation(self, train_grouped, test_grouped):
    """
    Calculates correlation between train and test trend of feature wrt target. See _GetGroupedData.
    
    Parameters
    ----------
    train_grouped: train grouped data
    test_grouped: test grouped data
    
    Return
    ------
    trend correlation between train and test group
    
    Examples
    --------
    >>> Coming soon!
    """
    try:
      cor = np.corrcoef(train_grouped['Target_mean'],test_grouped['Target_mean'])[1, 0]
      return cor
    except:      
      warnings.warn('An error occurred when calculating the correlation, 0 was returned')
      return 0
                       
  def _GetTrendChanges(self, train_grouped):
    """
    Calculates number of times the trend of feature wrt target changed direction.
    
    Parameters
    ----------
    train_grouped: train grouped data
    
    Return
    ------
    Number of trend chagnes for the feature
    
    Examples
    --------
    >>> Comming soon!
    """
    train_grouped = train_grouped[train_grouped['Groups']
                                  .notnull()].reset_index(drop=True)
    target_f = train_grouped['Target_mean']
    target_diffs = target_f.diff()[1:]  # Because zero index always be np.NaN
    max_diff = target_f.max() - target_f.min()
    
    target_diffs_mod = target_diffs.abs()
    low_change = target_diffs_mod < (self.threshold * max_diff)  
    target_diffs_norm = target_diffs.divide(target_diffs_mod)  
    target_diffs_lvl2 = target_diffs_norm[~low_change].diff()
        
    changes = target_diffs_lvl2.fillna(0).abs() / 2
    
    if ~np.isnan(changes.sum()):
      tot_trend_changes = int(changes.sum())
    else:
      tot_trend_changes = 0
      
    return tot_trend_changes
    
  def __draw_plot_train(self):    
      for feature in self.features:
        
        groups, input_data = self._GetGroupedData(feature = self.db_train[feature],target = self.db_train[self.target])
        trend_changes = self._GetTrendChanges(input_data)
        
        # Plot parameters -------
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(input_data["Target_mean"], marker='o')
        ax1.set_xticks(np.arange(len(input_data)))
        plt.xticks(rotation=45)
        ax1.set_xlabel('Bins of ' + feature)
        ax1.set_ylabel('Average of ' + self.target)
        ax1.set_xticklabels((input_data['Groups']).astype('str'))
        
        comment = "Trend changed " + str(trend_changes) + " times"          
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        
        ax1.text(
        0.05,
        0.95,
        comment,
        fontsize=12,
        verticalalignment='top',
        bbox=props,
        transform=ax1.transAxes)
                
        plt.title('Average of ' + self.target + ' wrt ' + feature)
        ax2 = plt.subplot(1, 2, 2)
        ax2.bar(
            np.arange(len(input_data)), input_data['Samples_in_bin'], alpha=0.5)
        ax2.set_xticks(np.arange(len(input_data)))
        ax2.set_xticklabels((input_data['Groups']).astype('str'))
        plt.xticks(rotation=45)
        ax2.set_xlabel('Bins of ' + feature)
        ax2.set_ylabel('Bin-wise sample size')
        plt.title('Samples in bins of ' + feature)
        plt.tight_layout()        
        
        plt.show()
      
  def __draw_plot_test(self):
      for feature in self.features:
        groups, input_data_train = self._GetGroupedData(feature = self.db_train[feature],target = self.db_train[self.target])
        _, input_data_test = self._GetGroupedData(feature = self.db_test[feature],target = self.db_test[self.target])
        trend_changes_train = self._GetTrendChanges(input_data_train)
        trend_changes_test = self._GetTrendChanges(input_data_test)
        trend_correlation = self._GetTrendCorrelation(input_data_train, input_data_test)
        
        xlimits = np.array([input_data_train['Target_mean'].min(), input_data_train['Target_mean'].max(),
                   input_data_test['Target_mean'].min(), input_data_test['Target_mean'].max()])
        extra_space = (xlimits.max() - xlimits.min())*0.05
        xlimits_p1 = (xlimits.min()-extra_space,xlimits.max()+extra_space)
        
        # Plot parameters _train -------
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(input_data_train["Target_mean"], marker='o')
        ax1.set_xticks(np.arange(len(input_data_train)))
        plt.xticks(rotation=45)
        ax1.set_xlabel('Bins of ' + feature)
        ax1.set_ylabel('Average of ' + self.target)
        ax1.set_xticklabels((input_data_train['Groups']).astype('str'))
        ax1.set_ylim(xlimits_p1)
        
        comment = "Trend changed " + str(trend_changes_train) + " times"          
        comment = comment + '\n' + 'Correlation with train trend: ' + str(int(trend_correlation * 100)) + '%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        
        ax1.text(
        0.05,
        0.95,
        comment,
        fontsize=12,
        verticalalignment='top',
        bbox=props,
        transform=ax1.transAxes)
                
        plt.title('DB_TRAIN - Average of ' + self.target + ' wrt ' + feature)
        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(
            np.arange(len(input_data_train)), input_data_train['Samples_in_bin'], alpha=0.5)
        ax2.set_xticks(np.arange(len(input_data_train)))
        ax2.set_xticklabels((input_data_train['Groups']).astype('str'))
        plt.xticks(rotation=45)
        ax2.set_xlabel('Bins of ' + feature)
        ax2.set_ylabel('Bin-wise sample size')        
        plt.title('DB_TRAIN - Samples in bins of ' + feature)
        plt.tight_layout()                
        plt.show()
        
        # Plot parameters _test -------
        
        plt.figure(figsize=(12, 5))
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(input_data_test["Target_mean"], marker='o')
        ax3.set_xticks(np.arange(len(input_data_test)))
        plt.xticks(rotation=45)
        ax3.set_xlabel('Bins of ' + feature)
        ax3.set_ylabel('Average of ' + self.target)
        ax3.set_xticklabels((input_data_test['Groups']).astype('str'))
        ax3.set_ylim(xlimits_p1)
        
        comment = "Trend changed " + str(trend_changes_test) + " times"          
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        
        ax1.text(
        0.05,
        0.95,
        comment,
        fontsize=12,
        verticalalignment='top',
        bbox=props,
        transform=ax1.transAxes)
                
        plt.title('DB_TEST - Average of ' + self.target + ' wrt ' + feature)
        ax4 = plt.subplot(2, 2, 4)
        ax4.bar(
            np.arange(len(input_data_test)), input_data_test['Samples_in_bin'], alpha=0.5)
        ax4.set_xticks(np.arange(len(input_data_test)))
        ax4.set_xticklabels((input_data_test['Groups']).astype('str'))
        plt.xticks(rotation=45)
        ax4.set_xlabel('Bins of ' + feature)
        ax4.set_ylabel('Bin-wise sample size')        
        plt.title('DB_TEST - Samples in bins of ' + feature)
        plt.tight_layout()            
        plt.show()
        
      
        
  def plot(self,features = None, target = None):
    """
    Draws univariate dependence plots for a feature (include data_test)

    Parameters
    ----------
    input_data: grouped data contained bins of feature and target mean.
    feature: feature column name
    target_col: target column
    trend_correlation: correlation between train and test trends of feature wrt target

    Return
    ------
    Draws trend plots for feature

    Examples
    --------
    >>> Comming Soon!
    """                
    if isinstance(features,list):
      self.features = features
    if isinstance(target, str):
      self.target = target
          
    if self.db_test is None:
      return self.__draw_plot_train()   
    else:
      return self.__draw_plot_test()
    
  def __table_train(self):
    agg_features = dict()
    for feature in self.features:
      groups, input_data_train = self._GetGroupedData(feature = self.db_train[feature],target = self.db_train[self.target])
      try:
        input_data_train.columns = [feature, 'Samples_in_bin', self.target+'_mean']            
      except:
        input_data_train.columns = [feature, 'Samples_in_bin', feature+'_mean', self.target+'_mean']  
      agg_features[feature] = input_data_train      
    return agg_features
  
  def __table_test(self):
    agg_features_train = dict()
    agg_features_test = dict()
    for feature in self.features:
      # db_train case
      groups, input_data_train = self._GetGroupedData(feature = self.db_train[feature],target = self.db_train[self.target])      
      try:
        input_data_train.columns = [feature, 'Samples_in_bin', self.target+'_mean']            
      except:
        input_data_train.columns = [feature, 'Samples_in_bin', feature+'_mean', self.target+'_mean']  
      agg_features_train[feature] = input_data_train      
      
      # db_test case
      self.bins =  groups
      _, input_data_test = self._GetGroupedData(feature = self.db_test[feature],target = self.db_test[self.target])
      try:
        input_data_test.columns = [feature, 'Samples_in_bin', self.target+'_mean']            
      except:
        input_data_test.columns = [feature, 'Samples_in_bin', feature+'_mean', self.target+'_mean']  
      agg_features_test[feature] = input_data_test                  
    return dict(db_train=agg_features_train,db_test=agg_features_test)
  
  def tables(self):
    """
    Calculates summaries tables for list of features
    
    Return
    ------
    A dictionary 

    Examples
    --------
    >>> Comming Soon!
    """                
    if self.db_test is None:
      return self.__table_train()   
    else:
      return self.__table_test()

  def __cor_train(self, remove = [], fig_size = (12,5), annot = True):
    clean_db = self.db_train.drop(remove,axis=1)
    corr = clean_db.corr()
  
    #Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
  
    # Set up the matplotlib figure
    fig = plt.figure(figsize=fig_size)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
  
    # Draw the heatmap with the mask and correct aspect ratio    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, annot=annot, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})  
    plt.show()
  
  
  def __cor_test(self, remove = [], fig_size = (12,5), annot =True):
    clean_db_train = self.db_train.drop(remove,axis=1)
    corr_train = clean_db_train.corr()
    
    clean_db_test = self.db_test.drop(remove,axis=1)
    corr_test = clean_db_test.corr()
    
    #Generate a mask for the upper triangle
    mask_train = np.zeros_like(corr_train, dtype=np.bool)
    mask_train[np.triu_indices_from(mask_train)] = True
  
    mask_test = np.zeros_like(corr_test, dtype=np.bool)
    mask_test[np.triu_indices_from(mask_test)] = True
    
    
    # Set up the matplotlib figure
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title('DB_TRAIN')
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title('DB_TEST')
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
  
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_train, mask=mask_train, cmap=cmap, vmax=.3, annot=annot, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},ax = ax1)
    sns.heatmap(corr_test, mask=mask_test, cmap=cmap, vmax=.3, annot=annot, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},ax = ax2)
    plt.show()
    
  def corplot(self, remove = [], fig_size = (12,5),annot=True):
    """
    Create a cor plot easily!
    
    Return
    ------
    A dictionary 

    Examples
    --------
    >>> Comming Soon!
    """                
    if self.db_test is None:
      return self.__cor_train(remove=remove, fig_size=fig_size, annot=annot)
    else:
      return self.__cor_test(remove=remove, fig_size=fig_size, annot=annot)
    
  def missingplot(self, remove = [], figsize = (12,5),**kwargs):
    if self.db_test is None:
      clean_db_train = self.db_train.drop(remove,axis=1)
      msno.matrix(clean_db_train,kwargs,figsize=figsize)
    else:
      clean_db_test = self.db_test.drop(remove,axis=1)
      clean_db_train = self.db_train.drop(remove,axis=1)
      msno.matrix(clean_db_train,kwargs,figsize=figsize)
      msno.matrix(clean_db_test,kwargs,figsize=figsize)  
  
  def missingbar(self, remove = [], figsize = (12,5),**kwargs):
    if self.db_test is None:
      clean_db_train = self.db_train.drop(remove,axis=1)
      msno.bar(clean_db_train,figsize,**kwargs)
    else:
      clean_db_test = self.db_test.drop(remove,axis=1)
      clean_db_train = self.db_train.drop(remove,axis=1)
      msno.bar(clean_db_train,figsize,**kwargs)
      msno.bar(clean_db_test,figsize,**kwargs)
  def catplot(self):
    if self.db_test is None:
      pass
    else:
      fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
      df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=axis0)
      sns.countplot(x=feature_name, hue=target_name, data=df,
                    palette=palettemap,ax=axis1)
      plt.show()
  def contplot(self):
    pass
