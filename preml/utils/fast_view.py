from __future__ import division, absolute_import, print_function

__all__ = ["fast_view", "shapiro_ps","agostino_ps", "zscore_outlier", "tukey_outlier",]

from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import kurtosis 


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
    outlier_dict = {x:(np.sum(tukey_outlier(numeric_features[x])),
                       np.sum(zscore_outlier(numeric_features[x]))
                      ) for x in f_numeric_name}
    outlier_pd = pd.DataFrame(outlier_dict, index = ['tukey_outlier_1.5','zscore_outlier_7.5'])
                             
    # 3. add normality information
    normality_dict = {x:(shapiro_ps(numeric_features[x]),
                         agostino_ps(numeric_features[x]),
                         numeric_features[x].skew(skipna = True),
                         kurtosis(numeric_features[x],nan_policy='omit')
                         ) for x in f_numeric_name}
    normality_pd = pd.DataFrame(normality_dict, index = ['shapiro_pvalue','DAgostino_pvalue','Skew','Kurtosis'])
    
    # 4. Simple describe
    other_info_num = df.describe()
    
    
    numeric_describe = pd.concat([other_info_num, resumen_pd_num, outlier_pd, normality_pd], sort = True)
    
  return object_describe, numeric_describe



def shapiro_ps(df):
  """  
  Perform the Shapiro-Wilk test for normality.

  The Shapiro-Wilk test tests the null hypothesis that the
  data was drawn from a normal distribution.

  Parameters
  ----------
  x : array_like
      Array of sample data.

  Returns
  -------
  W : float
      The test statistic.
  p-value : float
      The p-value for the hypothesis test.

  See Also
  --------
  anderson : The Anderson-Darling test for normality
  kstest : The Kolmogorov-Smirnov test for goodness of fit.

  Notes
  -----
  The algorithm used is described in [4]_ but censoring parameters as
  described are not implemented. For N > 5000 the W test statistic is accurate
  but the p-value may not be.

  The chance of rejecting the null hypothesis when it is true is close to 5%
  regardless of sample size.

  References
  ----------
  .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
  .. [2] Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for
         normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
  .. [3] Razali, N. M. & Wah, Y. B. (2011) Power comparisons of Shapiro-Wilk,
         Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests, Journal of
         Statistical Modeling and Analytics, Vol. 2, pp. 21-33.
  .. [4] ALGORITHM AS R94 APPL. STATIST. (1995) VOL. 44, NO. 4.

  Examples
  --------
  >>> from preml.utils import shapiro_ps
  >>> np.random.seed(12345678)
  >>> x = stats.norm.rvs(loc=5, scale=3, size=100)
  >>> shapiro_ps(x)
  (0.9772805571556091, 0.08144091814756393)  
  """
  stat, p = shapiro(df.dropna())
  return np.round(p,4)

def agostino_ps(df):
  """
  Test whether a sample differs from a normal distribution.
 
  This function tests the null hypothesis that a sample comes
  from a normal distribution.  It is based on D'Agostino and
  Pearson's [1]_, [2]_ test that combines skew and kurtosis to
  produce an omnibus test of normality.
   
   
  Parameters
  ----------
  a : array_like
      The array containing the sample to be tested.
  axis : int or None, optional
      Axis along which to compute test. Default is 0. If None,
      compute over the whole array `a`.
  nan_policy : {'propagate', 'raise', 'omit'}, optional
      Defines how to handle when input contains nan. 'propagate' returns nan,
      'raise' throws an error, 'omit' performs the calculations ignoring nan
      values. Default is 'propagate'.
   
  Returns
  -------
  statistic : float or array
      ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and
      ``k`` is the z-score returned by `kurtosistest`.
  pvalue : float or array
     A 2-sided chi squared probability for the hypothesis test.
   
  References
  ----------
  .. [1] D'Agostino, R. B. (1971), "An omnibus test of normality for
         moderate and large sample size", Biometrika, 58, 341-348
   
  .. [2] D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
         normality", Biometrika, 60, 613-622
   
  Examples
  --------
  >>> from scipy import stats
  >>> pts = 1000
  >>> np.random.seed(28041990)
  >>> a = np.random.normal(0, 1, size=pts)
  >>> b = np.random.normal(2, 1, size=pts)
  >>> x = np.concatenate((a, b))
  >>> k2, p = stats.normaltest(x)
  >>> alpha = 1e-3
  >>> print("p = {:g}".format(p))
  p = 3.27207e-11
  >>> if p < alpha:  # null hypothesis: x comes from a normal distribution
  ...     print("The null hypothesis can be rejected")
  ... else:
  ...     print("The null hypothesis cannot be rejected")
  The null hypothesis can be rejected
  """
  stat, p = normaltest(df.dropna())
  return np.round(p,4)

def tukey_outlier(ys, threshold = 1.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
  
    quartile_1, quartile_3 = np.percentile(ys, [5, 95])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * threshold)
    upper_bound = quartile_3 + (iqr * threshold)
    return (ys > upper_bound) | (ys < lower_bound)

def zscore_outlier(ys, threshold = 8):    
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """  
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.abs(z_scores) > threshold
