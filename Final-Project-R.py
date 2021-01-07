#!/usr/bin/env python
# coding: utf-8

# ### PROJECT - PREDICT FUTURE SALES
# 
# #### MODEL- Time Series Forecasting using Prophet
# 
# <font color='blue'>  A time series is a sequence of observations taken sequentially in time
# The purpose of time series analysis is generally twofold: to understand or model the stochastic mechanisms that gives rise to an observed series and to predict or forecast the future values of a series based on the history of that series</font>
# 
# 
# #### Advantages of using this model
# #### A) Descriptive analysis - determines the "trends" and pattern" of future of future using graphs.
# #### B) Forecasting :- used in financial, business forecasting based on trends and patterns.
# #### C) Explanative Analysis : the study of cross correlation / relationship between two time series and their dependency on one another.
# 

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[26]:


sales=pd.read_csv("/Users/charan/Downloads/project_sales/sales_train.csv")
item_cat=pd.read_csv("/Users/charan/Downloads/project_sales/item_categories.csv")
item=pd.read_csv("/Users/charan/Downloads/project_sales/items.csv")
shops=pd.read_csv("/Users/charan/Downloads/project_sales/shops.csv")
test=pd.read_csv("/Users/charan/Downloads/project_sales/test.csv")
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))


# In[27]:


monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales.head(20)


# # Time Series Forecasting using Prophet

# In[31]:


ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
fig.show()


# In[32]:


from rpy2.robjects import r
def decompose(series, frequency, s_window, **kwargs):
    df = pd.DataFrame()
    df['date'] = series.index
    s = [x for x in series.values]
    length = len(series)
    s = r.ts(s, frequency=frequency)
    decomposed = [x for x in r.stl(s, s_window, **kwargs).rx2('time.series')]
    df['observed'] = series.values
    df['trend'] = decomposed[length:2*length]
    df['seasonal'] = decomposed[0:length]
    df['residual'] = decomposed[2*length:3*length]
    return df


# In[34]:


from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic


# In[35]:


def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)


# In[36]:


from pandas import Series as Series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def inverse_difference(last_ob, value):
    return value + last_ob


# In[37]:


ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
new_ts=difference(ts,12)       # assuming the seasonality is 12 months long

test_stationarity(new_ts)


# In[38]:


def tsplot(y, lags=None, figsize=(10, 8), style='bmh',title=''):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


# In[40]:


best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))


# In[41]:


ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.head()


# In[ ]:




