#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:19:51 2018
    convert notebook to script
@author: xuji
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.cross_validation import KFold
import warnings
import time
import sys
import gc
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
warnings.simplefilter(action='ignore', category=FutureWarning)
gc.enable()

debug = False

train_dtype = {'card_id': 'category', 'feature_1': 'uint8', 'feature_2': 'uint8', 'feature_3': 'uint8'}
trans_dtype = {'card_id': 'category', 'city_id': 'int16', 'installments': 'int16', 'merchant_category_id': 'int16',
              'month_lag': 'int8', 'purchase_amount': 'float32', 'state_id': 'int8', 'subsector_id': 'int8'}

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
#                else:
#                    print('WRONG!!!')
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)
#                else:
#                    print('WRONG!!')
    return df


def add_subsector_id_feature_new_transcation(df):
    a = df['subsector_id'].dropna(how = 'all')
    a = set(a)
    a = list(a)
    agg_func1 = {'purchase_amount' : ['mean']}
    whole = pd.DataFrame()
    
    for i in range(len(a)):
        b = df[df['subsector_id'] == a[i]][['card_id', 'purchase_amount']]
        c = b.groupby(['card_id']).agg(agg_func1)
        c.columns = ['my' + '_'.join(col).strip() + '_' + str(i) for col in c.columns.values]       
        whole = whole.append(c)
    
    
    whole = whole.fillna(0)
    whole['card_id'] = whole.index
    agg_func2 = {'mypurchase_amount_mean_'+str(i) : ['mean'] for i in range(len(a))}
    whole = whole.groupby(['card_id']).agg(agg_func2)
    whole.columns = ['my' + '_'.join(col).strip() + '_new' for col in whole.columns.values]
    count_row = (whole != 0).sum(axis = 1)
    whole = whole.sum(axis = 1)
    whole2 = whole / count_row
    fin = pd.DataFrame({'card_id': whole.index, 'new_sum_cal': whole.values, 'new_mean_cal': whole2.values})
    return fin
    
def add_subsector_id_feature_historical_transcation(df):
    a = df['subsector_id'].dropna(how = 'all')
    a = set(a)
    a = list(a)
    agg_func1 = {'purchase_amount' : ['mean']}
    whole = pd.DataFrame()
    
    for i in range(len(a)):
        b = df[df['subsector_id'] == a[i]][['card_id', 'purchase_amount']]
        c = b.groupby(['card_id']).agg(agg_func1)
        c.columns = ['my' + '_'.join(col).strip() + '_' + str(i) for col in c.columns.values]       
        whole = whole.append(c)
    
    whole = whole.fillna(0)
    whole['card_id'] = whole.index
    agg_func2 = {'mypurchase_amount_mean_'+str(i) : ['mean'] for i in range(len(a))}
    whole = whole.groupby(['card_id']).agg(agg_func2)
    whole.columns = ['my' + '_'.join(col).strip() + '_new' for col in whole.columns.values]
    count_row = (whole != 0).sum(axis = 1)
    whole = whole.sum(axis = 1)
    whole2 = whole / count_row
    fin = pd.DataFrame({'card_id': whole.index, 'hist_sum_cal': whole.values, 'hist_mean_cal': whole2.values})
   # fin = pd.DataFrame({'card_id': whole.index, 'hist_sum_cal': whole.values})
    return fin    
    
    
    
def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum((pd.to_datetime(date_holiday)-df[date_ref]).dt.days, period), 0)


def aggregate_new_transactions(new_trans):   
    df = new_trans['purchase_date']
    df = pd.to_datetime(df)
    new_trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(new_trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    new_trans.loc[:, 'purchase_date2'] = df
    new_trans.loc[:, 'day_lag'] = new_trans['month_lag'] * (-1) * 30
    new_trans.loc[:, 'trans_dayofweek'] = df.dt.dayofweek
    new_trans.loc[:, 'purchase_day_dayofmonth'] = df.dt.day
    
    new_trans.loc[:, 'purchase_day_month'] = df.dt.month
    new_trans.loc[:, 'purchase_day_month'] += new_trans.loc[:, 'month_lag']
    
    new_trans.loc[:, 'purchase_day_lag'] = (datetime.date(2018, 2, 1) - df.dt.date).dt.days
    new_trans.loc[:, 'month_diff'] =  new_trans.loc[:, 'purchase_day_lag'] // 30
    new_trans.loc[:, 'month_diff'] += new_trans.loc[:, 'month_lag']
    #new_trans.loc[:, 'weekend'] = (df.dt.weekday >= 5).astype(int)
    agg_func = {
        'authorized_flag': ['sum','mean'], \
        'category_1' : ['mean'], \
        'category_2_1.0' : ['mean'], \
        'category_2_2.0' : ['mean'], \
        'category_2_3.0' : ['mean'], \
        'category_2_4.0' : ['mean'], \
        'category_2_5.0' : ['mean'], \
        'category_3_A' : ['mean'], \
        'category_3_B' : ['mean'], \
        'category_3_C' : ['mean'], \
        'merchant_id': ['nunique'], \
        'city_id': ['nunique'], \
        'state_id' : ['nunique'],
        'purchase_amount': ['sum', 'max', 'min', 'std', 'mean', np.ptp], \
        'installments': ['sum', 'max', 'min','std', 'mean'], \
        'purchase_date': [np.ptp], \
        'month_lag': ['min', 'max', 'std', 'mean'],
        'day_lag' : ['min', 'max', 'std', 'mean'],
        'purchase_day_lag' : ['min', 'max', 'mean', 'std', 'nunique'],
        #'merchant_category_id': ['nunique'],
        'trans_dayofweek' : ['min', 'max', 'mean', 'std'],
        'purchase_day_dayofmonth' : ['min', 'max', 'mean', 'std'],
        #'weekend' : ['sum', 'mean']
        'month_diff' : ['mean', 'max', 'min', 'std'],
        'purchase_date2' : ['max', 'min'],
        #'purchase_day_month' : ['mean', 'max', 'min']
        
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() \
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id') \
          .size() \
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    print('new_tran agg done!')
    return agg_new_trans

    
def aggregate_historical_transactions(history):
    df = history['purchase_date']
    df = pd.to_datetime(df)
    history.loc[:, 'purchase_date_cp'] = df
    
    history.loc[:, 'month'] = df.dt.month
    history.loc[:, 'day'] = df.dt.day
    history.loc[:, 'weekofyear'] = df.dt.weekofyear
    history.loc[:, 'weekday'] = df.dt.weekday
    history.loc[:, 'weekend'] = (df.dt.weekday >= 5).astype(int)
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    history.loc[:, 'purchase_date2'] = df
    history.loc[:, 'day_lag'] = history['month_lag'] * (-1) * 30
    history.loc[:, 'purchase_day_lag'] = (datetime.date(2018, 2, 1) - df.dt.date).dt.days
    history.loc[:, 'purchase_day_dayofmonth'] = df.dt.day
    history.loc[:, 'trans_dayofweek'] = df.dt.dayofweek
    history.loc[:, 'month_diff'] = history.loc[:, 'purchase_day_lag'] // 30
    history.loc[:, 'month_diff'] += history.loc[:, 'month_lag']
    
    history.loc[:, 'duration'] = history.loc[:, 'purchase_amount'] * history.loc[:, 'month_diff']
    history.loc[:, 'amount_month_ratio'] = history.loc[:, 'purchase_amount'] / history.loc[:, 'month_diff']
    
    history.loc[:, 'purchase_day_month'] = df.dt.month
    history.loc[:, 'purchase_day_month'] += history.loc[:, 'month_lag']
    
    holidays = [('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
        ('Mothers_Day_2018', '2018-05-13'),
        ]
    
    for d_name, d_day in holidays:
        dist_holiday(history, d_name, d_day, 'purchase_date_cp')
    
    agg_func = { \
        'authorized_flag': ['sum', 'mean'], \
        'category_1' : ['mean'], \
        'category_2_1.0' : ['mean'], \
        'category_2_2.0' : ['mean'],
        'category_2_3.0' : ['mean'], \
        'category_2_4.0' : ['mean'], \
        'category_2_5.0' : ['mean'], \
        'category_3_A' : ['mean'], \
        'category_3_B' : ['mean'], \
        'category_3_C' : ['mean'], \
        'merchant_id': ['nunique'], \
        'city_id': ['nunique'], \
        'state_id' : ['nunique'],
        'purchase_amount': ['sum', 'max', 'min', 'std', 'mean', np.ptp], \
        'installments': ['sum', 'max', 'min', 'std', 'mean'], \
        'purchase_date': [np.ptp], \
        'month_lag': ['min', 'max', 'std', 'mean'],
        'day_lag' : ['min', 'max', 'mean', 'std'],
        'purchase_day_lag' : ['min', 'max', 'mean', 'std', 'nunique'],
        #'merchant_category_id' : ['nunique'],
        'trans_dayofweek' : ['min', 'max', 'mean', 'std'],
        'purchase_day_dayofmonth' : ['min', 'max', 'mean', 'std'],
        'month_diff' : ['mean', 'max', 'min', 'std'],
        'purchase_date2' : ['max', 'min'],
        #'purchase_day_month' : ['mean', 'max', 'min']
        # new added 2019.2.15
        'month' : ['mean', 'min', 'max'],
        'day' : ['mean', 'min'],
#        'weekofyear' : ['mean'],
        'weekday' : ['mean'],
        'weekend' : ['mean'],
        'price': ['mean', 'max', 'min', 'var', 'sum'],
        'Christmas_Day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Black_Friday_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() \
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id') \
          .size() \
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


def aggregate_per_month(history):
    df = history['purchase_date']
    df = pd.to_datetime(df)
    history.loc[:, 'purchase_day_lag'] = (datetime.date(2018, 2, 1) - df.dt.date).dt.days
    grouped = history.groupby(['card_id', 'month_lag'])
    
    agg_func = {
            #'purchase_amount' : ['count', 'sum', 'mean', 'min', 'max', 'std'],
            #'installments' : ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'purchase_day_lag' : ['max', 'min', 'mean', 'std']
            }
    
    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)
    
    final_group = intermediate_group.groupby(['card_id']).agg(['mean', 'min', 'max'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    return final_group




def read_data_debug(input_file):
    df = pd.read_csv(input_file, dtype=train_dtype, nrows = 1000)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df

def read_data(input_file):
    print('Read train/test file...')
    df = pd.read_csv(input_file, dtype=train_dtype)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
    
#-----------------------Start-----------------------------------------------------   

if debug:
    new_transactions = pd.read_csv('./data/input/new_merchant_transactions.csv', dtype=trans_dtype, nrows=1000)
else:
    new_transactions = pd.read_csv('./data/input/new_merchant_transactions.csv', dtype=trans_dtype)


new_transactions = reduce_mem_usage(new_transactions)
new_transactions['authorized_flag'] = \
    new_transactions['authorized_flag'].map({'Y':1, 'N':0})
   
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])
new_transactions['category_1'] = new_transactions['category_1'].map({'Y' : 1, 'N' : 0})

#my_new_trans = add_subsector_id_feature_new_transcation(new_transactions)

new_trans = aggregate_new_transactions(new_transactions)
#new_trans = pd.merge(my_new_trans, new_trans, on='card_id', how='left')
new_trans['new_purchase_day_lag_elapse'] = new_trans['new_purchase_day_lag_max'] - new_trans['new_purchase_day_lag_min'] +1
new_trans['new_purchase_day_lag_rate'] = new_trans['new_purchase_day_lag_nunique'] / new_trans['new_purchase_day_lag_elapse']
new_trans['new_purchase_amount_mean_time_rate'] = new_trans['new_purchase_amount_mean'] / new_trans['new_purchase_date_ptp']

del new_transactions
gc.collect()
new_trans = reduce_mem_usage(new_trans)

if debug:
    historical_transactions = pd.read_csv('./data/input/historical_transactions.csv', dtype=trans_dtype, nrows=1000)
else:
    historical_transactions = pd.read_csv('./data/input/historical_transactions.csv', dtype=trans_dtype)

historical_transactions = reduce_mem_usage(historical_transactions)
##new added 2019.2.15
na_dict = {'category_2': 1.,
           'category_3': 'A'}
           #'merchant_id': 'M_ID_00a6ca8a8a'}
historical_transactions.fillna(na_dict, inplace=True)
historical_transactions['installments'].replace({-1: np.nan, 999: np.nan}, inplace=True)
#historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))





historical_transactions['authorized_flag'] = \
    historical_transactions['authorized_flag'].map({'Y':1, 'N':0})

historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y' : 1, 'N' : 0})

## new added 2019.2.15
historical_transactions['price'] = historical_transactions['purchase_amount'] / historical_transactions['installments']

historical_transactions = reduce_mem_usage(historical_transactions)

history = aggregate_historical_transactions(historical_transactions)
del historical_transactions
gc.collect()

history['hist_purchase_day_lag_elapse'] = history['hist_purchase_day_lag_max'] - history['hist_purchase_day_lag_min'] +1
history['hist_purchase_day_lag_rate'] = history['hist_purchase_day_lag_nunique'] / history['hist_purchase_day_lag_elapse']



#del hist_per_month


print('loading hist done!!')
if debug:
    train = read_data_debug('./data/input/train.csv')
    test = read_data_debug('./data/input/test.csv')
    
else:
    #train = read_data('./data/input/train_clean.csv')
    train = read_data('./data/input/train.csv')
    test = read_data('./data/input/test.csv')

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# find outliers
train['outliers'] = 0
train.loc[train['target'] < -30., 'outliers'] = 1

target = train['target']
del train['target']

train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
train = pd.merge(train, new_trans, on='card_id', how='left')
test = pd.merge(test, new_trans, on='card_id', how='left')

use_cols = [col for col in train.columns if col not in ['card_id', 'first_active_month', 'outliers']]
####new add
train_firstActiveMonth = train['first_active_month']
test_firstActiveMonth = test['first_active_month']
###

train = train[use_cols]
test = test[use_cols]

###new add

###

features = list(train[use_cols].columns)
categorical_feats = [col for col in features if 'feature_' in col]
categorical_feats = categorical_feats[:2]

for col in categorical_feats:
    #print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))
    
df_all = pd.concat([train, test])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = train.shape[0]

train = df_all[:len_train]
test = df_all[len_train:]

train.loc[:, 'purchase_amount_time_average'] = train.loc[:,'hist_purchase_amount_max'] / train.loc[:,'hist_purchase_day_lag_min']
#train['hist_first_buy'] = (train['hist_purchase_date2_min'] - train_firstActiveMonth).dt.days
#train['new_first_buy'] = (train['new_purchase_date2_min'] - train_firstActiveMonth).dt.days


test.loc[:, 'purchase_amount_time_average'] = test.loc[:,'hist_purchase_amount_max'] / test.loc[:,'hist_purchase_day_lag_min']
#test['hist_first_buy'] = (test['hist_purchase_date2_min'] - test_firstActiveMonth).dt.days
#test['new_first_buy'] = (test['new_purchase_date2_min'] - test_firstActiveMonth).dt.days


train = train.drop(['hist_purchase_date2_min', 'hist_purchase_date2_max', 'new_purchase_date2_min', 'new_purchase_date2_max'], axis=1)
test = test.drop(['hist_purchase_date2_min', 'hist_purchase_date2_max', 'new_purchase_date2_min', 'new_purchase_date2_max'], axis=1)

del train_firstActiveMonth
del test_firstActiveMonth
gc.collect()
#train['target'] = target
#train = train[train['target']>-30]
#del train['target']


#################LGB###################################
#lgb_params = {"objective" : "regression", "metric" : "rmse", \
#               "max_depth": 7, "min_child_samples": 20, \
#               "reg_alpha": 1, "reg_lambda": 1, \
#               "num_leaves" : 64, "learning_rate" : 0.001, \ 
#               "subsample" : 0.8, "colsample_bytree" : 0.8, \
#               "verbosity": -1}

lgb_params = {'num_leaves': 50, \
         'min_data_in_leaf': 30, \
         'objective':'regression', \
         'max_depth': -1, \
         'learning_rate': 0.005, \
         "boosting": "gbdt", \
         "feature_fraction": 0.9, \
         "bagging_freq": 1, \
         "bagging_fraction": 0.9, \
         "bagging_seed": 11, \
         "metric": 'rmse', \
         "lambda_l1": 0.1, \
         "verbosity": -1}

FOLDs = KFold(n=train.shape[0], n_folds=5, shuffle=True, random_state=15)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_folds
    
print('LGB mrs: ', np.sqrt(mean_squared_error(oof_lgb, target)))


#################XGB###################################
#xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, \
#           'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}
#
#
#
#FOLDs = KFold(n=train.shape[0], n_folds=5, shuffle=True, random_state=15)
#
#oof_xgb = np.zeros(len(train))
#predictions_xgb = np.zeros(len(test))
#
#
#for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
#    trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
#    val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
#    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
#    print("xgb " + str(fold_) + "-" * 50)
#    num_round = 4000
#    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=200)
#    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)
#
#    predictions_xgb += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_folds
#
#print('XRB mrs: ', np.sqrt(mean_squared_error(oof_xgb, target)))

#-------------------------------------------------------------------------------
print('lgb', np.sqrt(mean_squared_error(oof_lgb, target)))
#print('xgb', np.sqrt(mean_squared_error(oof_xgb, target)))
#total_sum = 0.5 * oof_lgb + 0.5 * oof_xgb
total_sum = oof_lgb
print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))


##############################importance plot#################################
cols = (feature_importance_df_lgb[["feature", "importance"]] \
        .groupby("feature") \
        .mean() \
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance", \
            y="feature", \
            data=best_features.sort_values(by="importance", \
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('./result/lgbm_importances1.png')



sub_df = pd.read_csv('data/input/sample_submission.csv')
#sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df['target'] = predictions_lgb
sub_df.to_csv("./result/submission1.csv", index=False)
