#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:17:16 2019

@author: xuji
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import sklearn
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import datetime

DATE_TODAY = pd.datetime(2019, 2, 18)
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

def additional_features_new(df, newPrefix='n_'):
    df[newPrefix + 'first_buy'] = ((df[newPrefix + 'purchase_date_min']) -(df['first_active_month'])).dt.days
    df[newPrefix + 'last_buy'] = ((df[newPrefix + 'purchase_date_max']) - (df['first_active_month'])).dt.days
    df['h_first_buy'] = ((df['h_purchase_date_min']) -(df['first_active_month'])).dt.days
    df['h_last_buy'] = ((df['h_purchase_date_max']) - (df['first_active_month'])).dt.days
#    date_features = [newPrefix + 'purchase_date_max', newPrefix + 'purchase_date_min']
#    for f in date_features:
#        df[f] = df[f].astype(np.int64) * 1e-9
    return df
    

def additional_features(df, histPrefix='hist_', newPrefix='new_', prefix='au_'):
#    df[newPrefix + 'first_buy'] = (df[newPrefix + 'purchase_date_min'] - df['first_active_month']).dt.days
#    df[newPrefix + 'last_buy'] = (df[newPrefix + 'purchase_date_max'] -df['first_active_month']).dt.days
#    date_features = [histPrefix + 'purchase_date_max', histPrefix + 'purchase_date_min']
#    for f in date_features:
#        df[f] = df[f].astype(np.int64) * 1e-9
    
    df[prefix + 'purchase_amount_total'] = df[newPrefix + 'purchase_amount_sum'] + df[histPrefix + 'purchase_amount_sum']
    df[prefix + 'purchase_amount_mean'] = df[newPrefix + 'purchase_amount_mean'] + df[histPrefix + 'purchase_amount_mean']
    df[prefix + 'purchase_amount_max'] = df[newPrefix + 'purchase_amount_max'] + df[histPrefix + 'purchase_amount_max']
    df[prefix + 'purchase_amount_min'] = df[newPrefix + 'purchase_amount_min'] + df[histPrefix + 'purchase_amount_min']
    df[prefix + 'purchase_amount_ratio'] = df[newPrefix + 'purchase_amount_sum'] / df[histPrefix + 'purchase_amount_sum']
    
    df[prefix + 'installments_total'] = df[newPrefix + 'installments_sum'] + df[histPrefix + 'installments_sum']
    df[prefix + 'installments_mean'] = df[newPrefix + 'installments_mean'] + df[histPrefix + 'installments_mean']
    df[prefix + 'installments_max'] = df[newPrefix + 'installments_max'] + df[histPrefix + 'installments_max']
    df[prefix + 'installments_ratio'] = df[newPrefix + 'installments_sum'] / df[histPrefix + 'installments_sum']
    
    df[prefix + 'price_total'] = df[prefix + 'purchase_amount_total'] / df[prefix + 'installments_total']
    df[prefix + 'price_mean'] = df[prefix + 'purchase_amount_mean'] / df[prefix + 'installments_mean']
    df[prefix + 'price_max'] = df[prefix + 'purchase_amount_max'] / df[prefix + 'installments_max']
    
    df[prefix + 'month_diff_mean'] = df[newPrefix + 'month_diff_mean'] + df[histPrefix + 'month_diff_mean']
    df[prefix + 'month_diff_ratio'] = df[newPrefix + 'month_diff_mean'] / df[histPrefix + 'month_diff_mean']
    
    df[prefix + 'month_lag_mean'] = df[newPrefix + 'month_lag_mean'] + df[histPrefix + 'month_lag_mean']
    df[prefix + 'month_lag_max'] = df[newPrefix + 'month_lag_max'] + df[histPrefix + 'month_lag_max']
    df[prefix + 'month_lag_min'] = df[newPrefix + 'month_lag_min'] + df[histPrefix + 'month_lag_min']
    df[prefix + 'category_1_mean'] = df[newPrefix + 'category_1_mean'] + df[histPrefix + 'category_1_mean']
    
#    df[prefix + 'duration_mean'] = df[newPrefix + 'duration_mean'] + df[histPrefix + 'duration_mean']
#    df[prefix + 'duration_min'] = df[newPrefix + 'duration_min'] + df[histPrefix + 'duration_min']
#    df[prefix + 'duration_max'] = df[newPrefix + 'duration_max'] + df[histPrefix + 'duration_max']
#    
#    df[prefix + 'amount_month_ratio_mean'] = df[newPrefix + 'amount_month_ratio_mean'] + df[histPrefix + 'amount_month_ratio_mean']
#    df[prefix + 'amount_month_ratio_min'] = df[newPrefix + 'amount_month_ratio_min'] + df[histPrefix + 'amount_month_ratio_min']
#    df[prefix + 'amount_month_ratio_max'] = df[newPrefix + 'amount_month_ratio_max'] + df[histPrefix + 'amount_month_ratio_max']
    
#    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']
#    df['CLV_sq'] = df['new_CLV'] * df['hist_CLV']
    
    return df


def outlier(train, test):
    train['outliers'] = 0
    train.loc[train['target'] < -30., 'outliers'] = 1
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    for f in feature_cols:
        order_label = train.groupby([f])['outliers'].mean()
        train[f + '_bp'] = train[f].map(order_label)
        test[f + '_bp'] = test[f].map(order_label)
    train['feature_sum'] = train['feature_1_bp'] + train['feature_2_bp'] + train['feature_3_bp']
    train['feature_mean'] = train['feature_sum'] / 3
    train['feature_max'] = train[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    train['feature_min'] = train[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    train['feature_var'] = train[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    
    test['feature_sum'] = test['feature_1_bp'] + test['feature_2_bp'] + test['feature_3_bp']
    test['feature_mean'] = test['feature_sum'] / 3
    test['feature_max'] = test[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    test['feature_min'] = test[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    test['feature_var'] = test[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    return train, test

def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum((pd.to_datetime(date_holiday)-df[date_ref]).dt.days, period), 0)

def aggragate_historical_holiday(history):
    df = history['purchase_date']
    df = pd.to_datetime(df)
    history.loc[:, 'purchase_date'] = df
    history.loc[:, 'month_diff'] = ((datetime.date(2018, 4, 30) - df.dt.date).dt.days) // 30
    history.loc[:, 'month_diff'] += history.loc[:, 'month_lag']
    history.loc[:, 'duration'] = history.loc[:, 'purchase_amount'] * history.loc[:, 'month_diff']
    history.loc[:, 'amount_month_ratio'] = history.loc[:, 'purchase_amount'] / history.loc[:, 'month_diff']
#    holidays = [('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
#        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
#        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
#        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
#        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
#        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
#        ('Mothers_Day_2018', '2018-05-13'),
#        ]
#    for d_name, d_day in holidays:
#        dist_holiday(history, d_name, d_day, 'purchase_date')
##    
#    agg_func = {
#            'Christmas_Day_2017': ['mean'],
#            'Mothers_Day_2017': ['mean'],
#            'fathers_day_2017': ['mean'],
#            'Valentine_Day_2017': ['mean'],
#            'Children_day_2017': ['mean'],
#            'Black_Friday_2017': ['mean'],
#            'Mothers_Day_2018': ['mean'],  
#            }
    agg_func = {
#            'price': ['mean', 'max', 'min', 'var']
#            'purchase_date': ['min', 'max'],
#            'card_id': ['size']
            'duration': ['mean', 'min', 'max'],
            'amount_month_ratio': ['mean', 'min', 'max']
            }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() \
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    return agg_history
       

def aggragate_new_holiday(new):
    df = new['purchase_date']
    df = pd.to_datetime(df)
    new.loc[:, 'purchase_date'] = df
    new.loc[:, 'month_diff'] = ((datetime.date(2018, 4, 30) - df.dt.date).dt.days) // 30
    new.loc[:, 'month_diff'] += new.loc[:, 'month_lag']
    new.loc[:, 'duration'] = new.loc[:, 'purchase_amount'] * new.loc[:, 'month_diff']
    new.loc[:, 'amount_month_ratio'] = new.loc[:, 'purchase_amount'] / new.loc[:, 'month_diff']
#    holidays = [('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
#        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
#        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
#        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
#        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
#        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
#        ('Mothers_Day_2018', '2018-05-13'),
#        ]
#    for d_name, d_day in holidays:
#        dist_holiday(new, d_name, d_day, 'purchase_date')
#    
#    agg_func = {
#            'Christmas_Day_2017': ['mean'],
#            'Mothers_Day_2017': ['mean'],
#            'fathers_day_2017': ['mean'],
#            'Valentine_Day_2017': ['mean'],
#            'Children_day_2017': ['mean'],
#            'Black_Friday_2017': ['mean'],
#            'Mothers_Day_2018': ['mean'],  
#            }
    agg_func = {
#            'price': ['mean', 'max', 'min', 'var']
#            'purchase_date': ['min', 'max'],
#            'card_id': ['size']
            'duration': ['mean', 'min', 'max'],
            'amount_month_ratio': ['mean', 'min', 'max']
    
            }
    agg_new = new.groupby(['card_id']).agg(agg_func)
    agg_new.columns = ['new_' + '_'.join(col).strip() \
                           for col in agg_new.columns.values]
    agg_new.reset_index(inplace=True)
    return agg_new

####################### Start ##########################################
extracted_features_train = pd.read_csv('feature/train2_RFM2_model.csv')
extracted_features_train = reduce_mem_usage(extracted_features_train)
extracted_features_test = pd.read_csv('feature/test2_RFM2_model.csv')
extracted_features_test = reduce_mem_usage(extracted_features_test)


original_train = pd.read_csv('data/input/train.csv', usecols=['card_id', 'target'])
original_test = pd.read_csv('data/input/test.csv', usecols=['card_id'])
original_train = reduce_mem_usage(original_train)
original_test = reduce_mem_usage(original_test)
#original_train, original_test = outlier(original_train, original_test)

extracted_features_train['card_id'] = original_train['card_id']
extracted_features_test['card_id'] = original_test['card_id']
target = original_train['target']
del original_train['target']
#del extracted_features_train['target']
#train = pd.merge(extracted_features_train, original_train, how='left', on='card_id')
#test = pd.merge(extracted_features_test, original_test, how='left', on='card_id')
train = extracted_features_train
test = extracted_features_test
del original_train, original_test
gc.collect()



## read historical & new_transactions
#historical = pd.read_csv('data/input/historical_transactions.csv', usecols=['card_id', 'purchase_date', 'month_lag', 'purchase_amount'])
#new_tran = pd.read_csv('data/input/new_merchant_transactions.csv', usecols=['card_id', 'purchase_date', 'month_lag', 'purchase_amount'])
#historical = reduce_mem_usage(historical)
#new_tran = reduce_mem_usage(new_tran)
#historical['price'] = historical['purchase_amount'] / historical['installments']
#new_tran['price'] = new_tran['purchase_amount'] / new_tran['installments']

#historical['price'] = historical['purchase_amount'] / historical['installments']
#new_tran['price'] = new_tran['purchase_amount'] / new_tran['installments']

#historical = aggragate_historical_holiday(historical)
#new_tran = aggragate_new_holiday(new_tran)
#historical = reduce_mem_usage(historical)
#new_tran = reduce_mem_usage(new_tran)
#
#
#train = pd.merge(extracted_features_train, historical, how='left', on='card_id')
#train = pd.merge(train, new_tran, how='left', on='card_id')
#test = pd.merge(extracted_features_test, historical, how='left', on='card_id')
#test = pd.merge(test, new_tran, how='left', on='card_id')

#train['hist_purchase_date_diff'] = (train['hist_purchase_date_max'] - train['hist_purchase_date_min']).dt.days
#train['histauth_purchse_date_average'] = (train['hist_purchase_date_diff']) / train['histauth_card_id_size']
#train['histunauth_purchse_date_average'] = (train['hist_purchase_date_diff']) / train['histunauth_card_id_size']
#train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days
#train['new_purchse_date_average'] = (train['new_purchase_date_diff']) / train['new_card_id_size']
#
#test['hist_purchase_date_diff'] = (test['hist_purchase_date_max'] - test['hist_purchase_date_min']).dt.days
#test['histauth_purchse_date_average'] = (test['hist_purchase_date_diff']) / test['histauth_card_id_size']
#test['histunauth_purchse_date_average'] = (test['hist_purchase_date_diff']) / test['histunauth_card_id_size']
#test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days
#test['new_purchse_date_average'] = (test['new_purchase_date_diff']) / test['new_card_id_size']


#train = additional_features_new(train)
#test = additional_features_new(test)
#train = additional_features(train, histPrefix='histauth_', prefix='au_')
#train = additional_features(train, histPrefix='histunauth_', prefix='unau_')
#test = additional_features(test, histPrefix='histauth_', prefix='au_')
#test = additional_features(test, histPrefix='histunauth_', prefix='unau_')
#extracted_features_train = additional_features_new(extracted_features_train)
#extracted_features_test = additional_features_new(extracted_features_test)
#train = additional_features(extracted_features_train, histPrefix='histunauth_', prefix='unau_')
#test = additional_features(extracted_features_test, histPrefix='histunauth_', prefix='unau_')
#del extracted_features_train
#del extracted_features_test
#gc.collect()

del extracted_features_train
del extracted_features_test
gc.collect()


use_cols = [col for col in train.columns if col not in 
            ['hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min', 'target', 'card_id', 'outliers', 'first_active_month','histauth_purchase_date_max', 'histauth_purchase_date_min',
             'histunauth_purchase_date_max', 'histunauth_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min',
             'new_card_id_size', 'histauth_card_id_size', 'histunauth_card_id_size','new_authorized_flag_mean']]
train = train[use_cols]
test = test[use_cols]

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
lgb_params2 = {'num_leaves': 50, \
         'min_data_in_leaf': 30, \
         'objective':'regression', \
         'max_depth': -1, \
         'learning_rate': 0.005, \
         "boosting": "dart", \
         "feature_fraction": 0.9, \
         "bagging_freq": 1, \
         "bagging_fraction": 0.9, \
         "bagging_seed": 11, \
         "metric": 'rmse', \
         "lambda_l1": 0.1, \
         "verbosity": -1}
xgb_params2 = {
        'objective': 'reg:linear',
        'booster': "gbtree",
        'eval_metric': "rmse",
        'eta': 0.02,
        'max_depth': 7,
        'min_child_weight': 100,
        'gamma': 0,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.85,
        'alpha': 0,
        'silent': True,
        'lambda': 0.1
        }
xgb_params = {
#            'gpu_id': 0, 
            #'n_gpus': 2, 
            'objective': 'reg:linear', 
            'eval_metric': 'rmse', 
            'silent': True, 
            'booster': 'gbtree', 
            'n_jobs': 8, 
            'n_estimators': 2500, 
#            'tree_method': 'gpu_hist', 
            'grow_policy': 'lossguide', 
            'max_depth': 12, 
            'seed': 538, 
            'colsample_bylevel': 0.9, 
            'colsample_bytree': 0.8, 
            'gamma': 0.0001, 
            'learning_rate': 0.006150886706231842, 
            'max_bin': 128, 
            'max_leaves': 47, 
            'min_child_weight': 40, 
            'reg_alpha': 10.0, 
            'reg_lambda': 10.0, 
            'subsample': 0.9}

#lgb_params2 = {
#        'objective': 'regression_l2',
#        'boosting_type': 'dart', 
#        'n_jobs': 8, 'max_depth': 7, 
#        'n_estimators': 20000, 
#        'subsample_freq': 2, 
#        'subsample_for_bin': 200000, 
#        'min_data_per_group': 100, 
#        'max_cat_to_onehot': 4, 
#        'cat_l2': 10.0, 
#        'cat_smooth': 10.0, 
#        'max_cat_threshold': 32, 
#        'metric_freq': 10, 
#        'verbosity': -1, 
#        'metric': 'rmse', 
#        'colsample_bytree': 0.5, 
#        'learning_rate': 0.0061033234451294376, 
#        'min_child_samples': 80, 
#        'min_child_weight': 100.0, 
#        'min_split_gain': 1e-06, 
#        'num_leaves': 47, 
#        'reg_alpha': 10.0, 
#        'reg_lambda': 10.0, 
#        'subsample': 0.9
#        }

FOLDs = KFold(n=train.shape[0], n_folds=10, shuffle=True, random_state=15)

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


#-------------------------------------------------------------------------------
FOLDs = KFold(n=train.shape[0], n_folds=10, shuffle=True, random_state=15)

oof_lgb2 = np.zeros(len(train))
predictions_lgb2 = np.zeros(len(test))


for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB2 " + str(fold_) + "-" * 50)
    num_round = 10000
    clf = lgb.train(lgb_params2, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
    oof_lgb2[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb2 += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_folds
    
print('LRB2 mrs: ', np.sqrt(mean_squared_error(oof_lgb2, target)))

#-------------------------------------------------------------------------------
#print('lgb', np.sqrt(mean_squared_error(oof_lgb, target)))
print('lgb2', np.sqrt(mean_squared_error(oof_lgb2, target)))
#total_sum = 0.5 * oof_lgb + 0.5 * oof_xgb
total_sum = oof_lgb2
print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))

#-------------------------------------------------------------------------------
#FOLDs = KFold(n=train.shape[0], n_folds=10, shuffle=True, random_state=15)
#
#oof_lgb3 = np.zeros(len(train))
#predictions_lgb3 = np.zeros(len(test))
#
#
#for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
#    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
#    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])
#
#    print("LGB3 " + str(fold_) + "-" * 50)
#    num_round = 10000
#    clf = lgb.train(lgb_params3, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
#    oof_lgb3[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
#    predictions_lgb3 += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_folds
#    
#print('LRB3 mrs: ', np.sqrt(mean_squared_error(oof_lgb3, target)))
#print('lgb3', np.sqrt(mean_squared_error(oof_lgb3, target)))
#total_sum = oof_lgb3
#print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))
#-------------------------------------------------------------------------------
FOLDs = KFold(n=train.shape[0], n_folds=10, shuffle=True, random_state=15)

oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
    trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 4000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=200)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_folds

print('XRB mrs: ', np.sqrt(mean_squared_error(oof_xgb, target)))

#-------------------------------------------------------------------------------
#FOLDs = KFold(n=train.shape[0], n_folds=10, shuffle=True, random_state=15)
#
#oof_xgb2 = np.zeros(len(train))
#predictions_xgb2 = np.zeros(len(test))
#for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
#    trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
#    val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
#    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
#    print("xgb2 " + str(fold_) + "-" * 50)
#    num_round = 4000
#    xgb_model = xgb.train(xgb_params2, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=200)
#    oof_xgb2[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)
#
#    predictions_xgb2 += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_folds
#
#print('XRB2 mrs: ', np.sqrt(mean_squared_error(oof_xgb2, target)))
##############################importance plot#################################
#cols = (feature_importance_df_lgb[["feature", "importance"]] \
#        .groupby("feature") \
#        .mean() \
#        .sort_values(by="importance", ascending=False)[:1000].index)
#
#best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]
#
#plt.figure(figsize=(30,30))
#sns.barplot(x="importance", \
#            y="feature", \
#            data=best_features.sort_values(by="importance", \
#                                           ascending=False))
#plt.title('LightGBM Features (avg over folds)')
#plt.tight_layout()
#plt.savefig('./result/lgbm_importances1.png')


######################################Stacking########################################
########stack lgb & xgb###################
train_stack = np.vstack([oof_lgb, oof_lgb2, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_lgb2, predictions_xgb]).transpose()
folds = KFold(n=train_stack.shape[0], n_folds=5, shuffle=True, random_state=15)
oof = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])
for fold_, (trn_idx, val_idx) in enumerate(folds):
    print("fold n°{}".format(fold_))
    clf = Ridge(alpha=1)
    clf.fit(train_stack[trn_idx], target.iloc[trn_idx])
    oof[val_idx] = clf.predict(train_stack[val_idx])
    predictions += clf.predict(test_stack) / folds.n_folds
print('After stacking, valid-rmse: ', np.sqrt(mean_squared_error(target.values, oof)))

sub_df = pd.read_csv('data/input/sample_submission.csv')
sub_df['target'] = predictions
#sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
#sub_df['target'] = predictions_xgb
sub_df.to_csv("./result/submission17.csv", index=False)



