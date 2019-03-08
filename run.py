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
              'month_lag': 'int8', 'purchase_amount': 'float32', 'state_id': 'int16', 'subsector_id': 'int16'}


def aggregate_new_transactions(new_trans):   
    new_trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(new_trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    agg_func = {
        'authorized_flag': ['sum', 'mean'], \
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
        #'city_id': ['nunique'], \
        'purchase_amount': ['sum', 'max', 'min', 'std', 'mean'], \
        'installments': ['sum', 'max', 'min', 'std', 'mean'], \
        'purchase_date': [np.ptp], \
        'month_lag': ['min', 'max'] \
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() \
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id') \
          .size() \
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

    
def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = { \
        'authorized_flag': ['mean','sum'], \
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
        #'city_id': ['nunique'], \
        'purchase_amount': ['sum', 'max', 'min', 'std', 'mean'], \
        'installments': ['sum', 'max', 'min', 'std', 'mean'], \
        'purchase_date': [np.ptp], \
        'month_lag': ['min', 'max'] \
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

def read_data_debug(input_file):
    df = pd.read_csv(input_file, dtype=train_dtype, nrows = 1000)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df

def read_data(input_file):
    df = pd.read_csv(input_file, dtype=train_dtype)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
    
#-----------------------Start-----------------------------------------------------   

if debug:
    new_transactions = pd.read_csv('./data/all/new_merchant_transactions.csv', dtype=trans_dtype, nrows=1000)
else:
    new_transactions = pd.read_csv('./data/all/new_merchant_transactions.csv', dtype=trans_dtype)
    
new_transactions['authorized_flag'] = \
    new_transactions['authorized_flag'].map({'Y':1, 'N':0})
   
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])
new_transactions['category_1'] = new_transactions['category_1'].map({'Y' : 1, 'N' : 0})
new_trans = aggregate_new_transactions(new_transactions)
del new_transactions
gc.collect()

if debug:
    historical_transactions = pd.read_csv('./data/all/historical_transactions.csv', dtype=trans_dtype, nrows=1000)
else:
    historical_transactions = pd.read_csv('./data/all/historical_transactions.csv', dtype=trans_dtype)

   
historical_transactions['authorized_flag'] = \
    historical_transactions['authorized_flag'].map({'Y':1, 'N':0})

historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y' : 1, 'N' : 0})
history = aggregate_historical_transactions(historical_transactions)
del historical_transactions
gc.collect()

if debug:
    train = read_data_debug('./data/all/train.csv')
    test = read_data_debug('./data/all/test.csv')
    
else:
    train = read_data('./data/all/train.csv')
    test = read_data('./data/all/test.csv')
    
target = train['target']
del train['target']

train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
train = pd.merge(train, new_trans, on='card_id', how='left')
test = pd.merge(test, new_trans, on='card_id', how='left')

use_cols = [col for col in train.columns if col not in ['card_id', 'first_active_month']]
train = train[use_cols]
test = test[use_cols]
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
         'learning_rate': 0.003, \
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
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 400)
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
plt.savefig('lgbm_importances6.png')



sub_df = pd.read_csv('data/all/sample_submission.csv')
#sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df['target'] = predictions_lgb
sub_df.to_csv("submission6.csv", index=False)
