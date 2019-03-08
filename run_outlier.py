# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:52:31 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
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

def outlier(df):
    df['outliers'] = 0
    df.loc[df['target'] < -30., 'outliers'] = 1
    # outliet's target is between -33 and -34
    return df

def classify_model(train, test):
    target = train['outliers']
    train_target = train['target']
    train = train.drop(['outliers'], axis=1)
    test_card_id = test['card_id']
    train_card_id = train['card_id']
    use_cols = [col for col in train.columns if col not in 
            ['hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min', 'target', 'card_id', 'outliers', 'first_active_month','histauth_purchase_date_max', 'histauth_purchase_date_min',
             'histunauth_purchase_date_max', 'histunauth_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min',
             'new_card_id_size', 'histauth_card_id_size', 'histunauth_card_id_size','new_authorized_flag_mean']]
    train = train[use_cols]
    test = test[use_cols]
#    lgb_params = {'num_leaves': 50, \
#         'min_data_in_leaf': 30, \
#         'objective':'regression', \
#         'max_depth': -1, \
#         'learning_rate': 0.005, \
#         "boosting": "gbdt", \
#         "feature_fraction": 0.9, \
#         "bagging_freq": 1, \
#         "bagging_fraction": 0.9, \
#         "bagging_seed": 11, \
#         "metric": 'rmse', \
#         "lambda_l1": 0.1, \
#         "verbosity": -1}
    lgb_params = {
            'objective': 'binary',
            'learning_rate': 0.02,
            'num_leaves': 76,
            'feature_fraction': 0.64,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'boosting_type': 'gbdt',
            'metric': 'binary_logloss'
            }
    FOLDs = KFold(n=train.shape[0], n_folds=5, shuffle=True, random_state=15)
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    
    for fold_, (trn_idx, val_idx) in enumerate(FOLDs):
        trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])
        print("LGB " + str(fold_) + "-" * 50)
        num_round = 10000
        clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
        oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
        predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_folds
    
    test['outliers'] = predictions_lgb
    test['card_id'] = test_card_id
    train['outliers'] = target
    train['card_id'] = train_card_id
    train['target'] = train_target
    return train, test

####################### Start ##########################################
extracted_features_train = pd.read_csv('feature/train_data2.csv')
extracted_features_train = reduce_mem_usage(extracted_features_train)
extracted_features_test = pd.read_csv('feature/test_data2.csv')
extracted_features_test = reduce_mem_usage(extracted_features_test)


original_train = pd.read_csv('data/input/train.csv', usecols=['card_id'])
original_test = pd.read_csv('data/input/test.csv', usecols=['card_id'])
original_train = reduce_mem_usage(original_train)
original_test = reduce_mem_usage(original_test)
#original_train, original_test = outlier(original_train, original_test)

extracted_features_train['card_id'] = original_train['card_id']
extracted_features_test['card_id'] = original_test['card_id']
target = extracted_features_train['target']
extracted_features_train = outlier(extracted_features_train)
train, test = classify_model(extracted_features_train, extracted_features_test)
del extracted_features_train, extracted_features_test
gc.collect()
test_outlier = test[test['outliers'] >= 0.3]
test_normal = test[test['outliers'] < 0.3]
del test
gc.collect()

test_outlier['target'] = -33.5
target = train.loc[train['outliers']==0, 'target']
train = train.loc[train['outliers']==0, :]

use_cols = [col for col in train.columns if col not in 
            ['hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min', 'target', 'card_id', 'outliers', 'first_active_month','histauth_purchase_date_max', 'histauth_purchase_date_min',
             'histunauth_purchase_date_max', 'histunauth_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min',
             'new_card_id_size', 'histauth_card_id_size', 'histunauth_card_id_size','new_authorized_flag_mean']]
train = train[use_cols]
test = test_normal[use_cols]

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
print('lgb', np.sqrt(mean_squared_error(oof_lgb, target)))
total_sum = oof_lgb
print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))
cols = (feature_importance_df_lgb[["feature", "importance"]] \
        .groupby("feature") \
        .mean() \
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(30,30))
sns.barplot(x="importance", \
            y="feature", \
            data=best_features.sort_values(by="importance", \
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('./result/lgbm_importances1.png')

test_normal['target'] = predictions_lgb


sub_df = pd.read_csv('data/input/sample_submission.csv')
for card in sub_df['card_id']:
    if card in list(test_normal['card_id']):
        sub_df.loc[sub_df['card_id']==card, 'target'] = test_normal.loc[test_normal['card_id']==card, 'target'].values
    elif card in list(test_outlier['card_id']):
        sub_df.loc[sub_df['card_id']==card, 'target'] = -33.219281
    else:
        print('something is wrong!!')
#sub_df['target'] = predictions_lgb
sub_df.to_csv("./result/submission6.csv", index=False)
    