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

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)

    return df



def referenceData(train):
    refer = pd.to_datetime(train['reference_date'])
    df = pd.to_datetime(train['hist_purchase_date_max'])
    df2 = pd.to_datetime(train['hist_purchase_date_min'])
    df3 = pd.to_datetime(train['new_purchase_date_max'])
    df4 = pd.to_datetime(train['new_purchase_date_min'])
    
    train.loc[:, 'reference_day_lag'] = (refer.dt.date - df.dt.date).dt.days
    train.loc[:, 'reference_month_diff'] = train.loc[:, 'reference_day_lag'] // 30
    train.loc[:, 'reference_month_diff'] += train.loc[:, 'category_month_lag']
    train.loc[:, 'reference_day_lag2'] = (refer.dt.date - df2.dt.date).dt.days
    train.loc[:, 'reference_month_diff2'] = train.loc[:, 'reference_day_lag2'] // 30
    train.loc[:, 'reference_month_diff2'] += train.loc[:, 'category_month_lag']
    
    
    train.loc[:, 'reference_day_lag3'] = (refer.dt.date - df3.dt.date).dt.days
    train.loc[:, 'reference_month_diff3'] = train.loc[:, 'reference_day_lag'] // 30
    train.loc[:, 'reference_month_diff3'] += train.loc[:, 'category_month_lag']
    train.loc[:, 'reference_day_lag4'] = (refer.dt.date - df4.dt.date).dt.days
    train.loc[:, 'reference_month_diff4'] = train.loc[:, 'reference_day_lag2'] // 30
    train.loc[:, 'reference_month_diff4'] += train.loc[:, 'category_month_lag']
    
    #train.loc[:, 'reference_elapsed_time'] = (refer.dt.date - pd.to_datetime(train['first_active_month']).dt.date).dt.days
    return train
    




def aggragate_historical_holiday(history):
#    df = history['purchase_date']
#    df = pd.to_datetime(df)
#    history.loc[:, 'purchase_date'] = df
#    history.loc[:, 'month_diff'] = ((datetime.date(2018, 4, 30) - df.dt.date).dt.days) // 30
#    history.loc[:, 'month_diff'] += history.loc[:, 'month_lag']
#    history.loc[:, 'duration'] = history.loc[:, 'purchase_amount'] * history.loc[:, 'month_diff']
#    history.loc[:, 'amount_month_ratio'] = history.loc[:, 'purchase_amount'] / history.loc[:, 'month_diff']

    agg_func = {
            'card_id': ['count'],
            'purchase_amount': ['sum']
#            'price': ['mean', 'max', 'min', 'var']
#            'active_months_lag3': ['min', 'max', 'mean', 'sum'],
#            'active_months_lag6': ['min', 'max', 'mean', 'sum'],
#            'active_months_lag12': ['min', 'max', 'mean', 'sum'],
#            'card_id': ['size']
#            'duration': ['mean', 'min', 'max'],
#            'amount_month_ratio': ['mean', 'min', 'max']
            }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() \
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    return agg_history
       

def aggragate_new_holiday(new):
#    df = new['purchase_date']
#    df = pd.to_datetime(df)
#    new.loc[:, 'purchase_date'] = df
#    new.loc[:, 'month_diff'] = ((datetime.date(2018, 4, 30) - df.dt.date).dt.days) // 30
#    new.loc[:, 'month_diff'] += new.loc[:, 'month_lag']
#    new.loc[:, 'duration'] = new.loc[:, 'purchase_amount'] * new.loc[:, 'month_diff']
#    new.loc[:, 'amount_month_ratio'] = new.loc[:, 'purchase_amount'] / new.loc[:, 'month_diff']

    agg_func = {
            'card_id': ['count'],
#            'purchase_amount': ['sum']
#            'price': ['mean', 'max', 'min', 'var']
#            'active_months_lag3': ['min', 'max', 'mean', 'sum'],
#            'active_months_lag6': ['min', 'max', 'mean', 'sum'],
#            'active_months_lag12': ['min', 'max', 'mean', 'sum'],
#            'card_id': ['size']
#            'duration': ['mean', 'min', 'max'],
#            'amount_month_ratio': ['mean', 'min', 'max']
    
            }
    agg_new = new.groupby(['card_id']).agg(agg_func)
    agg_new.columns = ['new_' + '_'.join(col).strip() \
                           for col in agg_new.columns.values]
    agg_new.reset_index(inplace=True)
    return agg_new

def RScore(x,p,d):
    if x <= d[p][0.011]:
        return 1
    elif x <= d[p][0.050]:
        return 2
    elif x <= d[p][0.25]: 
        return 3
    elif x <= d[p][0.5]:
        return 4
    elif x <= d[p][0.75]:
        return 5
    elif x <= d[p][0.95]:
        return 6
    elif x <= d[p][0.989]:
        return 7
    else:
        return 8
    
def FMScore(x,p,d):
    if x <= d[p][0.011]:
        return 8
    elif x <= d[p][0.050]:
        return 7
    elif x <= d[p][0.25]: 
        return 6
    elif x <= d[p][0.5]:
        return 5
    elif x <= d[p][0.75]:
        return 4
    elif x <= d[p][0.95]:
        return 3
    elif x <= d[p][0.989]:
        return 2
    else:
        return 1
    
def hist_RFM(cardrfm):
    quantiles = cardrfm.quantile(q=[0.011,0.05,0.25,0.5,0.75,0.95,0.989])
    quantiles = quantiles.to_dict()
    cardrfm['r_quantile'] = cardrfm['reference_day_lag'].apply(RScore, args=('reference_day_lag',quantiles))
    cardrfm['f_quantile'] = cardrfm['hist_card_id_count'].apply(FMScore, args=('hist_card_id_count',quantiles))
    cardrfm['v_quantile'] = cardrfm['hist_purchase_amount_sum'].apply(FMScore, args=('hist_purchase_amount_sum',quantiles))
    cardrfm['RFMindex'] = cardrfm.r_quantile.map(str) + cardrfm.f_quantile.map(str) + cardrfm.v_quantile.map(str) 
    cardrfm['RFMScore'] = cardrfm.r_quantile + cardrfm.f_quantile + cardrfm.v_quantile
    cardrfm.RFMindex = cardrfm.RFMindex.astype(int)
    RFMindex = pd.DataFrame(np.unique(np.sort(cardrfm.RFMindex)),columns=['RFMindex'])
    RFMindex.index = RFMindex.index.set_names(['RFMIndex'])
    RFMindex.reset_index(inplace=True)
    cardrfm = pd.merge(cardrfm, RFMindex, on='RFMindex', how='left')
    cardrfm.drop(["RFMindex"], axis=1, inplace=True)
    return cardrfm
    
def new_RFM(cardrfm_new):
    quantiles = cardrfm_new.quantile(q=[0.011,0.05,0.25,0.5,0.75,0.95,0.989])
    quantiles = quantiles.to_dict()
    cardrfm_new['rnew_quantile'] = cardrfm_new['reference_day_lag3'].apply(RScore, args=('reference_day_lag3',quantiles))
    cardrfm_new['fnew_quantile'] = cardrfm_new['new_card_id_count'].apply(FMScore, args=('new_card_id_count',quantiles))
    cardrfm_new['vnew_quantile'] = cardrfm_new['new_purchase_amount_sum'].apply(FMScore, args=('new_purchase_amount_sum',quantiles))
    cardrfm_new['RFMnewindex'] = cardrfm_new.rnew_quantile.map(str)+cardrfm_new.fnew_quantile.map(str)+cardrfm_new.vnew_quantile.map(str) 
    cardrfm_new['RFMnewScore'] = cardrfm_new.rnew_quantile+cardrfm_new.fnew_quantile+cardrfm_new.vnew_quantile 
    cardrfm_new.RFMnewindex= cardrfm_new.RFMnewindex.astype(int)
    RFMnewindex=pd.DataFrame(np.unique(np.sort(cardrfm_new.RFMnewindex)),columns=['RFMnewindex'])
    RFMnewindex.index=RFMnewindex.index.set_names(['RFMnewIndex'])
    RFMnewindex.reset_index(inplace=True)
    cardrfm_new =pd.merge(cardrfm_new,RFMnewindex,on='RFMnewindex',how='left')
    cardrfm_new.drop(["RFMnewindex"], axis=1, inplace=True)
    return cardrfm_new

def feature_comb(df):
    df['feature_comb'] = df.feature_1.map(str) + df.feature_2.map(str) + df.feature_3.map(str)
    df['feature_comb']= df['feature_comb'].astype(int)
    featureindex=pd.DataFrame(np.unique(np.sort(df['feature_comb'])),columns=['feature_comb'])
    featureindex.index=featureindex.index.set_names(['feature_comb_index'])
    featureindex.reset_index(inplace=True)
    df =pd.merge(df,featureindex,on='feature_comb',how='left')
    df.drop(["feature_comb"], axis=1, inplace=True)
    return df

def successive_aggregates(df, field1, field2):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    return u
####################### Start ##########################################
extracted_features_train = pd.read_csv('feature/train2_referenceDate_best.csv')
extracted_features_train = reduce_mem_usage(extracted_features_train)
extracted_features_test = pd.read_csv('feature/test2_referenceDate_best.csv')
extracted_features_test = reduce_mem_usage(extracted_features_test)


original_train = pd.read_csv('data/input/train.csv', usecols=['card_id', 'target', 'feature_1', 'feature_2', 'feature_3'])
original_test = pd.read_csv('data/input/test.csv', usecols=['card_id', 'feature_1', 'feature_2', 'feature_3'])
original_train = reduce_mem_usage(original_train)
original_test = reduce_mem_usage(original_test)

original_train = feature_comb(original_train)
original_test = feature_comb(original_test)
extracted_features_train['card_id'] = original_train['card_id']
extracted_features_test['card_id'] = original_test['card_id']
extracted_features_train['feature_comb_index'] = original_train['feature_comb_index']
extracted_features_test['feature_comb_index'] = original_test['feature_comb_index']

#target = original_train['target']

target = original_train['target']
del original_train, original_test
gc.collect()

#train = extracted_features_train
#test = extracted_features_test


## read historical & new_transactions
historical = pd.read_csv('data/input/historical_transactions.csv', usecols=['card_id', 'purchase_amount'])
new_tran = pd.read_csv('data/input/new_merchant_transactions.csv', usecols=['card_id', 'category_1', 'installments', 'purchase_amount', 'city_id'])
historical = reduce_mem_usage(historical)
new_tran = reduce_mem_usage(new_tran)

additional_fields = successive_aggregates(new_tran, 'category_1', 'purchase_amount')
additional_fields = additional_fields.merge(successive_aggregates(new_tran, 'installments', 'purchase_amount'), on = 'card_id', how='left')
additional_fields = additional_fields.merge(successive_aggregates(new_tran, 'city_id', 'purchase_amount'), on = 'card_id', how='left')
additional_fields = additional_fields.merge(successive_aggregates(new_tran, 'category_1', 'installments'), on = 'card_id', how='left')




historical = aggragate_historical_holiday(historical)
new_tran = aggragate_new_holiday(new_tran)

#
#
train = pd.merge(extracted_features_train, historical, how='left', on='card_id')
train = pd.merge(train, new_tran, how='left', on='card_id')
train = pd.merge(train, additional_fields, how='left', on='card_id')
test = pd.merge(extracted_features_test, historical, how='left', on='card_id')
test = pd.merge(test, new_tran, how='left', on='card_id')
test = pd.merge(test, additional_fields, how='left', on='card_id')
del historical
del new_tran
del extracted_features_train
del extracted_features_test
gc.collect()
train = hist_RFM(train)
train = new_RFM(train)
test = hist_RFM(test)
test = new_RFM(test)
#merge = pd.concat([train, test], axis=0, ignore_index=True)
#reference = pd.read_csv('data/input/Cardreferencedate.csv')
#merge = pd.merge(merge, reference, how='left', on='card_id')
#merge = referenceData(merge)
#train = merge.iloc[:201917]
#test = merge.iloc[201917:]
#del merge
#del reference
#del historical
#del new_tran

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


ref = pd.read_csv('data/input/Cardreferencedate.csv', usecols=['card_id', 'reference_date'])
ref['reference_date'] = pd.DatetimeIndex(ref['reference_date']).astype(np.int64) * 1e-9
merge = pd.concat([train, test], axis=0, ignore_index=True)
merge = pd.merge(merge, ref, how='left', on='card_id')

#merge['value_new_hist'] = merge['new_purchase_amount_sum'] / merge['hist_purchase_amount_sum']
#merge['frequency_new_hist'] = merge['new_card_id_count'] / merge['hist_card_id_count']

train = merge.iloc[:201917]
test = merge.iloc[201917:]
del ref
del merge
gc.collect()



use_cols = [col for col in train.columns if col not in 
            ['hist_purchase_amount_sum', 'hist_reference_day_lag_elapse', 'new_reference_day_lag_elapse', 'hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min', 'target', 'card_id', 'outliers', 'first_active_month','histauth_purchase_date_max', 'histauth_purchase_date_min',
             'histunauth_purchase_date_max', 'histunauth_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min',
             'new_card_id_count', 'hist_card_id_count', 'histauth_card_id_size', 'histunauth_card_id_size','new_authorized_flag_mean']]
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
#xgb_params = {
#        'objective': 'reg:linear',
#        'booster': "gbtree",
#        'eval_metric': "rmse",
#        'eta': 0.02,
#        'max_depth': 7,
#        'min_child_weight': 100,
#        'gamma': 0,
#        'subsample': 0.85,
#        'colsample_bytree': 0.8,
#        'colsample_bylevel': 0.85,
#        'alpha': 0,
#        'silent': True,
#        'lambda': 0.1
#        }
#lgb_params = {
#        'objective': 'regression_l2',
#        'boosting_type': 'gbdt', 
#        'n_jobs': 4, 'max_depth': 7, 
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


#################XGB###################################
#xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, \
#           'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}
#
#
#
#FOLDs = KFold(n=train.shape[0], n_folds=10, shuffle=True, random_state=15)
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
#total_sum = oof_lgb
print("CV score: {:<8.5f}".format(mean_squared_error(oof_lgb, target)**0.5))


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
#train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
#test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()
#folds = KFold(n=train_stack.shape[0], n_folds=5, shuffle=True, random_state=15)
#oof = np.zeros(train_stack.shape[0])
#predictions = np.zeros(test_stack.shape[0])
#for fold_, (trn_idx, val_idx) in enumerate(folds):
#    print("fold n°{}".format(fold_))
#    clf = Ridge(alpha=1)
#    clf.fit(train_stack[trn_idx], target.iloc[trn_idx])
#    oof[val_idx] = clf.predict(train_stack[val_idx])
#    predictions += clf.predict(test_stack) / folds.n_folds
#print('After stacking, valid-rmse: ', np.sqrt(mean_squared_error(target.values, oof)))

sub_df = pd.read_csv('data/input/sample_submission.csv')
#sub_df['target'] = predictions
#sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df.to_csv("./result/submission18.csv", index=False)



