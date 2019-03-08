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

merchant_dtype = {'merchant_category_id' : 'int16', 'merchant_group_id' : 'int32', 'subsector_id' : 'int8',
                  'category_2' : 'float16', 'numerical_1' : 'float32', 'numerical_2' : 'float32', 'city_id' : 'int16',
                  'state_id' : 'int8', 'active_months_lag12' : 'uint8', 'active_months_lag3' : 'uint8',
                  'active_months_lag6' : 'uint8'}

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
    
    
    



def aggregate_new_transactions(new_trans):
    df = new_trans['purchase_date']
    df = pd.to_datetime(df)
    new_trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(new_trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    tempTimeRecorder = pd.DatetimeIndex(new_trans['purchase_date'])    
    new_trans['purchase_date_year'] = tempTimeRecorder.year
    new_trans['purchase_date_month'] = tempTimeRecorder.month
    new_trans['purchase_date_day'] = tempTimeRecorder.day
    new_trans['purchase_date_hour'] = tempTimeRecorder.hour                                                  
    new_trans.loc[:, 'purchase_day_lag'] = (datetime.date(2018, 2, 1) - df.dt.date).dt.days                                  
    
    agg_func = {
        #'authorized_flag': ['sum', 'mean'], \
        'category_1' : ['mean', 'sum'], \
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
        'purchase_amount': ['sum', 'max', 'min', 'std', 'mean'], \
        'installments': ['sum', 'max', 'min', 'std', 'mean'], \
        'purchase_date': [np.ptp], \
        'month_lag': ['min', 'max', 'sum', 'mean'], 
        
        ##### new add #########################
        'merchant_category_id' : ['nunique'],
        'merchant_category_id_X' : ['nunique'],
        'merchant_group_id' : ['nunique'],
        'subsector_id' : ['nunique'],
        'subsector_id_X' : ['nunique'],
        'state_id' : ['nunique'],
        'state_id_X' : ['nunique'],
        'city_id_X' : ['nunique'],
        'numerical_1' : ['mean', 'sum', 'std', 'max', 'min'],
        'numerical_2' : ['mean', 'sum', 'std', 'max', 'min'],
        'avg_sales_lag3' : ['mean', 'sum', 'std', 'max', 'min'],
        'avg_purchases_lag3' : ['mean', 'sum', 'std', 'max', 'min'],
        'active_months_lag3' : ['mean', 'sum', 'std', 'max', 'min'],
        'avg_sales_lag6' : ['mean', 'sum', 'std', 'max', 'min'],
        'avg_purchases_lag6' : ['mean', 'sum', 'std', 'max', 'min'],
        'active_months_lag6' : ['mean', 'sum', 'std', 'max', 'min'],
        'avg_sales_lag12' : ['mean', 'sum', 'std', 'max', 'min'],
        'avg_purchases_lag12' : ['mean', 'sum', 'std', 'max', 'min'],
        'active_months_lag12' : ['mean', 'sum', 'std', 'max', 'min'],
        
        'category_1_X_N' : ['mean'],
        'category_1_X_Y' : ['mean'],
        'category_2_X_1.0' : ['mean'],
        'category_2_X_2.0' : ['mean'],
        'category_2_X_3.0' : ['mean'], 
        'category_2_X_4.0' : ['mean'],
        'category_2_X_5.0' : ['mean'], 
        'category_4_N' : ['mean'], 
        'category_4_Y' : ['mean'],
        'most_recent_sales_range_A' : ['mean'], 
        'most_recent_sales_range_B' : ['mean'],
        'most_recent_sales_range_C' : ['mean'], 
        'most_recent_sales_range_D' : ['mean'],
        'most_recent_sales_range_E' : ['mean'], 
        'most_recent_purchases_range_A' : ['mean'],
        'most_recent_purchases_range_B' : ['mean'], 
        'most_recent_purchases_range_C' : ['mean'],
        'most_recent_purchases_range_D' : ['mean'], 
        'most_recent_purchases_range_E': ['mean'],
        
        #'purchase_date_year' : ['sum', 'max', 'min', 'mean', 'std', 'nunique'],
        #'purchase_date_month' : ['max', 'min', 'mean', 'std'],
        'purchase_day_lag' : ['min', 'max', 'mean', 'std', 'nunique']
        #'purchase_date_day' : ['sum', 'max', 'min', 'mean', 'std', 'nunique'],
        #'purchase_date_hour' : ['sum', 'max', 'min', 'mean', 'std', 'nunique']

        }
    print('0')
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    print('1')
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() \
                           for col in agg_new_trans.columns.values]
    print('2')
    agg_new_trans.reset_index(inplace=True)
    print('3')
    df = (new_trans.groupby('card_id') \
          .size() \
          .reset_index(name='new_transactions_count'))
    
    tempValue = new_trans.groupby('card_id').size()
    df2 = ((tempValue / new_trans.shape[0]).reset_index(name = 'new_card_percentage_mean'))
    df3 = (((tempValue / new_trans.shape[0])*(tempValue)).reset_index(name = 'new_card_percentage_sum'))
    print('4')
    df = pd.merge(df, df2, on = 'card_id', how = 'left')
    df = pd.merge(df, df3, on = 'card_id', how = 'left')
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    print('5')
    return agg_new_trans

    
def aggregate_historical_transactions(history):
    df = history['purchase_date']
    df = pd.to_datetime(df)
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    #history.loc[:, 'day_lag'] = history['month_lag'] * (-1) * 30                                 
    tempTimeRecorder = pd.DatetimeIndex(history['purchase_date'])    
    history['purchase_date_year'] = tempTimeRecorder.year
    history['purchase_date_month'] = tempTimeRecorder.month
    history['purchase_date_day'] = tempTimeRecorder.day
    history['purchase_date_hour'] = tempTimeRecorder.hour
    history.loc[:, 'purchase_day_lag'] = (datetime.date(2018, 2, 1) - df.dt.date).dt.days
    agg_func = { \
        'authorized_flag': ['mean', 'sum'], \
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
        'merchant_category_id' : ['nunique'],
        'subsector_id' : ['nunique'],
        'state_id' : ['nunique'],
        'purchase_amount': ['sum', 'max', 'min', 'std', 'mean'], \
        'installments': ['sum', 'max', 'min', 'std', 'mean'], \
        'purchase_date': [np.ptp], \
        'month_lag': ['min', 'max', 'mean', 'std'],
        #'day_lag' : ['min', 'max', 'mean', 'std'],
        'purchase_day_lag' : ['min', 'max', 'mean', 'std','nunique'],
        #'purchase_date_year' : ['sum', 'max', 'min', 'mean', 'std'],
#        'purchase_date_month' : ['max', 'min', 'mean', 'std'],
        #'purchase_date_day' : ['sum', 'max', 'min', 'mean', 'std', 'nunique'],
        #'purchase_date_hour' : ['sum', 'max', 'min', 'mean', 'std', 'nunique']
        
        
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() \
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id') \
          .size() \
          .reset_index(name='hist_transactions_count'))
    
    tempValue = history.groupby('card_id').size()

    df2 = ((tempValue / history.shape[0]).reset_index(name = 'history_card_percentage_mean'))
    df3 = (((tempValue / history.shape[0])*(tempValue)).reset_index(name = 'history_card_percentage_sum'))
    df = pd.merge(df, df2, on='card_id', how='left')
    df = pd.merge(df, df3, on='card_id', how='left')
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history


def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])
    
    agg_func = {
            'purchase_amount' : ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments' : ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }
    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)
    
    final_group = intermediate_group.groupby(['card_id']).agg(['mean', 'std'])
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
    df = pd.read_csv(input_file, dtype=train_dtype)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
    
#-----------------------Start-----------------------------------------------------   
merchant_info = pd.read_csv('./data/input/merchants.csv', dtype = merchant_dtype)


if debug:
    new_transactions = pd.read_csv('./data/input/new_merchant_transactions.csv', dtype=trans_dtype, nrows=1000)
else:
    new_transactions = pd.read_csv('./data/input/new_merchant_transactions.csv', dtype=trans_dtype)

# merge new_tranac and merchant
new_transactions = pd.merge(merchant_info, new_transactions, on = 'merchant_id', how = 'left', suffixes = ('_X',''))

new_transactions = pd.get_dummies(new_transactions, columns = ['category_1_X', 'category_2_X', 'category_4', 
                                                               'most_recent_sales_range', 'most_recent_purchases_range'])



  
new_transactions['authorized_flag'] = \
    new_transactions['authorized_flag'].map({'Y':1, 'N':0})
   
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])
new_transactions['category_1'] = new_transactions['category_1'].map({'Y' : 1, 'N' : 0})

print('new_transcation data start aggregating...')
new_transactions = new_transactions.dropna(axis = 0, subset=['card_id'])
new_trans = aggregate_new_transactions(new_transactions)
print('new_transcation aggregate finished.')
del new_transactions
del merchant_info
gc.collect()



print('new_transcation data loaded!!')
print('start loading historical data...')


if debug:
    historical_transactions = pd.read_csv('./data/input/historical_transactions.csv', dtype=trans_dtype, nrows=1000)
else:
    historical_transactions = pd.read_csv('./data/input/historical_transactions.csv', dtype=trans_dtype)

   
historical_transactions['authorized_flag'] = \
    historical_transactions['authorized_flag'].map({'Y':1, 'N':0})

historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y' : 1, 'N' : 0})

#my_hist_trans = add_subsector_id_feature_historical_transcation(historical_transactions)
history = aggregate_historical_transactions(historical_transactions)

#history_per_month = aggregate_per_month(historical_transactions)

#history = pd.merge(history_per_month, history, on='card_id', how='left')

#del history_per_month
del historical_transactions
gc.collect()

print('historical_data loaded!!')

if debug:
    train = read_data_debug('./data/input/train.csv')
    test = read_data_debug('./data/input/test.csv')
    
else:
    train = read_data('./data/input/train.csv')
    test = read_data('./data/input/test.csv')
    
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


del df_all
gc.collect()


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
         'min_child_samples': 20,
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

plt.figure(figsize=(35,35))
sns.barplot(x="importance", \
            y="feature", \
            data=best_features.sort_values(by="importance", \
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances_new5.png')



sub_df = pd.read_csv('data/input/sample_submission.csv')
#sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df['target'] = predictions_lgb
sub_df.to_csv("submission_new5.csv", index=False)
