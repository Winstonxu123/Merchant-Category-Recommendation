# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:29:28 2019

@author: Administrator
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import gc

#model1_train = pd.read_csv('model/model1_train.csv', usecols=['card_id', 'target_like', 'target'])
model1_train = pd.read_csv('model/train2_referenceDate9features_stack_model.csv', usecols=['card_id', 'target_like'])
model2_train = pd.read_csv('model/train4_referenceDate9features_stack_model.csv', usecols=['card_id', 'target_like'])
#model1_test = pd.read_csv('model/model1_test.csv', usecols=['card_id', 'target'])
#model2_test = pd.read_csv('model/model2_test.csv', usecols=['card_id', 'target'])
tar = pd.read_csv('data/input/train.csv', usecols=['target'])
#----------------------stack-------------------------------------------------------
#from sklearn.linear_model import Ridge
#from sklearn.cross_validation import KFold
#from sklearn.metrics import mean_squared_error
#import numpy as np
#model1_test = pd.read_csv('result/submission_pub3.683_best.csv', usecols=['card_id', 'target'])
#model2_test = pd.read_csv('result/submission7.csv', usecols=['card_id', 'target'])
#train_stack = np.vstack([list(model1_train['target_like']), list(model2_train['target_like'])]).transpose()
#test_stack = np.vstack([list(model1_test['target']), list(model2_test['target'])]).transpose()
#target = model1_train['target']
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
#
#sub_df = pd.read_csv('data/input/sample_submission.csv')
#sub_df['target'] = predictions
#
#sub_df.to_csv("./result/submission8.csv", index=False)

#---------------------regression-----------------------------------------------------
X = pd.DataFrame()
X_1 = model1_train[['target_like']]
X_2 = model2_train[['target_like']]
X['target_like1'] = X_1
X['target_like2'] = X_2
y = tar['target']

del X_1, X_2
gc.collect()
model = LinearRegression()
model.fit(X, y)
print(model.coef_)
    

