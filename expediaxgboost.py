# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:36:32 2017

@author: wenja
"""

import numpy as np
import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
from sklearn import cross_validation
from xgboost import plot_importance

def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', -metric

train = pd.read_csv("train_n.csv")
test = pd.read_csv("test_n.csv")
test = test.drop(id, axis=1)
destinations = pd.read_csv("feature_pca.csv")

#impute missing values
train.isnull().any()
train['orig_destination_distance'] = train.orig_destination_distance.fillna(train.orig_destination_distance.mean())
train['len_stay'] = train.len_stay.fillna(train.len_stay.mean())
test.isnull().any()
test['orig_destination_distance'] = test.orig_destination_distance.fillna(test.orig_destination_distance.mean())
test['len_stay'] = test.len_stay.fillna(test.len_stay.mean())

train = pd.merge(train, destinations, how='left', on='srch_destination_id')
test = pd.merge(test, destinations, how='left', on = 'srch_destination_id')
y = train.hotel_cluster
trainX = train.drop('hotel_cluster', axis=1)
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainX, y, stratify=y, test_size=0.2)


clf = xgb.XGBClassifier(objective = 'multi:softmax', max_depth = 5, n_estimators=300,
                        learning_rate=0.01, nthread=4, subsample=0.7, colsample_bytree=0.7,
                        min_child_weight=3, silent=False)
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric=map5eval, eval_set=[(X_train, y_train),(X_valid, y_valid)])

pickle.dump(clf, open('xgbmodel', 'wb'))
#feature importance

plot_importance(clf)
cols = X_train.columns.tolist()
X_test = test[cols]
pred = clf.predict_proba(X_test)
col_ind = np.argsort(-pred, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

submission = pd.read_csv('sample_submission.csv')
sub=pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('predxgb_sub.csv', index=False)

#0.2151 for test data
pred_tr=clf.predict_proba(trainX)
col_ind = np.argsort(-pred_tr, axis=1)[:,:5]
#hc_tr = [' '.join(row.astype(str)) for row in col_ind]
predicted = pd.DataFrame(data=hc)
actual = pd.DataFrame(data=train.hotel_cluster)
def mapk(actual, predicted):
    score = 0.0
    for i in range(5):
        score += np.sum(actual==predicted.iloc[:,i])/(i+1)
    score /= actual.shape[0]
    return score