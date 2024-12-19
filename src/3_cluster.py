#!/usr/bin/env python
# coding: utf-8

import sys
import time
import datetime
import fcsparser
import math
import numpy as np
import pandas as pd
import statistics
import pickle
import os
import logging
import random
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import sklearn.metrics
from sklearn.utils import class_weight
from contextlib import redirect_stdout
import umap
import sklearn.cluster as cluster
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import multiprocessing
#import hdbscan

## ============================================================

## load UMAP projections, apply clustering and run RF

## ============================================================

random.seed(1)

RUN_ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

print(RUN_ID)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logfilename = "./log/3_" + RUN_ID + ".logfile.log"
        if os.path.exists(logfilename):
            os.remove(logfilename)
        self.log = open(logfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def col_standardize(x):
    return x/np.max(x)

def get_RF_params(x_train, y_train):
    kf = sklearn.model_selection.StratifiedKFold(n_splits=2)
    i, (xid_train, xid_test) = list(enumerate(kf.split(x_train, y_train)))[0]
    y_1 = y_train[xid_train]
    y_2 = y_train[xid_test]
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_1), y=y_1)
    cw = {np.unique(y_1)[0]:cw[0], np.unique(y_1)[1]:cw[1]}
    est_list = [10, 20, 50, 100, 200, 500]
    leaf_list = [10, 20, 50, 100, 200, 500]
    param_list = list(itertools.product(est_list, leaf_list))
    res_list = []
    for j in range(len(param_list)):
        R = RandomForestClassifier(n_estimators=param_list[j][0], max_leaf_nodes=param_list[j][1], n_jobs=-1, verbose=0, random_state=1)#, class_weight=cw)
        R.fit(x_train[xid_train, :], y_train[xid_train])
        y_pred = R.predict(x_train[xid_test, :])
        res_list.append(sklearn.metrics.balanced_accuracy_score(y_2, y_pred))
    param = param_list[np.argmax(res_list)]
    return param


## ---------------------------------------------------------------------

DIRPATH = ''
RUN_2 = '' # provide RUN_ID from running 2_UMAP.py
fold_i = 0 # provide fold to process

## ---------------------------------------------------------------------

t0 = datetime.datetime.now()
T0 = time.process_time()
print("TSTAMP 0 : " + str(t0))

ann = pd.read_csv(DIRPATH + "annotation.csv", sep="\t")
print(ann.shape)

sel_pns = pd.value_counts(ann['PNS']).index.tolist()[0]
sel_ix = np.where(ann['PNS'] == sel_pns)[0]
sel_ann = ann.iloc[sel_ix, ]

print('Selecting panel ' + sel_pns)

xid_sel = np.where((sel_ann['Diagnosis'] == 'positive') | (sel_ann['Diagnosis'] == 'negative'))
y_true = np.asarray(sel_ann.iloc[xid_sel]['Diagnosis'])
perc_true = np.asarray(sel_ann.iloc[xid_sel]['Percentage'])
le = preprocessing.LabelEncoder()
le.fit(y_true)
print(perc_true.shape)
print(y_true.shape)

fcs_files = sel_ann.iloc[xid_sel]['New File Name']
fcs_files = fcs_files.to_numpy()

os.makedirs("./obj/3/" + RUN_ID)
os.makedirs("./obj/3/" + RUN_ID + '/' + str(fold_i))

with open('./obj/2/' + RUN_2 + '/' + str(fold_i) + '/dat.obj', 'rb') as f:
    xid_train, xid_test = pickle.load(f)

print(pd.value_counts(y_true[xid_train]))
print(pd.value_counts(y_true[xid_test]))

y_train = y_true[xid_train]
y_test = y_true[xid_test]

# set number of clusters for k-means
nclust = 1000

U = np.load('./obj/2/' + RUN_2 + '/' + str(fold_i) + "/U.npy")
V = np.load('./obj/2/' + RUN_2 + '/' + str(fold_i) + "/V.npy")
print(U.shape)
print(V.shape)

print("nclust = " + str(nclust))

t1 = datetime.datetime.now()
T1 = time.process_time()
print("TSTAMP 1 : " + str(t1))
print("time elapsed: " + str(t1-t0))
print("process time elapsed: " + str(T1-T0))

KM = cluster.KMeans(n_clusters=nclust)
clust_labels = KM.fit_predict(V)

with open('./obj/3/' + RUN_ID + '/' + str(fold_i) + '/dat.obj', 'wb') as f:
    pickle.dump([y_train, y_test, RUN_2, nclust, clust_labels], f)

C_list = []
for i in range(len(fcs_files)):
    if i % 20 == 0:
        print(i, "/", len(fcs_files))
    Vi = np.load('./obj/2/' + RUN_2 + '/' + str(fold_i) + '/projections/PNS1/' + fcs_files[i] + '.npy')
    Ci = KM.predict(Vi)
    C_list.append(Ci)

print(len(C_list))
with open('./obj/3/' + RUN_ID + '/' + str(fold_i) + '/C_list.obj', 'wb') as f:
    pickle.dump(C_list, f)

t2 = datetime.datetime.now()
T2 = time.process_time()
print("TSTAMP 2 : " + str(t2))
print("time elapsed: " + str(t2-t1))
print("process time elapsed: " + str(T2-T1))

x_list = []
for i in range(len(C_list)):
    xi = [np.sum(C_list[i] == j) for j in range(nclust)]
    xi = np.asarray(xi)
    x_list.append(xi)

Cmat = np.stack(x_list)
np.save('./obj/3/' + RUN_ID + '/' + str(fold_i) + "/Cmat.npy", Cmat)

x_list = []
for i in range(len(C_list)):
    xi = [np.sum(C_list[i] == j) for j in range(nclust)]
    xi = np.asarray(xi)
    xi = xi/len(C_list[i])
    x_list.append(xi)

Z = np.stack(x_list)
np.save('./obj/3/' + RUN_ID + '/' + str(fold_i) + "/Z.npy", Z)

Z_train = Z[xid_train, :]
Z_test = Z[xid_test, :]

t3 = datetime.datetime.now()
T3 = time.process_time()
print("TSTAMP 3 : " + str(t3))
print("time elapsed: " + str(t3-t2))
print("process time elapsed: " + str(T3-T2))

params = get_RF_params(Z_train, y_train)
print(params)
R = RandomForestClassifier(n_estimators=params[0], max_leaf_nodes=params[1], n_jobs=-1, verbose=0, random_state=1)
R.fit(Z_train, y_train)
y_pred_train = R.predict(Z_train)
y_pred_test = R.predict(Z_test)
ba_train = sklearn.metrics.balanced_accuracy_score(y_true=y_train, y_pred=y_pred_train)
ba_test = sklearn.metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred_test)
print(ba_train, ba_test)
print("--- train ---")
print(sklearn.metrics.classification_report(y_true=y_train, y_pred=y_pred_train))
print("---- test ---")
print(sklearn.metrics.classification_report(y_true=y_test, y_pred=y_pred_test))

t4 = datetime.datetime.now()
T4 = time.process_time()
print("TSTAMP 4 : " + str(t4))
print("time elapsed: " + str(t4-t3))
print("process time elapsed: " + str(T4-T3))

print("-------- end of fold ----------")
print("total time elapsed: " + str(t4-t0))
print("total process time elapsed: " + str(T4-T0))






















#
