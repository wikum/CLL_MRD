#!/usr/bin/env python
# coding: utf-8

## ------------------
## 2025-05-20
## Wikum Dinalankara
## ------------------

import sys
import time
import datetime
import fcsparser #
import math
import numpy as np
import pandas as pd #
import statistics
import pickle
import os
import logging
import random
import itertools
import umap
import multiprocessing
import sklearn.model_selection
from sklearn import preprocessing
import sklearn.cluster as cluster
import configparser
import re

## ============================================================

## cluster UMAP + projections

## ============================================================

random.seed(1)

RUN_ID = '000000' # provide RUN ID from 2_UMAP.py here

CUR_ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

print(RUN_ID)
print(CUR_ID)

if not os.path.exists('./log'):
    os.makedirs('./log')

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logfilename = "./log/4_cluster_" + RUN_ID + '_' + CUR_ID + ".logfile.log"
        if os.path.exists(logfilename):
            os.remove(logfilename)
        self.log = open(logfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def cluster_projections(fold_i, KM, nclust, fcs_files, projection_path):
    C_list = []
    for i in range(len(fcs_files)):
        if i % 20 == 0:
            print(i, "/", len(fcs_files))
        Vi = np.load(projection_path + fcs_files[i] + '.npy')
        Ci = KM.predict(Vi)
        C_list.append(Ci)
    print(len(C_list))
    with open('./obj/4/' + CUR_ID + '/' + str(fold_i) + '/C_list.obj', 'wb') as f:
        pickle.dump(C_list, f)
    x_list = []
    for i in range(len(C_list)):
        xi = [np.sum(C_list[i] == j) for j in range(nclust)]
        xi = np.asarray(xi)
        x_list.append(xi)
    Cmat = np.stack(x_list)
    print(Cmat.shape)
    np.save('./obj/4/' + CUR_ID + '/' + str(fold_i) + "/Cmat.npy", Cmat)
    x_list = []
    for i in range(len(C_list)):
        xi = [np.sum(C_list[i] == j) for j in range(nclust)]
        xi = np.asarray(xi)
        xi = xi/len(C_list[i])
        x_list.append(xi)
    Z = np.stack(x_list)
    print(Z.shape)
    np.save('./obj/4/' + CUR_ID + '/' + str(fold_i) + "/Z.npy", Z)
    return Z

def process_fold(fold_i, nclust, fcs_files, y_true):
    print('---------------- FOLD ' + str(fold_i) + ' [cluster] ----------------')
    print('nclust = ' + str(nclust))
    os.makedirs("./obj/4/" + CUR_ID, exist_ok=True)
    os.makedirs("./obj/4/" + CUR_ID + '/' + str(fold_i), exist_ok=True)
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/dat.obj", "rb") as f:
        xid_train, xid_test, M = pickle.load(f)
    f_train = fcs_files[xid_train]
    f_test = fcs_files[xid_test]
    y_train = y_true[xid_train]
    y_test = y_true[xid_test]
    # sample M cells
    m = int(M/len(f_train))
    # load
    #U = np.load("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/U.npy")
    V = np.load("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/V.npy")
    KM = cluster.KMeans(n_clusters=nclust)
    clust_labels = KM.fit_predict(V)
    with open('./obj/4/' + CUR_ID + '/' + str(fold_i) + '/dat.obj', 'wb') as f:
        pickle.dump([y_train, y_test, f_train, f_test, RUN_ID, nclust, clust_labels], f)
    projection_path = "./obj/2/" + RUN_ID + "/" + str(fold_i) + "/projections/"
    Z = cluster_projections(fold_i=fold_i, KM=KM, nclust=nclust, fcs_files=fcs_files, projection_path=projection_path)
    Z_train = Z[xid_train, :]
    Z_test = Z[xid_test, :]
    print(Z_train.shape)
    print(Z_test.shape)
    np.save('./obj/4/' + CUR_ID + '/' + str(fold_i) + "/Z_train.npy", Z_train)
    np.save('./obj/4/' + CUR_ID + '/' + str(fold_i) + "/Z_test.npy", Z_test)
    print('-- done FOLD --')


## ---------------------------------------------------------------------

t0 = datetime.datetime.now()
T0 = time.process_time()
print("TSTAMP 0 : " + str(t0))

config = configparser.ConfigParser()
config.read('../settings.ini')

if config.has_option('DEFAULT', 'dirpath'):
    DIRPATH = config.get('DEFAULT', 'dirpath')
else:
    DIRPATH = '' # insert path here

ann = pd.read_csv(DIRPATH + "annotation.csv", sep="\t")
print(str(ann.shape[0]) + ' cases available')

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

t1 = datetime.datetime.now()
T1 = time.process_time()
print("TSTAMP 1 : " + str(t1))
print("time elapsed: " + str( round((t1-t0).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T1-T0, 5)))

process_fold(fold_i=0, nclust=1000, fcs_files=fcs_files, y_true=y_true)

t2 = datetime.datetime.now()
T2 = time.process_time()
print("TSTAMP 2 : " + str(t2))
print("time elapsed: " + str( round((t2-t1).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T2-T1, 5)))

process_fold(fold_i=1, nclust=1000, fcs_files=fcs_files, y_true=y_true)

t3 = datetime.datetime.now()
T3 = time.process_time()
print("TSTAMP 3 : " + str(t3))
print("time elapsed: " + str( round((t3-t2).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T3-T2, 5)))

process_fold(fold_i=2, nclust=1000, fcs_files=fcs_files, y_true=y_true)

t4 = datetime.datetime.now()
T4 = time.process_time()
print("TSTAMP 4 : " + str(t4))
print("time elapsed: " + str( round((t4-t3).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T4-T3, 5)))

print("time elapsed [TOTAL]: " + str( round((t4-t0).total_seconds(), 5) ))
print("process time elapsed [TOTAL]: " + str(round(T4-T0, 5)))

print('done.')


