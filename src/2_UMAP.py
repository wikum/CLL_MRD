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
import umap
import multiprocessing
import sklearn.model_selection
from sklearn import preprocessing
import configparser

## ============================================================

## prepare UMAP + projections

## ============================================================

random.seed(1)

RUN_ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

print(RUN_ID)

if not os.path.exists('./log'):
    os.makedirs('./log')

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logfilename = "./log/2_" + RUN_ID + ".logfile.log"
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

def get_fcsdata(f):
    meta, fcsdat = fcsparser.parse(f, reformat_meta=True)
    # omit time
    H = fcsdat.iloc[:, range(1, fcsdat.shape[1])]
    H = H.to_numpy()
    H[H < 0] = 0
    H =  np.log(H + 1)
    return H

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

num_folds = 3
if config.has_option('DEFAULT', 'K'):
    num_folds = config.getint('DEFAULT', 'K')

M = 1000000
if config.has_option('DEFAULT', 'M'):
    M = config.getint('DEFAULT', 'M')

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

kf = sklearn.model_selection.StratifiedKFold(n_splits=num_folds)

# create directory for this run
if not os.path.exists('./obj/2'):
    os.makedirs('./obj/2')
os.makedirs("./obj/2/" + RUN_ID)

# start k-folds
for fold_i, (xid_train, xid_test) in enumerate(kf.split(fcs_files, y_true)):
    t1 = datetime.datetime.now()
    T1 = time.process_time()
    print('---------------- FOLD ' + str(fold_i) + '----------------')
    print("TSTAMP 1 : " + str(t1))
    os.makedirs("./obj/2/" + RUN_ID + "/" + str(fold_i))
    os.makedirs("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/projections")
    f_train = fcs_files[xid_train]
    f_test = fcs_files[xid_test]
    y_train = y_true[xid_train]
    y_test = y_true[xid_test]
    # sample M cells
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/dat.obj", "wb") as f:
        pickle.dump([xid_train, xid_test, M], f)
    m = int(M/len(f_train))
    print('sampling...')
    Ui_list = []
    for i in range(len(f_train)):
        Xi = get_fcsdata(DIRPATH + f_train[i])
        random.seed(1)
        Ui_list.append(Xi[random.sample(range(Xi.shape[0]), min(m, Xi.shape[0])), :])
    U = np.concatenate(Ui_list)
    print(U.shape)
    t2 = datetime.datetime.now()
    T2 = time.process_time()
    print("TSTAMP 2 : " + str(t2))
    print("time elapsed: " + str( round((t2-t1).total_seconds(), 5) ))
    print("process time elapsed: " + str(round(T2-T1, 5)))
    print('UMAP...')
    embedder = umap.UMAP() #min_dist=min_dist, n_neighbors=n_neighbors, metric=metric)
    embedding = embedder.fit(U)
    V = embedder.embedding_
    print(V.shape)
    t3 = datetime.datetime.now()
    T3 = time.process_time()
    print("TSTAMP 3 : " + str(t3))
    print("time elapsed: " + str( round((t3-t2).total_seconds(), 5) ))
    print("process time elapsed: " + str(round(T3-T2, 5)))
    # save
    np.save("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/U.npy", U)
    np.save("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/V.npy", V)
    print("projections..")
    def process_case_umap(f):
        tf0 = time.process_time()
        Xi = get_fcsdata(DIRPATH + f)
        Xi = Xi[range(100), :]
        Vi = embedder.transform(Xi)
        np.save("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/projections/" + f + ".npy", Vi)
        tf1 = time.process_time()
        return tf1-tf0

    w_list = []
    #for i in range(len(fcs_files)):
    #    w_list.append(process_case_umap(fcs_files[i]))
    with multiprocessing.Pool() as pool:
        w_list = pool.map(process_case_umap, fcs_files[0:5])
    print('threads completed:' + str(length(w_list)))
    print("process time elapsed within threads: " + str(round(sum(w_list), 5)))
    t4 = datetime.datetime.now()
    T4 = time.process_time()
    print("TSTAMP 4 : " + str(t4))
    print("time elapsed: " + str( round((t4-t3).total_seconds(), 5) ))
    print("process time elapsed: " + str(round(T4-T3, 5)))
    print('done [FOLD]')
    print("time elapsed [FOLD]: " + str( round((t4-t1).total_seconds(), 5) ))
    print("process time elapsed [FOLD]: " + str(round(T4-T1, 5)))


t5 = datetime.datetime.now()
T5 = time.process_time()
print("TSTAMP 5 : " + str(t5))
print("time elapsed [TOTAL]: " + str( round((t5-t0).total_seconds(), 5) ))
print("process time elapsed [TOTAL]: " + str(round(T5-T0, 5)))

print('done.')

















#
