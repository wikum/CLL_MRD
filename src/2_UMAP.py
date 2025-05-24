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
import configparser
import re

## ============================================================

## prepare UMAP

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
        pass

sys.stdout = Logger()

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def col_standardize(x):
    return x/np.max(x)

def get_fcsdata(f, omit_singlets=False, omit_doublets=False):
    meta, fcsdat = fcsparser.parse(f, reformat_meta=True)
    # omit time
    H = fcsdat.iloc[:, range(1, fcsdat.shape[1])]
    H = H.to_numpy()
    H[H < 0] = 0
    if omit_doublets:
        H = remove_doublets(H)
    if omit_singlets:
        H = remove_singlets(H)
    H =  np.log(H + 1)
    return H

def write_xid_train_test(xid_train, xid_test, file):
    u1 = pd.DataFrame(zip(xid_train, ['train' for i in range(xid_train.shape[0])]))
    u1.columns = ['id', 'set']
    u2 = pd.DataFrame(zip(xid_test, ['test' for i in range(xid_test.shape[0])]))
    u2.columns = ['id', 'set']
    u = pd.concat([u1, u2])
    u.to_csv(file)


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
#M = 1000

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
    os.makedirs("./obj/2/" + RUN_ID + "/" + str(fold_i))
    os.makedirs("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/projections")
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/dat.obj", "wb") as f:
        pickle.dump([xid_train, xid_test, M], f)
    write_xid_train_test(xid_train=xid_train, xid_test=xid_test, file="./obj/2/" + RUN_ID + "/" + str(fold_i) + "/xid.txt")


## generate umap
def gen_UMAP(fold_i, omit_singlets=False, omit_doublets=False, omit_lowCD19=False):
    t1 = datetime.datetime.now()
    T1 = time.process_time()
    print('---------------- FOLD ' + str(fold_i) + ' [UMAP] ----------------')
    print("TSTAMP 1 : " + str(t1))
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/dat.obj", "rb") as f:
        xid_train, xid_test, M = pickle.load(f)
    f_train = fcs_files[xid_train]
    f_test = fcs_files[xid_test]
    y_train = y_true[xid_train]
    y_test = y_true[xid_test]
    # sample M cells
    m = int(M/len(f_train))
    print('sampling...')
    Ui_list = []
    for i in range(len(f_train)):
        Xi = get_fcsdata(DIRPATH + f_train[i])
        random.seed(1)
        Ui_list.append(Xi[random.sample(range(Xi.shape[0]), min(m, Xi.shape[0])), :])
    U = np.concatenate(Ui_list)
    print(U.shape)
    # remove non-viable cells, singlets
    if omit_singlets:
        print("removing cells based on viability dye")
        U = U[np.where(U[:, 9] <= np.log(50000))[0], :]
    fsratio = np.exp(U[:, 0] - U[:, 1])
    cutoff = fsratio.mean() + 2 * fsratio.std()
    if omit_doublets:
        print("removing cells based on viability dye, fsc-a/fsc-h ratio")
        U = U[np.where(fsratio <= cutoff)[0], :]
    print(U.shape)
    # remove CD19 < 3 (in log space)
    # update: remove CD19 < 8
    if omit_lowCD19:
        print("removing low CD19")
        print("using 8 as CD19 cutoff")
        U = U[np.where(U[:, 7] > 8)[0], :]
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
    # save
    np.save("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/U.npy", U)
    np.save("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/V.npy", V)
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/embedder.obj", "wb") as f:
        pickle.dump(embedder, f)
    t3 = datetime.datetime.now()
    T3 = time.process_time()
    print("TSTAMP 3 : " + str(t3))
    print("time elapsed: " + str( round((t3-t2).total_seconds(), 5) ))
    print("process time elapsed: " + str(round(T3-T2, 5)))
    return fold_i

# --
t4 = datetime.datetime.now()
T4 = time.process_time()
print("TSTAMP 4 : " + str(t4))

# generate umap for fold 1
gen_UMAP(0)

t5 = datetime.datetime.now()
T5 = time.process_time()
print("TSTAMP 5 : " + str(t5))
print("time elapsed: " + str( round((t5-t4).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T5-T4, 5)))

# generate umap for fold 2
gen_UMAP(1)

t6 = datetime.datetime.now()
T6 = time.process_time()
print("TSTAMP 6 : " + str(t6))
print("time elapsed: " + str( round((t6-t5).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T6-T5, 5)))

# generate umap for fold 3
gen_UMAP(2)

t6 = datetime.datetime.now()
T6 = time.process_time()
print("TSTAMP 6 : " + str(t6))
print("time elapsed: " + str( round((t6-t5).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T6-T5, 5)))

print("time elapsed [TOTAL]: " + str( round((t6-t0).total_seconds(), 5) ))
print("process time elapsed [TOTAL]: " + str(round(T6-T0, 5)))

print('done.')

