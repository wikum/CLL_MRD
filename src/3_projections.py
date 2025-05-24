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

## prepare UMAP  projections

## ============================================================

random.seed(1)

RUN_ID = '0000000' # enter RUN ID from running 2_UMAP.py here

CUR_ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

print(RUN_ID)
print(CUR_ID)

if not os.path.exists('./log'):
    os.makedirs('./log')

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logfilename = "./log/3_projections_" + RUN_ID + '_' + CUR_ID + ".logfile.log"
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

def process_case_umap(f, embedder, cutoff, projection_path, omit_singlets=False, omit_doublets=False, omit_lowCD19=False):
    print('[copy] starting ' + f)
    tf0 = time.process_time()
    outfile = projection_path + f + ".npy"
    if not os.path.exists(outfile):
        Xi = get_fcsdata(DIRPATH + f, omit_singlets=False, omit_doublets=False)
        if omit_singlets:
            Xi = Xi[np.where(Xi[:, 9] <= np.log(50000))[0], :]
        fsratio_i = np.exp(Xi[:, 0] - Xi[:, 1])
        if omit_doublets:
            Xi = Xi[np.where(fsratio_i <= cutoff)[0], :]
        if omit_lowCD19:
            Xi = Xi[np.where(Xi[:, 7] > 8)[0], :]
        Vi = embedder.transform(Xi)
        np.save(outfile, Vi)
    tf1 = time.process_time()
    #print('completed ' + f)
    return tf1-tf0

## generate projections
def gen_projections(fold_i, parallel=True, randomize=False, rev=False, omit_singlets=False, omit_doublets=False, omit_lowCD19=False):
    t2 = datetime.datetime.now()
    T2 = time.process_time()
    print('---------------- FOLD ' + str(fold_i) + ' [projections] ----------------')
    print("TSTAMP 2 : " + str(t2))
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/dat.obj", "rb") as f:
        xid_train, xid_test, M = pickle.load(f)
    f_train = fcs_files[xid_train]
    f_test = fcs_files[xid_test]
    y_train = y_true[xid_train]
    y_test = y_true[xid_test]
    # sample M cells
    m = int(M/len(f_train))
    # load
    U = np.load("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/U.npy")
    V = np.load("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/V.npy")
    print(V.shape)
    with open("./obj/2/" + RUN_ID + "/" + str(fold_i) + "/embedder.obj", "rb") as f:
        embedder = pickle.load(f)
    fsratio = np.exp(U[:, 0] - U[:, 1])
    cutoff = fsratio.mean() + 2 * fsratio.std()
    print("projections..")
    if omit_singlets:
        print("[cell omission based on viability dye, fsc-a/fsc-h ratio from U will be applied]")
    if omit_doublets:
        print("[cell omission based on fsc-a/fsc-h ratio from U will be applied]")
    if omit_lowCD19:
        print("[cell omission based on low CD19 will be applied](cutoff=8)")
    projection_path = "./obj/2/" + RUN_ID + "/" + str(fold_i) + "/projections/"
    done_files = os.listdir(projection_path)
    done_files_fcs = [re.sub(".npy", "", x) for x in done_files]
    print(str(len(done_files_fcs)) + " files done")
    left_files_fcs = list(set(fcs_files) - set(done_files_fcs))
    print(str(len(left_files_fcs)) + " files left")
    if randomize:
        random.shuffle(left_files_fcs)
    elif rev:
        print('[rev]')
        left_files_fcs.reverse()
    t3 = datetime.datetime.now()
    T3 = time.process_time()
    print("TSTAMP 3 : " + str(t3))
    print("time elapsed: " + str( round((t3-t2).total_seconds(), 5) ))
    print("process time elapsed: " + str(round(T3-T2, 5)))
    w_list = []
    if parallel:
        ncpu = multiprocessing.cpu_count()
        print('[par] ' + str(ncpu))
        pool_args = [(f, embedder, cutoff, projection_path, omit_singlets, omit_doublets, omit_lowCD19) for f in left_files_fcs]
        with multiprocessing.Pool(processes=ncpu-1) as pool:
            w_list = pool.starmap(process_case_umap, pool_args)
    else:
        print('[serial]')
        for i in range(len(left_files_fcs)):
            w_list.append(process_case_umap(f=left_files_fcs[i], embedder=embedder, cutoff=cutoff, projection_path=projection_path))
    # done projections
    print('threads completed:' + str(len(w_list)))
    print("process time elapsed within threads: " + str(round(sum(w_list), 5)))
    t4 = datetime.datetime.now()
    T4 = time.process_time()
    print("TSTAMP 4 : " + str(t4))
    print("time elapsed: " + str( round((t4-t3).total_seconds(), 5) ))
    print("process time elapsed: " + str(round(T4-T3, 5)))
    print('done [FOLD]')
    print("time elapsed [FOLD]: " + str( round((t4-t2).total_seconds(), 5) ))
    print("process time elapsed [FOLD]: " + str(round(T4-T2, 5)))
    return fold_i

# --
t1 = datetime.datetime.now()
T1 = time.process_time()
print("TSTAMP 1 : " + str(t1))

gen_projections(fold_i=2, parallel=True, randomize=False, rev=True)

t5 = datetime.datetime.now()
T5 = time.process_time()
print("TSTAMP 5 : " + str(t5))
print("time elapsed: " + str( round((t5-t1).total_seconds(), 5) ))
print("process time elapsed: " + str(round(T5-T1, 5)))

print("time elapsed [TOTAL]: " + str( round((t5-t0).total_seconds(), 5) ))
print("process time elapsed [TOTAL]: " + str(round(T5-T0, 5)))

print('done.')


