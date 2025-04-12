# testset_rank_split.py
# split the testing dataset into simple/medium/hard level

import json
import sympy as sp
import numpy as np
import pandas as pd
import tqdm
import scipy
from collections import defaultdict
import traceback
import json

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parsers import get_parser
import symbolicregression as symbolicregression
from symbolicregression.envs import build_env

np.seterr(all="ignore")

########## UTILITY ##########

def bool_to_int(bool):
    _bool = bool.astype(np.int32)
    _str = "".join(list(map(str, _bool)))
    return int(_str, 2)

def r2_matrix(A, B=None):
    if B is not None:
        A = np.concatenate((A, B), axis=0)
    numer = np.sum((A[:,None,:] - A[None,:,:]) ** 2, axis=2)
    denom = np.sum((A - np.mean(A, axis=1).reshape((-1, 1))) ** 2, axis=1)
    return 1 - numer / denom

def exchange_node(tree):
    dfs_stack = [tree]
    while dfs_stack:
        current_node = dfs_stack.pop(0)
        if len(current_node.children) != 2:
            pass
        elif len(current_node.children[0]) > len(current_node.children[1]) and\
            current_node.value != "sub":
            current_node.children = [
                current_node.children[1],
                current_node.children[0]
            ]
        dfs_stack.extend(current_node.children)


test_rank = False
feynman_rank = True


if test_rank:

    ########## INITIALIZE ##########

    with open("./data/exprs_train.json", "r") as fi:
        exprs_train = json.load(fi)
    with open("./data/exprs_test.json", "r") as fi:
        exprs_test = json.load(fi)

    parser = get_parser()
    params = parser.parse_args()
    env = build_env(params)
    generator = env.generator

    ########## STEP1 ##########

    num_datapoints = 40
    verbose = False

    nums = []
    bs = []
    xs = [np.random.normal(0, 3, size=(num_datapoints, i)) for i in range(1, 11)]
    ys = np.empty(shape=(len(exprs_test), num_datapoints))

    #some xs are close to 0
    for i in range(len(xs)):
        xs[i][abs(xs[i]) < 0.2] *= 2
        xs[i][abs(xs[i]) < 0.05] += 0.05 * np.sign(xs[i][abs(xs[i])<0.05])

    for i,expr in enumerate(exprs_test):
        node, v = generator.infix_to_node(expr)
        exchange_node(node)

        num = len(v)
        x = xs[num-1]
        y = node.val(x).reshape((-1))
        y[abs(y) > 1e50] = np.nan
        n = np.isnan(y)
        b = bool_to_int(n)
        y[np.isnan(y)] = 0

        nums.append(num)
        bs.append(b)
        ys[i] = y.copy()

    nums = np.array(nums)
    bs = np.array(bs)

    ########## STEP2 ##########

    maxi_corr = np.zeros((len(exprs_test)))

    nums_train = []
    bs_train = []
    ys_train = np.empty(shape=(len(exprs_train), num_datapoints))

    for i,expr in enumerate(exprs_train):
        node, v = generator.infix_to_node(expr)
        exchange_node(node)

        num = len(v)
        x = xs[num-1]
        y = node.val(x).reshape((-1))
        y[abs(y) > 1e50] = np.nan
        n = np.isnan(y)
        b = bool_to_int(n)
        y[np.isnan(y)] = 0

        nums_train.append(num)
        bs_train.append(b)
        ys_train[i] = y.copy()

    nums_train = np.array(nums_train)
    bs_train = np.array(bs_train)

    corr_types = set(
        [
            str(n1)+"_"+str(n2) for n1,n2 in zip(nums, bs)
        ] + [
            str(n1)+"_"+str(n2) for n1,n2 in zip(nums_train, bs_train)
        ]
    )

    s = ""
    for corr_type in corr_types:
        n1, n2 = corr_type.split("_")

        _idx1 = nums_train == int(n1)
        _idx2 = bs_train == int(n2)
        idx_train = np.logical_and(_idx1, _idx2)
        train_feature = ys_train[idx_train]

        _idx1 = nums == int(n1)
        _idx2 = bs == int(n2)
        idx_test = np.logical_and(_idx1, _idx2)
        test_feature = ys[idx_test]

        if len(train_feature) == 0 or len(test_feature) == 0:
            continue
        if n2 == str(int("1"*num_datapoints, 2)):
            continue

        flag = False
        res_train = train_feature.shape[0]
        eval_train = 0
        while not flag:
            _train_feature = train_feature[eval_train:eval_train+min(10000, res_train)]

            _corrs = r2_matrix(_train_feature, test_feature)
            _corrs_log = r2_matrix(
                np.log(np.abs(_train_feature)+1e-100), np.log(np.abs(test_feature)+1e-100)
            )

            _corrs_outer = _corrs[len(_train_feature):,:-len(test_feature)]
            _corrs_log_outer = _corrs_log[len(_train_feature):,:-len(test_feature)]

            _corrs_outer = np.max(_corrs_outer, axis=-1)
            _corrs_log_outer = np.max(_corrs_log_outer, axis=-1)

            _corrs_outer = np.minimum(_corrs_outer, _corrs_log_outer)

            _corrs_maxi = np.zeros((len(exprs_test)))
            _corrs_maxi[idx_test] = _corrs_outer

            maxi_corr = np.maximum(maxi_corr, _corrs_maxi)

                
            eval_train += min(10000, res_train)
            res_train -= 10000
            if res_train <= 0:
                flag = True

    # about 800 formulas has corrs > 0.95
    # about 300 formulas has 0.95 >= corrs > 0.8
    # about 200 formulas has corrs <= 0.8

    exprs_simple = []
    exprs_midium = []
    exprs_hard = []

    for c, e in zip(maxi_corr, exprs_test):
        if c > 0.95:
            exprs_simple.append(e)
        elif c > 0.8:
            exprs_midium.append(e)
        else:
            exprs_hard.append(e)

    exprs_test_ranked = exprs_simple[:200] + exprs_midium[:200] + exprs_hard[:200]

    assert len(exprs_test_ranked) == 600

    with open("./data/exprs_test_ranked.json", "w") as fi:
        json.dump(exprs_test_ranked, fi)


if feynman_rank:

    ########## INITIALIZE ##########

    with open("./data/exprs_train.json", "r") as fi:
        exprs_train = json.load(fi)
    exprs_test = pd.read_excel("./data/FeynmanEquations.xlsx")
    exprs_test = list(exprs_test["Formula"])

    for i in range(len(exprs_test)):
        exprs_test[i] = exprs_test[i].replace("COS", "cos")

    parser = get_parser()
    params = parser.parse_args()
    env = build_env(params)
    generator = env.generator

    ########## STEP1 ##########

    num_datapoints = 40
    verbose = False

    nums = []
    bs = []
    xs = [np.random.normal(0, 3, size=(num_datapoints, i)) for i in range(1, 11)]
    ys = np.empty(shape=(len(exprs_test), num_datapoints))

    #some xs are close to 0
    for i in range(len(xs)):
        xs[i][abs(xs[i]) < 0.2] *= 2
        xs[i][abs(xs[i]) < 0.05] += 0.05 * np.sign(xs[i][abs(xs[i])<0.05])

    for i,expr in enumerate(exprs_test):
        node, v = generator.infix_to_node(expr)
        exchange_node(node)

        num = len(v)
        x = xs[num-1]
        y = node.val(x).reshape((-1))
        y[abs(y) > 1e50] = np.nan
        n = np.isnan(y)
        b = bool_to_int(n)
        y[np.isnan(y)] = 0

        nums.append(num)
        bs.append(b)
        ys[i] = y.copy()

    nums = np.array(nums)
    bs = np.array(bs)

    ########## STEP2 ##########

    maxi_corr = np.zeros((len(exprs_test)))

    nums_train = []
    bs_train = []
    ys_train = np.empty(shape=(len(exprs_train), num_datapoints))

    for i,expr in enumerate(exprs_train):
        node, v = generator.infix_to_node(expr)
        exchange_node(node)

        num = len(v)
        x = xs[num-1]
        y = node.val(x).reshape((-1))
        y[abs(y) > 1e50] = np.nan
        n = np.isnan(y)
        b = bool_to_int(n)
        y[np.isnan(y)] = 0

        nums_train.append(num)
        bs_train.append(b)
        ys_train[i] = y.copy()

    nums_train = np.array(nums_train)
    bs_train = np.array(bs_train)

    corr_types = set(
        [
            str(n1)+"_"+str(n2) for n1,n2 in zip(nums, bs)
        ] + [
            str(n1)+"_"+str(n2) for n1,n2 in zip(nums_train, bs_train)
        ]
    )

    s = ""
    for corr_type in corr_types:
        n1, n2 = corr_type.split("_")

        _idx1 = nums_train == int(n1)
        _idx2 = bs_train == int(n2)
        idx_train = np.logical_and(_idx1, _idx2)
        train_feature = ys_train[idx_train]

        _idx1 = nums == int(n1)
        _idx2 = bs == int(n2)
        idx_test = np.logical_and(_idx1, _idx2)
        test_feature = ys[idx_test]

        if len(train_feature) == 0 or len(test_feature) == 0:
            continue
        if n2 == str(int("1"*num_datapoints, 2)):
            continue

        flag = False
        res_train = train_feature.shape[0]
        eval_train = 0
        while not flag:
            _train_feature = train_feature[eval_train:eval_train+min(10000, res_train)]

            _corrs = r2_matrix(_train_feature, test_feature)
            _corrs_log = r2_matrix(
                np.log(np.abs(_train_feature)+1e-100), np.log(np.abs(test_feature)+1e-100)
            )

            _corrs_outer = _corrs[len(_train_feature):,:-len(test_feature)]
            _corrs_log_outer = _corrs_log[len(_train_feature):,:-len(test_feature)]

            _corrs_outer = np.max(_corrs_outer, axis=-1)
            _corrs_log_outer = np.max(_corrs_log_outer, axis=-1)

            _corrs_outer = np.minimum(_corrs_outer, _corrs_log_outer)

            _corrs_maxi = np.zeros((len(exprs_test)))
            _corrs_maxi[idx_test] = _corrs_outer

            maxi_corr = np.maximum(maxi_corr, _corrs_maxi)
 
            eval_train += min(10000, res_train)
            res_train -= 10000
            if res_train <= 0:
                flag = True


    print(maxi_corr)

    for i, (c, e) in enumerate(zip(maxi_corr, exprs_test)):
        print(i, np.round(c, 3), e)    