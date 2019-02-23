# author bao
import pickle
import numpy as np  # We recommend to use numpy arrays
import gc
import os
from os.path import isfile
import random

os.system("pip3 install lightgbm==2.2.2")
import lightgbm as lgb

os.system('pip3 install pandas==0.23.4')
import pandas as pd
import time
import hashlib
import multiprocessing as mp

import math
import collections
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample
from sklearn import preprocessing
from sklearn import metrics
from scipy import sparse
import json
import re
import operator
from collections import deque
from multiprocessing import Pool


#################### utils for system #############################

# seed everything
def seed_everything(seed=2016):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(2016)


def process_in_parallel(function, list_):
    with Pool(2) as p:
        tmp = p.map(function, list_)
    return tmp


def timmer(func):
    def warpper(*args, **kwargs):
        strat_time = time.time()
        r = func(*args, **kwargs)
        stop_time = time.time()
        print("[Success] Info: function: {}() done".format(func.__name__))
        print("the func run time is %.2fs" % (stop_time - strat_time))
        return r

    return warpper


def universe_mp_generator(gen_func, feat_list):
    pool = mp.Pool(7)
    result = [pool.apply_async(gen_func, feats) for feats in feat_list]
    pool.close()
    pool.join()
    return [aresult.get() for aresult in result]


def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except Exception as err:
                print(err)
                print(l.head())
    return result


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def count_helper(df, i):
    df['count_' + i] = df.groupby(i)[i].transform('count') / df.shape[0]
    return df[['count_' + i]]


def bi_count_helper(df, i, j):
    df['bicount_{}_{}'.format(i, j)] = df.groupby([i, j])[i].transform('count') / df.shape[0]
    return df[['bicount_{}_{}'.format(i, j)]]


def ke_feature_helper(df, time_col, col):
    df['ke_cnt_' + col] = df.groupby(col)[time_col].rank(ascending=False)
    df2 = df[[col, 'ke_cnt_' + col, time_col]].copy()
    df2['ke_cnt_' + col] = df2['ke_cnt_' + col] - 1
    df3 = pd.merge(df, df2, on=[col, 'ke_cnt_' + col], how='left')
    df['ke_time_' + col] = df3[time_col + '_x'] - df3[time_col + '_y']
    del df2, df3
    gc.collect()
    return df[['ke_cnt_' + col, 'ke_time_' + col]]


def target_encoding_helper(df, i, to_merge_df, last_prior):
    f_name = "target_{}".format(i)
    # to_merge_df = self.last_target_encoding[i]
    to_merge_df = to_merge_df.rename({'label': f_name}, axis=1)
    to_merge_df[f_name] = add_noise(to_merge_df[f_name], noise_level=0.0001)
    df = df.merge(to_merge_df, on=[i], how='left').fillna(last_prior)
    return df[[f_name]]


#################### Model define #############################
class Model():
    def __init__(self, datainfo, timeinfo):
        # dataset batch for time long summary
        self.batch_idx = 1
        # time feature size
        self.ntime = datainfo['loaded_feat_types'][0]
        # num feature size
        self.nnum = datainfo['loaded_feat_types'][1]
        # cat feature size
        self.ncat = datainfo['loaded_feat_types'][2]
        # multy value feature size
        self.nmv = datainfo['loaded_feat_types'][3]
        # time + num + cat + mv or dense len(x.split())
        self.feat_num = datainfo['feat_num']
        ## AA, B, C, D, E
        self.data_name = datainfo['name']
        self.time_budget = datainfo['time_budget']
        # when this model start to trian
        self.start_time = timeinfo[1]
        # if this feature will droped

        # self.current_batch_fit = None
        self.is_autotune = True
        self.is_use_cast = True
        self.is_dynamic = True
        self.is_use_kfold = True
        # select config
        self.is_lgb_select = False
        self.is_kflod_lgb_select = True
        self.is_random_lgb_select = False
        self.is_stat_select = True
        self.is_select_every_step = False
        self.is_drop_feature = True

        # feature generation
        self.is_use_counts_f = True
        self.is_use_bicounts_f = True
        self.is_binum_cross = False
        self.is_rank = True
        self.is_target = True
        self.is_bi_target = False
        self.target_off = 3
        self.is_time = True
        self.is_ti=False
        self.is_num_rank=False

        # class matience
        self.drop_name = []
        self.slide_window = 1
        self.window = None
        self.fea_importance_now = None
        self.fea_importance_batch = None
        self.best_iter = None
        self.last_target_encoding = {}
        self.last_prior = 0
        self.feature_nunique = {}
        self.is_easy_overfit = False

        # featyre seection
        self.is_use_adversial = True
        self.count_rate = 0.3
        self.target_encode_rate = 0.3
        self.train_test_rate = 0.2
        self.imbalanced_rate = 0.01
        self.random_state = 2016
        self.total_shrink = 1600000
        self.batch_shrink = 250000
        self.sample_base = 1.00001

        self.time_col = None
        self.params_lgb = {
            "objective": "binary",
            "metric": "binary_logloss,auc",
            'verbose': -1,
            "seed": self.random_state,
            'two_round': False,
            'num_threads': 8,
            'num_leaves': 20,
            'learning_rate': 0.03,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'min_sum_hessian_in_leaf': 5,
            'lambda_l1': 0.5, 'lambda_l2': 0.5,
            'max_depth': -1,
            # 'is_unbalance': 'false',
            'two_round': False,
            # 'boost_from_average': False,
        }

        self.featureMap={}

    def fit(self, F, y, datainfo, timeinfo):
        if self.batch_idx == 1:
            df = self.get_df_from_F(F, y)
            df = self.feature_preprocess(df, True)
            self.gen_config(self.balance(df))
            # balance 和 feature的关系
            if self.is_lgb_select:
                fea_imp = self.lgb_select(df)
                self.fea_importance_now = fea_imp
                print(fea_imp.head(10))
            if self.is_kflod_lgb_select:
                fea_imp = self.kflod_lgb_select(df)
                self.fea_importance_now = fea_imp
                print(fea_imp.head(10))
            if self.is_random_lgb_select:
                fea_imp = self.random_lgb_select(df)
                self.fea_importance_now = fea_imp
                print(fea_imp.head(10))
            df = self.feature_extraction(df, True)
        else:
            df = self.pop_add_label(y)

        if self.is_target:
            self.prepare_target_encoding(df)
            self.last_prior = y.mean()

        self.start_batch = time.time()
        if self.batch_idx == 1:
            pass
            # upate config after balance.
            # self.gen_config(df)
        self.window.append(df)
        return

    def predict(self, F, datainfo, timeinfo):
        current_test_df = self.get_df_from_F(F, None)
        current_test_df = self.feature_preprocess(current_test_df, True)
        fearure_extract = self.feature_extraction(current_test_df)
        slide_window_train = self.load_slide_window()
        self.window.append(fearure_extract)
        slide_window_train = self.sample(slide_window_train)
        fearure_extract = fearure_extract.append(slide_window_train)
        print("ALL DATA SIZE:", fearure_extract.shape)
        # self.best_iter = None
        if self.is_autotune:
            # if self.batch_idx == 1 or slide_window_train.shape[0] < 100000:
            if self.batch_idx == 1:  # or slide_window_train.shape[0] < 100000:
                start_tune = time.time()
                print("Begin Bayes")
                print("Params update: ")
                feature_name = [i for i in slide_window_train.columns if i not in self.drop_name + ['label']]
                # slide_window_train[feature_name + ['label']].to_csv("train_{}.csv".format(self.batch_idx) ,index = False)
                self.params, self.best_iter = self.bayes_parameter_opt_lgb(slide_window_train[feature_name],
                                                                           slide_window_train['label'])
                # self.params, self.best_iter  = self.hyperband_lgb(slide_window_train[feature_name], slide_window_train['label'])
                self.params_lgb.update(self.params)
                print(self.params_lgb, self.best_iter)
                self.start_batch += time.time() - start_tune

        del slide_window_train, current_test_df
        gc.collect()
        if self.is_use_adversial:
            self.adversial_validation(fearure_extract)
        if self.batch_idx == 1 and self.is_use_kfold:
            y = self.lgb_kfold_train(fearure_extract)
        else:
            y = self.lgb_mode_preict(fearure_extract)
        self.batch_time = time.time() - self.start_batch
        self.batch_idx += 1
        return y

    #################### utils for preprocess #############################

    def get_hash_func(self, _series):
        nunique = _series.nunique()
        if nunique < 8:
            return lambda x: hash(x) % 255
        elif nunique < 256:
            return lambda x: hash(x) % 65535
        else:
            return lambda x: hash(x)

    def get_hash_func_numpy(self, _array, f_name='cat_n'):
        if f_name in self.feature_nunique.keys():
            nunique = self.feature_nunique[f_name]
        else:
            nunique = np.size(np.unique(_array.astype(str)))
            self.feature_nunique[f_name] = nunique
        if nunique <= 1:
            return None
        if nunique < 8:
            return np.vectorize(lambda x: hash(x) % 255)
        elif nunique < 256:
            return np.vectorize(lambda x: hash(x) % 65535)
        else:
            return np.vectorize(lambda x: hash(x))

    @timmer
    def get_df_from_F(self, F, y):
        df = pd.DataFrame()
        if self.ntime > 0:
            for i in range(self.ntime):
                col_name = "time_{}".format(str(i))
                # vfunc = np.vectorize(lambda x: (x // 60) % (24 * 60))
                vfunc = np.vectorize(lambda x: x)
                if col_name in self.drop_name:
                    continue
                if self.batch_idx == 1:
                    nunique = np.size(np.unique(F['numerical'][:, i]))
                    if nunique <= 1:
                        self.drop_name.append(col_name)
                        print('SKIP', col_name)
                        continue
                df[col_name] = vfunc(F['numerical'][:, i])
                # 找出时间周期差在两个周期里面的两天之内的自然时间列。leakage
                if self.batch_idx == 1 and self.time_col is None and (len(df[df[col_name].isnull()]) == 0) and \
                        max(F['numerical'][:, i]) - min(F['numerical'][:, i]) <= 24 * 3600 * 2:
                    self.time_col = col_name
                    print("natural time coll")
                    print(self.time_col)
        if self.nnum > 0:
            for i in range(self.nnum):
                col_name = "num_{}".format(str(i))
                if col_name in self.drop_name:
                    continue
                if self.batch_idx == 1:
                    nunique = np.size(np.unique(F['numerical'][:, i + self.ntime]))
                    if nunique <= 1:
                        self.drop_name.append(col_name)
                        continue
                df[col_name] = F['numerical'][:, i + self.ntime]
        if self.ncat > 0:
            for i in range(self.ncat):
                col_name = "cat_{}".format(str(i))
                if col_name in self.drop_name:
                    continue
                vfunc = self.get_hash_func_numpy(F['CAT'].values[:, i], col_name)
                if vfunc is None:
                    print('SKIP', col_name)
                    self.drop_name.append(col_name)
                    continue
                df[col_name] = vfunc(F['CAT'].values[:, i])
        if self.nmv > 0:
            for i in range(self.nmv):
                col_name = "mv_{}".format(str(i))
                df[col_name] = F['MV'].values[:, i]
        if y is not None:
            df['label'] = y
            print("label_ratio:", df['label'].mean())
        return df

    #################### utils for data & feature ######################
    @timmer
    def pop_add_label(self, label):
        df = self.window.pop()
        df['label'] = label
        return df

    @timmer
    def load_slide_window(self):
        df = None
        for i in self.window:
            if df is None:
                df = i
            else:
                df = df.append(i).reset_index(drop=True)
        print("TRAIN DATA SIZE", df.shape)
        return df

    @timmer
    def stat_selection(self, df):
        print("shape before stat selection:", df.shape)
        stat_df = self.mete_stat(df)
        no_use = self.navie_select(stat_df)
        self.drop_name.extend(no_use)
        df.drop(no_use, inplace=True, axis=1)
        print("ATTENTION: drop", no_use)
        print("shape after stat selection:", df.shape)
        return df

    @timmer
    def feature_preprocess(self, df_train, is_train):
        # It is not reasonable. I want to check if it chane with time?
        df_train = df_train[[i for i in df_train.columns if i not in self.drop_name]]
        if self.batch_idx == 1 and self.is_stat_select and is_train:
            # only select in first batch train phase
            df_train = self.stat_selection(df_train)
        # df_train = self.cat_encode(df_train)
        df_train = self.mv_encode(df_train)
        if self.is_use_cast:
            df_train = self.reduce_mem_usage(df_train)
        return df_train

    @timmer
    def feature_extraction(self, df_train, is_train=False):
        print('test t')
        for i in range(self.ntime):
            if 'time_'+str(i) in self.drop_name:
                continue
            tst=df_train['time_'+str(i)].values
            tst1=tst[1:]-tst[:tst.shape[0]-1]
            print('time_'+str(i),tst1[tst1>0].shape[0],tst1[tst1==0].shape[0],tst1[tst1<0].shape[0])
        print("shape before feature_extraction", df_train.shape)
        if self.is_time:
            df_train = self.bi_time_count(df_train)
        if self.is_use_counts_f:
            df_train = self.count_encode(df_train)
        if self.is_use_bicounts_f:
            df_train = self.bi_count_encode(df_train)
        if self.is_binum_cross:
            df_train = self.bi_num_cross(df_train)
        if self.is_rank:
            df_train = self.group_by_encoding(df_train)
        if self.is_target:
            df_train = self.calc_target_encoding(df_train, is_train)
        if self.is_bi_target:
            df_train = self.bi_target_encoding(df_train)
        if self.is_ti:
            df_train = self.ti_encode(df_train)
        if self.is_num_rank:
            df_train = self.num_group_by_encoding(df_train)
        print("shape after feature_extraction", df_train.shape)
        return df_train

    def count_encode(self, df):
        num_df0 = df.shape[1]
        if self.ntime == 0:
            count_rate = 1
        else:
            count_rate = self.count_rate
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', count_rate)
        count_columns = []
        if self.is_easy_overfit:
            import_cat_col = import_cat_col[int(self.delete_overfit_rate * len(import_cat_col)):]
        for i in import_cat_col:
            if i in self.drop_name:
                continue
            count_columns.append([df[[i]], i])
        count_result = universe_mp_generator(count_helper, count_columns)
        del count_columns
        gc.collect()
        count_result.append(df)
        df = concat(count_result)
        num_df1 = df.shape[1]
        print("number of count feature is {}".format(num_df1 - num_df0))
        #    #df['count_' + i] = df[[i]].groupby(i)[i].transform('count') / df.shape[0]
        return df
    def count_encode_new(self, df):
        num_df0 = df.shape[1]
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', 1)
        for i in import_cat_col:
            if i in self.drop_name:
                continue
            if i not in self.featureMap:
                self.featureMap[i]={}
            curr_featureMap = dict(pd.value_counts(df[i]))
            self.featureMap[i] = dict(Counter(self.featureMap[i]) + Counter(curr_featureMap))
            keys = self.featureMap[i].keys()
            vals = np.array(list(self.featureMap[i].values())).astype(float)
            freqMap = dict(zip(keys, vals))
            freq_encoded_col = np.vectorize(freqMap.get)(df[i].values)
            df['count_'+i]=freq_encoded_col
        num_df1 = df.shape[1]
        print("number of count feature is {}".format(num_df1 - num_df0))
        #    #df['count_' + i] = df[[i]].groupby(i)[i].transform('count') / df.shape[0]
        return df

    def target_encode(self, trn_series=None, target=None, min_samples_leaf=20, smoothing=10, noise_level=0.001):
        temp = pd.concat([trn_series, target], axis=1)
        averages = temp.groupby(by=trn_series.name, as_index=False)[target.name].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        prior = target.mean()
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        return averages.reset_index()

    @timmer
    def prepare_target_encoding(self, df):
        if self.batch_idx == 1:
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', 0.85)
        else:
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.target_encode_rate)
        for i in import_cat_col:  # + import_mv_col:
            if i in self.drop_name:
                continue
            # if df[i].nunique() < self.target_off:
            #    continue
            self.last_target_encoding[i] = self.target_encode(trn_series=df[i], target=df['label'])
        return

    @timmer
    def calc_target_encoding(self, df, is_train):
        num_df0 = df.shape[1]
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.target_encode_rate)
        if self.is_easy_overfit:
            import_cat_col = import_cat_col[int(self.delete_overfit_rate * len(import_cat_col)):]
        target_encode_columns = []
        for i in import_cat_col:  # + import_mv_col:
            if i in self.drop_name:
                continue
            # if df[i].nunique() < self.target_off:
            #    continue
            if self.batch_idx == 1 and is_train:
                pass
            else:
                target_encode_columns.append([df[[i]], i, self.last_target_encoding[i], self.last_prior])

        target_encode_result = universe_mp_generator(target_encoding_helper, target_encode_columns)
        del target_encode_columns
        gc.collect()
        target_encode_result.append(df)
        df = concat(target_encode_result)
        num_df1 = df.shape[1]
        print("number of target_encoding feature is {}".format(num_df1 - num_df0))
        return df

    def bi_target_encoding(self, df):
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.count_rate)
        top = min(9, round((self.ncat + self.nmv) + 1))
        count = 0
        for x in range(len(import_cat_col)):
            for y in range(x + 1, len(import_cat_col)):
                if count == top:
                    break
                i = import_cat_col[x]
                j = import_cat_col[y]
                f_name = "bi_target_{}_{}".format(i, j)
                merge_name = 't_{}_{}'.format(i, j)
                temp = df[[i, j]]
                temp[merge_name] = temp[i].astype(str) + '_' + temp[j].astype(str)
                if self.batch_idx == 1:
                    pass
                else:
                    to_merge_df = self.last_target_encoding[merge_name]
                    to_merge_df = to_merge_df.rename({'label': f_name}, axis=1)
                    to_merge_df[f_name] = add_noise(to_merge_df[f_name], noise_level=0.0001)
                    temp = temp.merge(to_merge_df, on=[merge_name], how='left').fillna(self.last_prior)
                    df.loc[:, merge_name] = temp[f_name]
                # if 'label' in df.columns:
                #     self.last_target_encoding[merge_name] = self.target_encode(trn_series = temp[merge_name], target = df['label'])
                count += 1
                del temp
                gc.collect()
        return df

    def group_by_encoding(self, df):
        if self.time_col is not None:
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.gp_rate)
            # import_mv_col = self.get_imp_fea_seperate('MV', self.count_rate)
            for col in import_cat_col:
                if col in self.drop_name:
                    continue
                df['ke_cnt_' + col] = df.groupby(col)[self.time_col].rank(ascending=False,method='dense')
                df2 = df[[col, 'ke_cnt_' + col, self.time_col]].copy()
                df2=df2.drop_duplicates()
                df2['ke_cnt_' + col] = df2['ke_cnt_' + col] - 1
                df3 = pd.merge(df, df2, on=[col, 'ke_cnt_' + col], how='left')
                df['ke_time_' + col] = df3[self.time_col + '_x'] - df3[self.time_col + '_y']
                del df2, df3
                gc.collect()
        return df
    def num_group_by_encoding(self, df):
        if self.ntime == 0:
            count_rate = 1
        else:
            count_rate = self.count_rate
        import_num_col = self.get_imp_fea_seperate('NUM', count_rate)[:4]
        for i in import_num_col:
            if i in self.drop_name:
                continue
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.gp_rate)
            # import_mv_col = self.get_imp_fea_seperate('MV', self.count_rate)
            import_cat_col=import_cat_col[:5]
            for col in import_cat_col:
                if col in self.drop_name:
                    continue
                df['num_ke_cnt_' + i+'_'+col] = df.groupby(col)[i].rank(ascending=False)
                df2 = df[[col, 'num_ke_cnt_' +i+'_'+ col, i]].copy()
                #df2=df2.drop_duplicates()
                df2['num_ke_cnt_' + i+'_'+col] = df2['num_ke_cnt_' + i+'_'+col] - 1
                df3 = pd.merge(df, df2, on=[col, 'num_ke_cnt_' + i+'_'+col], how='left')
                df['num_ke_time_' + i+'_'+col] = df3[i + '_x'] - df3[i + '_y']
                del df2, df3
                gc.collect()
        return df

    def group_by_encoding_mp(self, df):
        if self.time_col is not None:
            import_cat_col = self.get_imp_fea_seperate('CAT+MV', self.gp_rate)
            # import_mv_col = self.get_imp_fea_seperate('MV', self.count_rate)
            gp_columns = []
            for col in import_cat_col:
                if col in self.drop_name:
                    continue
                gp_columns.append([df[[self.time_col, col]], self.time_col, col])
            gp_result = universe_mp_generator(ke_feature_helper, gp_columns)
            del gp_columns
            gc.collect()
            gp_result.append(df)
            df = concat(gp_result)
        return df

    def bi_count_encode(self, df):
        num_df0 = df.shape[1]
        count_rate = self.bi_count_rate
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', count_rate)
        if self.is_easy_overfit:
            import_cat_col = import_cat_col[int(self.delete_overfit_rate * len(import_cat_col)):]
        top = min(10, round((self.ncat + self.nmv) + 1))
        # top =  min(10, round((self.ncat + self.nmv) + 1))
        count = 0
        # use this for multiy processing
        # dangerous
        bi_count_columns = []
        first = max(3, int(len(import_cat_col) / 4 * 3))
        second = max(2, int(len(import_cat_col) / 3 * 2))
        for x in range(first):
            for y in range(x + 1, second):
                if count == top:
                    break
                if x >= len(import_cat_col) or y >= len(import_cat_col):
                    continue
                i = import_cat_col[x]
                j = import_cat_col[y]
                if i in self.drop_name or j in self.drop_name:
                    continue
                bi_count_columns.append([df[[i, j]], i, j])
                # df['bicount_{}_{}'.format(i,j)] = df[[i,j]].groupby([i,j])[i].transform('count') / df.shape[0]
                count += 1
        bi_count_results = universe_mp_generator(bi_count_helper, bi_count_columns)
        del bi_count_columns
        gc.collect()
        bi_count_results.append(df)
        df = concat(bi_count_results)
        num_df1 = df.shape[1]
        print("number of bi_count feature is {}".format(num_df1 - num_df0))
        return df

    def bi_num_cross(self, df):
        import_cat_col = self.get_imp_fea_seperate('NUM', self.numcross_rate)
        count = 0
        top = min(10, self.nnum + 2)
        for x in range(len(import_cat_col)):
            for y in range(x + 1, len(import_cat_col)):
                if count == top:
                    break
                i = import_cat_col[x]
                j = import_cat_col[y]
                if i in self.drop_name or j in self.drop_name:
                    continue
                df['bicount_{}_{}_add'.format(i, j)] = df[i] + df[j]
                df['bicount_{}_{}_sub'.format(i, j)] = df[i] - df[j]
                df['bicount_{}_{}_div'.format(i, j)] = df[i] / (df[j] + 0.000001)
                count += 1
        return df

    def bi_time_cross(self, df):
        num_df0 = df.shape[1]
        if self.ntime <= 0:
            pass
        else:
            import_time_col = self.get_imp_fea_seperate('TIME', 1)
            for i in range(len(import_time_col)):
                for j in range(i + 1, len(import_time_col)):
                    if import_time_col[i] in self.drop_name or import_time_col[j] in self.drop_name:
                        continue
                    df['cross_{}_{}'.format(import_time_col[i], import_time_col[j])] = df[import_time_col[i]] - df[
                        import_time_col[j]]
        num_df1 = df.shape[1]
        print("number of bi_time_cross feature is {}".format(num_df1 - num_df0))
        return df

    def bi_time_count(self, df):
        if self.ntime <= 0:
            return df
        num_df0 = df.shape[1]
        count_rate = self.bi_count_rate
        import_cat_col = self.get_imp_fea_seperate('CAT+MV', count_rate)
        if self.is_easy_overfit:
            import_cat_col = import_cat_col[int(self.delete_overfit_rate * len(import_cat_col)):]
        top = min(12, round((self.ncat + self.nmv) + 2))
        count = 0
        bi_count_columns = []
        second = max(3, int(len(import_cat_col) / 3 * 2))
        df[self.time_col] = df[self.time_col].apply(lambda x: (x // 60) % (24 * 60))
        for x in range(0, second):
            if count == top:
                break
            if x >= len(import_cat_col):
                continue
            i = import_cat_col[x]
            if i in self.drop_name or self.time_col in self.drop_name:
                continue
            bi_count_columns.append([df[[i, self.time_col]], i, self.time_col])
            count += 1
        bi_count_results = universe_mp_generator(bi_count_helper, bi_count_columns)
        del bi_count_columns
        gc.collect()
        bi_count_results.append(df)
        df = concat(bi_count_results)
        num_df1 = df.shape[1]
        print("number of bi_time_count feature is {}".format(num_df1 - num_df0))
        return df

    def ti_encode(self, df):
        num_df0 = df.shape[1]
        year=np.empty(df.shape[0])
        month=np.empty(df.shape[0])
        dayofmonth=np.empty(df.shape[0])
        hour=np.empty(df.shape[0])
        minute=np.empty(df.shape[0])
        second=np.empty(df.shape[0])
        dayofweek=np.empty(df.shape[0])
        for i in range(self.ntime):
            if 'time_'+str(i) in self.drop_name:
                continue
            for j in range(i+1,self.ntime):
                if 'time_'+str(j) in self.drop_name:
                    continue
                if len(np.nonzero(df['time_'+str(i)]))>0 and len(np.nonzero(df['time_'+str(j)]))>0:
                    print('AutoGBT[GenericStreamPreprocessor]:datediff from nonzero cols:',i,j)
                    df['ti_'+str(i)+'-'+str(j)]=df['time_'+str(i)]-df['time_'+str(j)]
            timestamp = np.nan_to_num(df['time_'+str(i)].values).astype(int)
            for j in range(timestamp.shape[0]):
                dates=time.localtime(timestamp[j])
                year[j] = dates[0]
                month[j] = dates[1]
                dayofmonth[j] = dates[2]
                hour[j] = dates[3]
                minute[j] = dates[4]
                second[j] = dates[5]
                dayofweek[j] = dates[6]
            #dates = time.localtime(np.nan_to_num(df['time_'+str(i)].values).astype(int))

            df['ti_year_'+str(i)]=year
            df['ti_month_'+str(i)]=month
            df['ti_dayofmonth_'+str(i)]=dayofmonth
            df['ti_hour_'+str(i)]=hour
            df['ti_minute_'+str(i)]=minute
            df['ti_second_'+str(i)]=second
            df['ti_dayofweek_'+str(i)]=dayofweek
        num_df1 = df.shape[1]
        print("number of ti feature is {}".format(num_df1 - num_df0))
        #    #df['count_' + i] = df[[i]].groupby(i)[i].transform('count') / df.shape[0]
        return df

    def cat_encode(self, df):
        for i in range(self.ncat):
            f_name = "cat_{}".format(i)
            if f_name in self.drop_name:
                continue
            hash_func = self.get_hash_func(df[f_name])
            df.loc[:, f_name] = df[f_name].apply(hash_func)
        return df

    def mv_encode(self, df):
        for i in range(self.nmv):
            f_name = "mv_{}".format(i)
            if f_name in self.drop_name:
                continue
            hash_func = self.get_hash_func(df[f_name])
            df.loc[:, f_name] = df[f_name].apply(hash_func)
        return df

    #################### utils for select ######################

    def navie_select(self, df):
        drop_name = list(df[(df['Unique_values'] == 0) | (df['Percentage of missing values'] > 99.999) | (
                    df['Percentage of values in the biggest category'] > 99.999)]['Feature'])
        drop_name = [i for i in drop_name if i not in ['label']]
        return drop_name

    @timmer
    def random_lgb_select(self, df):
        num_leaves = [618, 16, 382]
        max_depth = [10, -1, 8]
        boosting = ['rf', 'gbdt', 'rf']
        feature_importance_df = pd.DataFrame()
        for n_l, m_d, b_t in zip(num_leaves, max_depth, boosting):
            lgb_params = {
                "objective": "binary",
                "metric": "auc",
                'verbose': -1,
                "seed": self.random_state,
                'num_threads': 8,
                'boosting': b_t,
                'num_leaves': n_l,
                'max_depth': m_d,
                'bagging_freq': 1,
                'bagging_fraction': 0.8,
                'feature_fraction': max(math.log(df.shape[1], 2) / df.shape[1], 0.1),
            }
            feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]
            train_x = df[feature_name][df['label'].isnull() == False]
            train_y = df[['label']][df['label'].isnull() == False]
            clf = lgb.train(lgb_params, lgb.Dataset(train_x, train_y),
                            verbose_eval=20, num_boost_round=100)
            del train_x, train_y
            gc.collect()
            fold_importance_df_fold = self.get_fea_importance(clf, feature_name)
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df_fold], axis=0)
        feature_importance_df = feature_importance_df[
            ["feature", "split", "gain", "gain_percent"]
        ].groupby("feature", as_index=False).mean().sort_values(by="gain", ascending=False)
        print(feature_importance_df)
        self.gain_filter_f(feature_importance_df)
        return feature_importance_df

    @timmer
    def lgb_select(self, df):

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            'verbose': -1,
            "seed": self.random_state,
            'num_threads': 8,
            'boosting': 'rf',
            'num_leaves': 618,
            'max_depth': 10,
            'bagging_freq': 1,
            'bagging_fraction': 0.8,
            'feature_fraction': max(math.log(df.shape[1], 2) / df.shape[1], 0.1),
        }
        feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]
        train_x = df[feature_name][df['label'].isnull() == False]
        train_y = df[['label']][df['label'].isnull() == False]
        clf = lgb.train(lgb_params, lgb.Dataset(train_x, train_y),
                        verbose_eval=20, num_boost_round=100)
        del train_x, train_y
        gc.collect()
        fold_importance_df = self.get_fea_importance(clf, feature_name)
        # if self.is_select_every_step:
        self.gain_filter_f(fold_importance_df)
        return fold_importance_df

    @timmer
    def kflod_lgb_select(self, df):
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            'verbose': -1,
            "seed": self.random_state,
            'num_threads': 8,
            'boosting': 'rf',
            'num_leaves': 618,
            'max_depth': 10,
            'bagging_freq': 1,
            'bagging_fraction': 0.8,
            'feature_fraction': max(math.log(df.shape[1], 2) / df.shape[1], 0.1),
        }
        feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]
        train_X = df[feature_name][df['label'].isnull() == False]
        train_Y = df[['label']][df['label'].isnull() == False]
        feature_importance_df = pd.DataFrame()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for n_fold, (train_idx, valid_idx) in enumerate(skf.split(train_X, train_Y['label'])):
            train_x, train_y = train_X.iloc[train_idx], train_Y.iloc[train_idx]
            valid_x, valid_y = train_X.iloc[valid_idx], train_Y.iloc[valid_idx]

            clf = lgb.train(lgb_params, lgb.Dataset(train_x, train_y),
                            verbose_eval=20, num_boost_round=100)
            fold_importance_df_fold = self.get_fea_importance(clf, feature_name)
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df_fold], axis=0)
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        feature_importance_df = feature_importance_df[
            ["feature", "split", "gain", "gain_percent"]
        ].groupby("feature", as_index=False).mean().sort_values(by="gain", ascending=False)
        # print(feature_importance_df)
        self.gain_filter_f(feature_importance_df)
        return feature_importance_df

    def gain_filter_f(self, fold_importance_df):
        lgb_drop = fold_importance_df[fold_importance_df['gain'] <= 0.001]['feature']
        lgb_drop = [i for i in lgb_drop if i not in self.drop_name]
        self.drop_name.extend(lgb_drop)
        print("LGB SELECTION DROP", lgb_drop)
        return

    def get_imp_fea_seperate(self, kind, rate):
        fea_imp = self.fea_importance_now
        if kind == 'CAT':
            df = fea_imp[fea_imp.feature.str.startswith('cat_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'NUM':
            df = fea_imp[fea_imp.feature.str.startswith('num_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'MV':
            df = fea_imp[fea_imp.feature.str.startswith('mv_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'TIME':
            df = fea_imp[fea_imp.feature.str.startswith('time_')]
            return list(df['feature'][:int(len(df) * rate)])
        elif kind == 'CAT+MV':
            df = fea_imp[fea_imp.feature.str.startswith('cat_') | fea_imp.feature.str.startswith('mv_')]
            return list(df['feature'][:int(len(df) * rate)])

    def get_fea_importance(self, clf, feature_name):
        gain = clf.feature_importance('gain')
        importance_df = pd.DataFrame({
            'feature': clf.feature_name(),
            'split': clf.feature_importance('split'),
            'gain': gain,  # * gain / gain.sum(),
            'gain_percent': 100 * gain / gain.sum(),
        }).sort_values('gain', ascending=False)
        return importance_df

    def get_redundant_pairs(self, df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(self, df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = self.get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    #################### utils for model train ######################

    @timmer
    def lgb_model_train(self, df):
        df = df[df['label'].isnull() == False]
        if self.batch_idx != 1:
            feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]
        else:
            feature_name = [i for i in df.columns if i not in (self.drop_name + ['label']) and 'target' not in i]
        if self.is_use_adversial:
            feature_name = [i for i in feature_name if i not in self.adversial_drop]
        print(df['label'].mean())
        if self.best_iter is None or self.batch_idx == 1:
            X_train, X_val, y_train, y_val = self.train_test_split(df[feature_name], df['label'].values,
                                                                   self.train_test_rate, self.random_state)

            del df
            gc.collect()
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            del X_train, X_val, y_train, y_val
            gc.collect()
            clf = lgb.train(
                self.params_lgb, lgb_train, valid_sets=lgb_val, valid_names='eval',
                verbose_eval=50, early_stopping_rounds=30, num_boost_round=1500)
            self.best_iter = clf.best_iteration
        else:
            lgb_train = lgb.Dataset(df[feature_name], df['label'].values)

            del df
            gc.collect()
            round_num = int(self.best_iter * 1.1)
            clf = lgb.train(self.params_lgb, lgb_train, num_boost_round=round_num)
            # clf = lgb.train(self.params_lgb, lgb_train, num_boost_round= 1000)
        fea_importance_now = self.get_fea_importance(clf, feature_name)
        self.fea_importance_batch = fea_importance_now
        if self.is_select_every_step:
            self.gain_filter_f(fea_importance_now)
        print("BATCH {} FEATURE IMPROTANCE".format(self.batch_idx))
        print("DROP:", self.drop_name)
        print(fea_importance_now.head(15))
        return clf

    def lgb_kfold_train(self, df):
        test = df[df['label'].isnull() == True]
        # df = df[df['label'].isnull()==False]
        if self.is_use_adversial:
            feature_name = [i for i in df.columns if
                            i not in (self.drop_name + self.adversial_drop + ['label']) and 'target' not in i]
        else:
            feature_name = [i for i in df.columns if i not in (self.drop_name + ['label']) and 'target' not in i]
        train_X = df[feature_name][df['label'].isnull() == False]
        train_Y = df[['label']][df['label'].isnull() == False]
        feature_importance_df = pd.DataFrame()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        sub_preds = np.zeros(test.shape[0])
        best_iter = []
        for n_fold, (train_idx, valid_idx) in enumerate(skf.split(train_X, train_Y['label'])):
            train_x, train_y = train_X.iloc[train_idx], train_Y.iloc[train_idx]
            valid_x, valid_y = train_X.iloc[valid_idx], train_Y.iloc[valid_idx]
            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_val = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
            clf = lgb.train(self.params_lgb, lgb_train, valid_sets=lgb_val, feature_name=feature_name,
                            verbose_eval=50, early_stopping_rounds=30, num_boost_round=1500)
            best_iter.append(clf.best_iteration)
            fold_importance_df_fold = self.get_fea_importance(clf, feature_name)
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df_fold], axis=0)
            del train_x, train_y, valid_x, valid_y
            gc.collect()
            sub_preds += clf.predict(test[feature_name], num_iteration=clf.best_iteration) / skf.n_splits
        del train_X, train_Y
        gc.collect()
        feature_importance_df = feature_importance_df[
            ["feature", "split", "gain", "gain_percent"]
        ].groupby("feature", as_index=False).mean().sort_values(by="gain", ascending=False)
        self.best_iter = int(np.mean(best_iter))
        # fea_importance_now = self.get_fea_importance(clf, feature_name)
        return sub_preds

    @timmer
    def adversial_validation(self, df):
        feature_name = [i for i in df.columns if i not in (self.drop_name + ['label']) and \
                        ('target' not in i) and (i.startswith('time_') == False)]
        if len(df) > 60000:
            frac = 60000.0 / len(df)
            df = df.sample(frac=frac)
        train_X = df[feature_name][df['label'].isnull() == False]
        train_Y = np.ones(train_X.shape[0])
        test_X = df[feature_name][df['label'].isnull() == True]
        test_Y = np.zeros(test_X.shape[0])

        X = np.concatenate((train_X.values, test_X.values), axis=0)
        y = np.concatenate((train_Y, test_Y), axis=0)
        test_size = int(len(X) / 5)
        X, X_test, y, y_test = self.train_test_split(X, y, test_size, self.random_state)

        para = {
            'num_leaves': 6,
            'learning_rate': 0.1,
            'bagging_fraction': 0.2,
            'feature_fraction': 0.5,
            'max_depth': 3,
            "objective": "binary",
            "metric": "auc",
            'verbose': -1,
            "seed": self.random_state,
            'num_threads': 8,
        }
        lgb_train = lgb.Dataset(X, y, free_raw_data=True)
        lgb_val = lgb.Dataset(X_test, y_test, free_raw_data=True, reference=lgb_train)
        del X
        del y
        gc.collect()
        lgb_model = lgb.train(para, lgb_train, valid_sets=lgb_val, valid_names='eval', feature_name=feature_name,
                              verbose_eval=False, early_stopping_rounds=10, num_boost_round=50)
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration))
        auc = metrics.auc(fpr, tpr)
        print("----Adversial Score is {}------".format(auc))
        fea_importance_adversial = self.get_fea_importance(lgb_model, feature_name)
        self.adversial_drop = list(fea_importance_adversial['feature'][fea_importance_adversial.gain_percent > 15])
        print(fea_importance_adversial.head())
        return

    @timmer
    def lgb_mode_preict(self, df):
        clf = self.lgb_model_train(df)
        if self.batch_idx == 1:
            feature_name = [i for i in df.columns if i not in (self.drop_name + ['label']) and 'target' not in i]
        else:
            feature_name = [i for i in df.columns if i not in self.drop_name + ['label']]

        if self.is_use_adversial:
            feature_name = [i for i in feature_name if i not in self.adversial_drop]

        X = df[df['label'].isnull() == True][feature_name]
        y = clf.predict(X, num_iteration=clf.best_iteration)
        del X
        gc.collect()
        return y

    def mete_stat(self, df):
        stats = []
        for col in df.columns:
            stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0],
                          df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
        stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                                'Percentage of values in the biggest category', 'type'])
        stats_df.sort_values('Percentage of missing values', ascending=False)
        print(stats_df.head(10))
        return stats_df

    def gen_config(self, df):
        X = df
        print('Topk-rate score: ', (X.shape[0] / 10000) * self.ncat / 10 / math.sqrt(self.time_budget / 600))
        if (X.shape[0] / 10000) * self.ncat / 10 / math.sqrt(self.time_budget / 600) < 20:
            print("--------SMALL--------")
            # C
            self.count_rate = 0.6
            self.gp_rate = 0.5
            self.numcross_rate = 0.35
            self.bi_count_rate = 0.6
            self.delete_overfit_rate = 0.2
            # self.is_use_bicounts_f = False
            # self.is_easy_overfit = True
            self.slide_window = 10
            self.target_encode_rate = 0.5
            self.window = deque(maxlen=self.slide_window)
        elif (X.shape[0] / 10000) * self.ncat / 10 / math.sqrt(self.time_budget / 600) < 50:
            print("------MEDIEM-------")
            # B D MEDIEM
            self.count_rate = 0.6
            self.gp_rate = 0.45
            self.numcross_rate = 0.3
            self.bi_count_rate = 0.6
            self.slide_window = 10
            self.target_encode_rate = 0.6
            self.window = deque(maxlen=self.slide_window)
        elif (X.shape[0] / 10000) * self.ncat / 10 / math.sqrt(self.time_budget / 600) < 120:
            print("------BIG------")
            self.count_rate = 0.6
            self.bi_count_rate = 0.6
            self.target_encode_rate = 0.6
            self.gp_rate = 0.382
            self.numcross_rate = 0.3
            self.slide_window = 6
            self.window = deque(maxlen=self.slide_window)
        elif (X.shape[0] / 10000) * self.ncat / 10 / math.sqrt(self.time_budget / 600) < 180:
            print("-----LARGE------")
            # A LARGE
            self.is_easy_overfit = True
            self.delete_overfit_rate = 0.15
            self.count_rate = 0.8
            self.bi_count_rate = 0.8
            self.gp_rate = 0.25
            self.target_encode_rate = 0.3
            self.numcross_rate = 0.3
            self.slide_window = 2
            self.window = deque(maxlen=self.slide_window)
            self.is_use_kfold = False
        else:
            print("-----EXTREME------")
            # E extreme
            self.count_rate = 0.85
            self.bi_count_rate = 0.85
            self.gp_rate = 0.25
            self.numcross_rate = 0.3
            self.slide_window = 2
            self.window = deque(maxlen=self.slide_window)
            self.is_use_kfold = False
        return

    def train_test_split(self, X, y, test_size, random_state=2018):
        sss = list(StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state).split(X, y))

        X_train = np.take(X, sss[0][0], axis=0)
        X_test = np.take(X, sss[0][1], axis=0)

        y_train = np.take(y, sss[0][0], axis=0)
        y_test = np.take(y, sss[0][1], axis=0)

        return [X_train, X_test, y_train, y_test]

    def log_stage_time(self, stage):
        time_left = self.time_budget + self.start_time - time.time()
        time_usage = time.time() - self.start_time
        print("stage : {}, time usage {}, time left {} \n".format(stage, time_left, time_usage))

        # staring kit save

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

        # starting kit load

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

    def memory_usage():
        # process = psutil.Process(os.getpid())
        # mem = process.memory_info()[0] / float(2 ** 20)
        return 1

    # utils to resuce the memory usage
    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            if col == 'label':
                continue
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df.loc[:, col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                        df.loc[:, col] = df[col].astype(np.uint8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df.loc[:, col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                        df.loc[:, col] = df[col].astype(np.uint16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df.loc[:, col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                        df.loc[:, col] = df[col].astype(np.uint32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df.loc[:, col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df.loc[:, col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df.loc[:, col] = df[col].astype(np.float32)
                    else:
                        df.loc[:, col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df

    #################### utils for feature sample #############################
    @timmer
    def balance(self, df):
        print("balance before: len X: is {}".format(len(df)))
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        if len(pos_df) > len(neg_df):
            more_df, less_df = pos_df, neg_df
        else:
            less_df, more_df = pos_df, neg_df
        if len(less_df) * 1.0 / len(more_df) < self.imbalanced_rate:
            more_df = more_df.sample(n=int(less_df.shape[0] * 1.0 / self.imbalanced_rate),
                                     random_state=self.random_state)
            #            self.params_lgb['is_unbalance'] = 'true'
            self.params_lgb['scale_pos_weight'] = int(1 / self.imbalanced_rate * 0.06)
        else:
            # self.params_lgb['scale_pos_weight'] = 2
            self.params_lgb['is_unbalance'] = 'false'
        df = less_df.append(more_df).reset_index(drop=True).sample(frac=1)
        del less_df, more_df
        gc.collect()
        print("balance after: len X: is {}".format(len(df)))
        return df

    def shrink(self, df, shrink_sample, random_state=2018):
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        if len(pos_df) > len(neg_df):
            more_df, less_df = pos_df, neg_df
        else:
            less_df, more_df = pos_df, neg_df
        if (len(pos_df) + len(neg_df)) > shrink_sample:
            if 2 * len(less_df) < shrink_sample:
                more_df = more_df.sample(n=int(shrink_sample - len(less_df)), random_state=random_state)
            else:
                more_df = more_df.sample(n=int(shrink_sample / 2), random_state=random_state)
                less_df = less_df.sample(n=int(shrink_sample / 2), random_state=random_state)
        df = less_df.append(more_df).reset_index(drop=True).sample(frac=1)
        del less_df, more_df
        gc.collect()
        return df

    @timmer
    def sample(self, df, random_state=2018):
        start_time = time.time()
        if self.batch_idx == 1:
            self.total_size = df.shape[0]
            # for i in self.window:
            #    self.total_size += i.shape[0]

        print('total shape', self.total_size)
        if self.batch_idx != 1:
            left_avg = (self.start_time - time.time() +
                        self.time_budget) / (10 - self.batch_idx)
            print('batch_time: ', self.batch_time)
            print('left_avg: ', left_avg, '\n')
            self.total_shrink = int(
                min(self.total_shrink, self.total_size * left_avg * 2 / self.batch_time))
            if self.batch_time > left_avg and self.is_dynamic:
                self.shrink_decay = left_avg / self.batch_time * 0.9
                self.batch_shrink = int(max(
                    (self.batch_shrink * self.shrink_decay), 100))
                self.total_shrink = int(max(
                    (self.total_size * self.shrink_decay), 100))
                if self.batch_idx >= 3 and self.is_drop_feature:
                    num_fea = len([i for i in df.columns if i not in self.drop_name + ['label']])
                    topk = int(num_fea * self.shrink_decay)
                    shrink_drop = list(self.fea_importance_batch['feature'][topk:])
                    print('shrink drop', shrink_drop)
                    self.drop_name.extend(shrink_drop)
                    # for i in range(len(self.slide_window)):
                    #    slide_window[i].drop(shrink_drop, inplace=True, axis=1)
                    df.drop(shrink_drop, inplace=True, axis=1)
                    self.is_drop_feature = False
            shrink_now = self.total_shrink
            df = self.shrink(df, self.total_shrink, self.random_state)
            # def shrink(self, df, shrink_sample, random_state=2018):
            print("after sample size is {}".format(df.shape[0]))
        return df

    ######################### utils for feature select #################################
    @timmer
    def bayes_parameter_opt_lgb(self, X, y):
        # test_size = min(1200, int(X.shape[0] / 4.0))
        test_size = int(X.shape[0] / 5.0)
        X, X_test, y, y_test = self.train_test_split(X, y, test_size, self.random_state)
        # parameters
        params = self.params_lgb.copy()
        print(X.shape[0])
        if X.shape[0] < 50000:
            params_lgb = [
                {'num_leaves': 25, 'learning_rate': 0.01,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 50,
                 "boost_from_average": False,
                 },

                {'num_leaves': 15, 'learning_rate': 0.05,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 50,
                 "boost_from_average": False,
                 },

                {'num_leaves': 20, 'learning_rate': 0.05,
                 'bagging_fraction': 0.7, 'feature_fraction': 0.7,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.6, 'lambda_l2': 0.6, "min_data": 50,
                 "boost_from_average": False,
                 },

                {'num_leaves': 25, 'learning_rate': 0.05,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 50,
                 "boost_from_average": False,
                 },

                {'num_leaves': 25, 'learning_rate': 0.05,
                 'bagging_fraction': 0.9, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 60,
                 "boost_from_average": False,
                 },

                {'num_leaves': 30, 'learning_rate': 0.05,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 75,
                 "boost_from_average": False,
                 },

                {'num_leaves': 50, 'learning_rate': 0.1,
                 'bagging_fraction': 0.9, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 150,
                 "boost_from_average": False,
                 },

                {'num_leaves': 32, 'learning_rate': 0.1,
                 'bagging_fraction': 0.85, 'feature_fraction': 0.85,
                 'max_depth': -1,
                 },
            ]
        elif X.shape[0] < 240000:
            params_lgb = [
                {'num_leaves': 15, 'learning_rate': 0.025,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 25,
                 "boost_from_average": True,
                 },

                {'num_leaves': 20, 'learning_rate': 0.05,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 50,
                 "boost_from_average": True,
                 },

                {'num_leaves': 30, 'learning_rate': 0.01,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 100,
                 "boost_from_average": True,
                 },

                {'num_leaves': 30, 'learning_rate': 0.075,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 100,
                 "boost_from_average": True,
                 },

                {'num_leaves': 50, 'learning_rate': 0.1,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.9,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 200,
                 "boost_from_average": True,
                 },

                {'num_leaves': 32, 'learning_rate': 0.025,
                 'bagging_fraction': 0.9, 'feature_fraction': 0.5,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 25,
                 # "boost_from_average": True,
                 },
                {'num_leaves': 32, 'learning_rate': 0.03,
                 'bagging_fraction': 0.85, 'feature_fraction': 0.55,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.6, 'lambda_l2': 0.6,  # "min_data": 25,
                 # "boost_from_average": True,
                 },
            ]
        else:
            params_lgb = [
                {'num_leaves': 50, 'learning_rate': 0.13,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 5, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 100,
                 "boost_from_average": True,
                 },

                {'num_leaves': 60, 'learning_rate': 0.13,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.4, 'lambda_l2': 0.4, "min_data": 250,
                 "boost_from_average": True,
                 },

                {'num_leaves': 80, 'learning_rate': 0.13,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 300,
                 "boost_from_average": True,
                 },
                {'num_leaves': 80, 'learning_rate': 0.05,
                 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
                 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.5, 'lambda_l2': 0.5, "min_data": 300,
                 "boost_from_average": True,
                 },

                # {'num_leaves': 32, 'learning_rate': 0.1,
                # 'bagging_fraction': 0.85, 'feature_fraction': 0.85,
                # 'max_depth': -1,
                # },
                # { 'num_leaves': 60, 'learning_rate': 0.13,
                # 'bagging_fraction': 0.8, 'bagging_freq': 1,
                # 'min_sum_hessian_in_leaf': 0.1, 'lambda_l1': 0.4,
                # 'lambda_l2': 0.4, 'boost_from_average': True,
                # 'feature_fraction': 0.8, 'min_data': 250}
            ]

        result = []
        # X_train, X_val, y_train, y_val = self.train_test_split(X, y, test_size, self.random_state + 107)
        lgb_train = lgb.Dataset(X, y, free_raw_data=True)
        lgb_val = lgb.Dataset(X_test, y_test, free_raw_data=True, reference=lgb_train)
        del X
        del y
        gc.collect()
        for i, param in enumerate(params_lgb):
            params.update(param)
            lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_val, valid_names='eval',
                                  verbose_eval=False, early_stopping_rounds=30, num_boost_round=1000)
            # y_pred = lgb_model.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(
                y_test, lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration))
            auc = metrics.auc(fpr, tpr)
            best_iter = lgb_model.best_iteration
            result.append({"params": param, "score": auc, "best_iter": best_iter})
            print("score:", str(auc), ", best_iter:", str(best_iter))

        sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)
        # print(sorted_result)
        return sorted_result[0]["params"], sorted_result[0]["best_iter"]

    @timmer
    def hyperband_lgb(self, X, y, max_iter=800, eta=2):
        import hyperband

        search_space = {'learning_rate': {'_type': 'quniform', '_value': [0.01, 0.1, 0.02]},
                        'num_leaves': {'_type': 'quniform', '_value': [15, 80, 20]},
                        'bagging_fraction': {'_type': 'choice', '_value': [0.8, 0.9]},
                        'feature_fraction': {'_type': 'choice', '_value': [0.8, 0.9]},
                        'min_sum_hessian_in_leaf': {'_type': 'choice', '_value': [0.1, 5]},
                        'lambda_l1': {'_type': 'choice', '_value': [0.4, 0.5]},
                        'lambda_l2': {'_type': 'choice', '_value': [0.1, 0.6]},
                        }

        hyperband_tuner = hyperband.Hyperband(max_iter=max_iter, eta=eta, X=X, y=y, params_lgb=self.params_lgb)
        # init
        hyperband_tuner.update_search_space(search_space)
        print("init done...")
        # test request
        result, best_iter = hyperband_tuner.run()
        # return result['param'], result['best_iter']
        return result, best_iter











































    ""
