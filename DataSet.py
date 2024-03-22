# -*- Encoding:UTF-8 -*-
import collections
import re
from config import Config
import numpy as np
#import sys
import json
from tqdm import tqdm
import pickle
import pandas as pd

#from sklearn.decomposition import NMF
class DataSet(object):
    def __init__(self, rating_filename, item_attr_filename, friend_filename):
        self.config = Config()
        self.rating_filename = rating_filename
        self.item_attr_filename = item_attr_filename
        self.friend_filename = friend_filename

    def get_data(ds):
        rating_raw_data = []
        print("loading file...")
        with open(ds.rating_filename, "r", encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                json_data = json.loads(line)
                rating_raw_data.append(dict(json_data))
        print("source data loaded. rating record number: ", len(rating_raw_data))

        item_attr_raw_data = []
        print("loading file...")
        with open(ds.item_attr_filename, "r", encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                json_data = json.loads(line)
                item_attr_raw_data.append(dict(json_data))
        print("source data loaded. item attr record number: ", len(item_attr_raw_data))

        friend_raw_data = []
        print("loading file...")
        with open(ds.friend_filename, "r", encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                json_data = json.loads(line)
                friend_raw_data.append(dict(json_data))
        print("source data loaded. friend record number: ", len(friend_raw_data))

        print("getting TrainSet and TestSet...")
        train_ratings = []
        test_ratings = []
        for i in tqdm(range(len(rating_raw_data))):
            user = rating_raw_data[i]['user_id']
            item = rating_raw_data[i]['item_id']
            rate = rating_raw_data[i]['rating']
            date = rating_raw_data[i]['date']
            try:
                if rating_raw_data[i]['user_id'] != rating_raw_data[i+1]['user_id']:
                    test_ratings.append((user, item, rate, date))
                else:
                    train_ratings.append((user, item, rate, date))
            except IndexError:
                test_ratings.append((user, item, rate, date))

        train_ratings_user_sorted = sorted(train_ratings, key=lambda x: [x[0], x[3]], reverse=True)
        history_u_lists = collections.defaultdict(list)
        history_ur_lists = collections.defaultdict(list)
        history_ut_lists = collections.defaultdict(list)
        v_list = []
        r_list = []
        t_list = []
        index = 0
        for i in tqdm(range(len(train_ratings_user_sorted))):
            try:
                if train_ratings_user_sorted[i][0] != train_ratings_user_sorted[i + 1][0]:
                    v_list.append(train_ratings_user_sorted[i][1])
                    r_list.append(int(train_ratings_user_sorted[i][2]) - 1)
                    t_list.append(index)
                    history_u_lists[train_ratings_user_sorted[i][0]] = v_list
                    history_ur_lists[train_ratings_user_sorted[i][0]] = r_list
                    history_ut_lists[train_ratings_user_sorted[i][0]] = t_list
                    v_list = []
                    r_list = []
                    t_list = []
                    index = 0
                else:
                    v_list.append(train_ratings_user_sorted[i][1])
                    r_list.append(int(train_ratings_user_sorted[i][2]) - 1)
                    t_list.append(index)
                    index += 1
            except IndexError:
                v_list.append(train_ratings_user_sorted[i][1])
                r_list.append(int(train_ratings_user_sorted[i][2]) - 1)
                t_list.append(index)
                history_u_lists[train_ratings_user_sorted[i][0]] = v_list
                history_ur_lists[train_ratings_user_sorted[i][0]] = r_list
                history_ut_lists[train_ratings_user_sorted[i][0]] = t_list

        train_ratings_item_sorted = sorted(train_ratings, key=lambda x: [x[1], x[3]], reverse=True)
        history_v_lists = collections.defaultdict(list)
        history_vr_lists = collections.defaultdict(list)
        history_vt_lists = collections.defaultdict(list)
        vu_list = []
        vr_list = []
        vt_list = []
        index = 0
        for i in tqdm(range(len(train_ratings_item_sorted))):
            try:
                if train_ratings_item_sorted[i][1] != train_ratings_item_sorted[i + 1][1]:
                    vu_list.append(train_ratings_item_sorted[i][0])
                    vr_list.append(int(train_ratings_item_sorted[i][2]) - 1)
                    vt_list.append(index)
                    history_v_lists[train_ratings_item_sorted[i][1]] = vu_list
                    history_vr_lists[train_ratings_item_sorted[i][1]] = vr_list
                    history_vt_lists[train_ratings_item_sorted[i][1]] = vt_list
                    vu_list = []
                    vr_list = []
                    vt_list = []
                    index = 0
                else:
                    vu_list.append(train_ratings_item_sorted[i][0])
                    vr_list.append(int(train_ratings_item_sorted[i][2]) - 1)
                    vt_list.append(index)
                    index += 1
            except IndexError:
                vu_list.append(train_ratings_item_sorted[i][0])
                vr_list.append(int(train_ratings_item_sorted[i][2]) - 1)
                vt_list.append(index)
                history_v_lists[train_ratings_item_sorted[i][1]] = vu_list
                history_vr_lists[train_ratings_item_sorted[i][1]] = vr_list
                history_vt_lists[train_ratings_item_sorted[i][1]] = vt_list

        history_va_lists = collections.defaultdict(list)
        va_list = []
        for i in tqdm(range(len(item_attr_raw_data))):
            for j in item_attr_raw_data[i]['attr'].split(', '):
                va_list.append(int(j))
            history_va_lists[int(item_attr_raw_data[i]['item_id'])] = va_list
            va_list = []

        ai_dict = dict()
        for i in tqdm(range(len(item_attr_raw_data))):
            for j in item_attr_raw_data[i]['attr'].split(', '):
                ai_dict.setdefault(int(j), []).append(int(item_attr_raw_data[i]['item_id']))
        history_au_lists = collections.defaultdict(list)
        au_set = set()
        for attr in tqdm(ai_dict):
            for item in ai_dict[attr]:
                au_set.update(history_v_lists[item])
            history_au_lists[attr] = au_set
            au_set = set()

        print("friend...")
        social_adj_lists = collections.defaultdict(list)
        social_list = set()
        for i in tqdm(range(len(friend_raw_data))):
            for j in friend_raw_data[i]['friends'].split(', '):
                social_list.add(int(j))
            social_adj_lists[int(friend_raw_data[i]['user_id'])] = social_list
            social_list = set()

        print("rating list...")

        ratings_list = collections.defaultdict(list)
        rating_set = set()
        for i in tqdm(range(len(rating_raw_data))):
            rating_set.add(int(rating_raw_data[i]['rating']))
        ratings_list.update(zip(rating_set, range(len(rating_set))))

        print("train and test...")

        train_u = []
        train_v = []
        train_r = []
        train_a = []
        test_u = []
        test_v = []
        test_r = []
        for i in train_ratings:
            train_u.append(i[0])
            train_v.append(i[1])
            train_r.append(i[2])
            train_a.append(history_va_lists[i[1]][0])
        for i in test_ratings:
            test_u.append(i[0])
            test_v.append(i[1])
            test_r.append(i[2])

        num_items = history_v_lists.__len__()
        print("testNeg...")
        testNeg = ds.getTestNeg(test_u, test_v, test_r, history_u_lists, num_items, ds.config.negNum)

        print("attr famous...")
        # attr famous
        temp_dict = dict()
        for attr in history_au_lists:
            temp_dict[attr] = len(history_au_lists[attr])

        data = list(temp_dict.values())

        k = ds.config.famous_labels_num
        d1 = pd.cut(data, k, labels=range(k))
        count = 0
        famous_labels = list(d1)
        for key in temp_dict:
            temp_dict[key] = famous_labels[count]
            count += 1

        history_af_lists = temp_dict

        write_filename = "data/" + ds.config.dataset + ".pickle"
        with open(write_filename, "wb") as f:
            pickle.dump((history_u_lists, history_ur_lists, history_ut_lists, \
            history_v_lists, history_vr_lists, history_vt_lists, \
            history_va_lists, \
            social_adj_lists, ratings_list, history_af_lists, \
            train_u, train_v, train_r, train_a, test_u, test_v, test_r, testNeg), f)

        return history_u_lists, history_ur_lists, history_ut_lists, \
            history_v_lists, history_vr_lists, history_vt_lists, \
            history_va_lists, \
            social_adj_lists, ratings_list, history_af_lists, \
            train_u, train_v, train_r, train_a, test_u, test_v, test_r, testNeg

    def getTestNeg(self, test_u, test_v, test_r, history_u_lists, num_items, negNum):
        user = []
        item = []
        rate = []
        for i in tqdm(range(len(test_u))):
            tmp_user = []
            tmp_item = []
            tmp_rate = []

            u = test_u[i]
            v = test_v[i]
            r = test_r[i]
            tmp_user.append(u)
            tmp_item.append(v)
            tmp_rate.append(r)

            neglist = set()
            neglist.add(v)
            temp_num = num_items - len(history_u_lists[u])
            for t in range(negNum):
                j = np.random.randint(num_items)
                if temp_num < negNum:
                    while j in history_u_lists[u]:
                        j = np.random.randint(num_items)
                else:
                    while j in history_u_lists[u] or j in neglist:
                        j = np.random.randint(num_items)
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
                tmp_rate.append(1)
            user.append(tmp_user)
            item.append(tmp_item)
            rate.append(tmp_rate)
        return (np.array(user), np.array(item), np.array(rate))
