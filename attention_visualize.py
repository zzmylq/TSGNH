import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import sklearn.decomposition


rating_filename = "data/kuai_big/kuai_big_rating.json"
attr_filename = "data/kuai_big/kuai_big_item_attr.json"
friend_filename = "data/kuai_big/kuai_big_friend.json"

rating_raw_data = []
with open(rating_filename, "r", encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        json_data = json.loads(line)
        rating_raw_data.append(dict(json_data))
print("source data loaded. rating record number: ", len(rating_raw_data))

attr_raw_data = []
with open(attr_filename, "r", encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        json_data = json.loads(line)
        attr_raw_data.append(dict(json_data))
print("source data loaded. attr record number: ", len(attr_raw_data))

friend_raw_data = []
with open(friend_filename, "r", encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        json_data = json.loads(line)
        friend_raw_data.append(dict(json_data))
print("source data loaded. friend record number: ", len(friend_raw_data))

rating_raw_data.sort(key=lambda x: [x['user_id'], x['date']], reverse=True)
attr_raw_data.sort(key=lambda x: [x['item_id']])
friend_raw_data.sort(key=lambda x: [x['user_id']])

for k in range(len(attr_raw_data)):
    temp_list = []
    for i in attr_raw_data[k]['attr'].split(', '):
        temp_list.append(str(i))
    attr_raw_data[k]['attr'] = temp_list

ratings = pd.DataFrame(rating_raw_data)
attrs = pd.DataFrame(attr_raw_data)
friends = pd.DataFrame(friend_raw_data)

multi_hot_attrs = attrs["attr"].str.join("|").str.get_dummies()
multi_hot_attrs = multi_hot_attrs[[str(i) for i in range(31)]]
attrs_multi_hot = attrs.join(multi_hot_attrs)[['item_id']+[str(i) for i in range(31)]]

data = pd.merge(ratings, attrs_multi_hot).sort_values(['user_id', 'date'], ascending=[True, False]).drop_duplicates()

from sklearn.metrics.pairwise import cosine_similarity

selected_users = [1341, 4282, 6345, 6582, 1121, 807, 1996, 2520, 3162, 6028]
selected_users_friends_similarity = []

feature_sum_sim = False

for user in selected_users:
    friends_similarity = []
    user_friends = friends[friends['user_id'] == user]['friends'].to_string(index=False).split(', ')
    if feature_sum_sim:
        user_feature = data[data['user_id'] == user][[str(i) for i in range(31)]].to_numpy().sum(axis=0)
    else:
        user_feature = set(data[data['user_id'] == user]['item_id'].to_list())
    for friend in user_friends:
        if feature_sum_sim:
            friend_feature = data[data['user_id'] == int(friend)][[str(i) for i in range(31)]].to_numpy().sum(axis=0)
            sim = cosine_similarity(user_feature.reshape(1, -1), friend_feature.reshape(1, -1))
            friends_similarity.append(sim[0][0])
        else:
            friend_feature = set(data[data['user_id'] == int(friend)]['item_id'].to_list())
            sim = len(user_feature & friend_feature) / len(user_feature | friend_feature)
            friends_similarity.append(sim)
    selected_users_friends_similarity.append(friends_similarity)



