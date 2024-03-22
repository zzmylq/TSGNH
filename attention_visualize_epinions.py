import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import sklearn.decomposition


rating_filename = "data/epinions/epinions_rating.json"
attr_filename = "data/epinions/epinions_item_attr.json"
friend_filename = "data/epinions/epinions_friend.json"

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

selected_users = [3671]
selected_users_friends_similarity = []


for user in selected_users:
    friends_similarity = []
    user_friends = friends[friends['user_id'] == user]['friends'].array.tolist()[0].split(', ')
    user_feature = set(ratings[ratings['user_id'] == user]['item_id'].to_list())
    for friend in user_friends:
        friend_feature = set(ratings[ratings['user_id'] == int(friend)]['item_id'].to_list())
        sim = len(user_feature & friend_feature) / len(user_feature | friend_feature)
        friends_similarity.append(sim)
    selected_users_friends_similarity.append(friends_similarity)

    print("selected_user : ", user)
    for u,sim in zip(user_friends, friends_similarity):
        print(u,sim/sum(friends_similarity))



