import heapq
import math

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from VA_Encoders import VA_Encoder
from VA_Aggregators import VA_Aggregator
from UA_Encoders import UA_Encoder
from UA_Aggregators import UA_Aggregator
from Merge import Merge
from Merge_Encoder import Merge_Encoder

from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
from DataSet import DataSet
from config import Config
from Loss import Loss
import datetime
import time
from tqdm import tqdm

args_global = Config()

torch.manual_seed(args_global.random_seed)
torch.cuda.manual_seed(args_global.random_seed)
torch.cuda.manual_seed_all(args_global.random_seed) # if you are using multi-GPU.
np.random.seed(args_global.random_seed) # Numpy module.
random.seed(args_global.random_seed) # Python random module.
torch.manual_seed(args_global.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.config = Config()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.config.hash_code_length)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.config.hash_code_length)

        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()
        self.atanh_alpha = nn.Parameter(torch.FloatTensor([self.config.atanh_alpha_init]))

    def forward(self, nodes_u, nodes_v, device):
        embeds_u = self.enc_u(nodes_u).to(device)
        embeds_v = self.enc_v_history(nodes_v).to(device)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.bn1(self.w_ur1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_ur2(x_v)

        if self.config.atanh:
            x_u = torch.tanh(torch.mul(x_u, self.atanh_alpha))
            x_v = torch.tanh(torch.mul(x_v, self.atanh_alpha))
            scores = 0.5 + (1 / (2 * self.config.hash_code_length)) * torch.sum(torch.mul(x_u, x_v), dim=1)
        else:
            scores = (torch.cosine_similarity(x_u, x_v) + 1) / 2

        return scores.squeeze(), x_u, x_v

    def get_u_embeddings(self, nodes_u, device):
        embeds_u = self.enc_u(nodes_u).to(device)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        if self.config.atanh:
            x_u = torch.tanh(torch.mul(x_u, self.atanh_alpha))
        return x_u

    def get_v_embeddings(self, nodes_v, device):
        embeds_v = self.enc_v_history(nodes_v).to(device)


        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        if self.config.atanh:
            x_v = torch.tanh(torch.mul(x_v, self.atanh_alpha))
        return x_v

    def loss(self, nodes_u, nodes_v, labels_list, device):
        scores, x_u, x_v = self.forward(nodes_u, nodes_v, device)
        loss = Loss()
        return loss(scores, labels_list, self.atanh_alpha, x_u, x_v, device)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()

    running_loss = 0.0
    t1 = time.time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device), device)
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()

    print("epoch", epoch, "average loss:", running_loss / (i + 1), "time cost:", time.time() - t1)
    return running_loss / (i + 1)


def evaluate(model, device, testNeg):
    model.eval()
    def getHitRatio(ranklist, targetItem):
        for item in ranklist:
            if item == targetItem:
                return 1
        return 0
    def getNDCG(ranklist, targetItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == targetItem:
                return math.log(2) / math.log(i+2)
        return 0

    hr1 = []
    NDCG1 = []
    hr5 = []
    NDCG5 = []
    hr10 = []
    NDCG10 = []
    hr15 = []
    NDCG15 = []
    hr20 = []
    NDCG20 = []

    testUser = testNeg[0]
    testItem = testNeg[1]

    for i in tqdm(range(len(testUser))):#用i遍历测试记录数
        target = testItem[i][0] #testItem[i]序列的第一个元素是i这条记录的ground truth item，后面的99个是负实例item
        this_test_u = torch.tensor(testUser[i]).to(device)
        this_test_v = torch.tensor(testItem[i]).to(device)
        predict = model.forward(this_test_u, this_test_v, device)
        predict = predict.data.cpu().numpy()
        item_score_dict = {}# { 被测试item的id : 相应预测分 }

        temp_ = list(range(len(testItem[i])))
        random.shuffle(temp_)
        for j in temp_:
            item = testItem[i][j]
            item_score_dict[item] = predict[j]

        ranklist1 = heapq.nlargest(1, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr1 = getHitRatio(ranklist1, target)
        tmp_NDCG1 = getNDCG(ranklist1, target)
        hr1.append(tmp_hr1)
        NDCG1.append(tmp_NDCG1)

        ranklist5 = heapq.nlargest(5, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr5 = getHitRatio(ranklist5, target)
        tmp_NDCG5 = getNDCG(ranklist5, target)
        hr5.append(tmp_hr5)
        NDCG5.append(tmp_NDCG5)

        ranklist10 = heapq.nlargest(10, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr10 = getHitRatio(ranklist10, target)
        tmp_NDCG10 = getNDCG(ranklist10, target)
        hr10.append(tmp_hr10)
        NDCG10.append(tmp_NDCG10)

        ranklist15 = heapq.nlargest(15, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr15 = getHitRatio(ranklist15, target)
        tmp_NDCG15 = getNDCG(ranklist15, target)
        hr15.append(tmp_hr15)
        NDCG15.append(tmp_NDCG15)

        ranklist20 = heapq.nlargest(20, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr20 = getHitRatio(ranklist20, target)
        tmp_NDCG20 = getNDCG(ranklist20, target)
        hr20.append(tmp_hr20)
        NDCG20.append(tmp_NDCG20)

    return np.mean(hr1), np.mean(NDCG1), np.mean(hr5), np.mean(NDCG5), np.mean(hr10), np.mean(NDCG10), np.mean(hr15), np.mean(NDCG15), np.mean(hr20), np.mean(NDCG20)

def evaluate_fast(model, device, testNeg, num_users, num_items, args):
    model.eval()
    def getHitRatio(ranklist, targetItem):
        for item in ranklist:
            if item == targetItem:
                return 1
        return 0
    def getNDCG(ranklist, targetItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == targetItem:
                return math.log(2) / math.log(i+2)
        return 0

    embed_u = []
    nodes_u = []
    count = 0
    for user in tqdm(range(num_users)):
        nodes_u.append(user)
        count += 1
        if count == 128:
            batch_nodes_u = torch.LongTensor(nodes_u).to(device)
            u_embeds = model.get_u_embeddings(batch_nodes_u, device)
            for kkk in range(len(u_embeds)):
                embed_u.append(u_embeds[kkk].data.cpu())
            count = 0
            nodes_u = []

    if nodes_u != []:
        if len(nodes_u) == 1:
            batch_nodes_u = torch.LongTensor(nodes_u * 2).to(device)
            u_embeds = model.get_u_embeddings(batch_nodes_u, device)
            embed_u.append(u_embeds[0].data.cpu())
        else:
            batch_nodes_u = torch.LongTensor(nodes_u).to(device)
            u_embeds = model.get_u_embeddings(batch_nodes_u, device)
            for kkk in range(len(u_embeds)):
                embed_u.append(u_embeds[kkk].data.cpu())

    embed_v = []
    nodes_v = []
    count = 0
    for item in tqdm(range(num_items)):
        nodes_v.append(item)
        count += 1
        if count == 128:
            batch_nodes_v = torch.LongTensor(nodes_v).to(device)
            v_embeds = model.get_v_embeddings(batch_nodes_v, device)
            for kkk in range(len(v_embeds)):
                embed_v.append(v_embeds[kkk].data.cpu())
            count = 0
            nodes_v = []

    if nodes_v != []:
        if len(nodes_v) == 1:
            batch_nodes_v = torch.LongTensor(nodes_v * 2).to(device)
            v_embeds = model.get_v_embeddings(batch_nodes_v, device)
            embed_v.append(v_embeds[0].data.cpu())
        else:
            batch_nodes_v = torch.LongTensor(nodes_v).to(device)
            v_embeds = model.get_v_embeddings(batch_nodes_v, device)
            for kkk in range(len(v_embeds)):
                embed_v.append(v_embeds[kkk].data.cpu())

    embed_u = torch.stack(embed_u, dim=0)
    embed_v = torch.stack(embed_v, dim=0)

    hr1 = []
    NDCG1 = []
    hr5 = []
    NDCG5 = []
    hr10 = []
    NDCG10 = []
    hr15 = []
    NDCG15 = []
    hr20 = []
    NDCG20 = []

    testUser = testNeg[0]
    testItem = testNeg[1]
    testRating = testNeg[2]

    tmp_pred = []
    tmp_target = []

    for i in tqdm(range(len(testUser))):
        target = testItem[i][0]
        if args.atanh:
            predict = 0.5 + (1 / (2 * args.hash_code_length)) * torch.sum(torch.mul(embed_u[testUser[i]], embed_v[testItem[i]]), dim=1)
            predict = predict.data.cpu().numpy()
        else:
            predict = (torch.cosine_similarity(embed_u[testUser[i]], embed_v[testItem[i]]) + 1) / 2

        #RMSE,MAE
        tmp_pred.append(predict[0] * 5)
        tmp_target.append(testRating[i][0])

        item_score_dict = {}


        temp_ = list(range(len(testItem[i])))
        random.shuffle(temp_)
        for j in temp_:
            item = testItem[i][j]
            item_score_dict[item] = predict[j]

        ranklist1 = heapq.nlargest(1, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr1 = getHitRatio(ranklist1, target)
        tmp_NDCG1 = getNDCG(ranklist1, target)
        hr1.append(tmp_hr1)
        NDCG1.append(tmp_NDCG1)

        ranklist5 = heapq.nlargest(5, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr5 = getHitRatio(ranklist5, target)
        tmp_NDCG5 = getNDCG(ranklist5, target)
        hr5.append(tmp_hr5)
        NDCG5.append(tmp_NDCG5)

        ranklist10 = heapq.nlargest(10, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr10 = getHitRatio(ranklist10, target)
        tmp_NDCG10 = getNDCG(ranklist10, target)
        hr10.append(tmp_hr10)
        NDCG10.append(tmp_NDCG10)

        ranklist15 = heapq.nlargest(15, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr15 = getHitRatio(ranklist15, target)
        tmp_NDCG15 = getNDCG(ranklist15, target)
        hr15.append(tmp_hr15)
        NDCG15.append(tmp_NDCG15)

        ranklist20 = heapq.nlargest(20, item_score_dict, key=item_score_dict.get)  # topK个测试分最高的item的id
        tmp_hr20 = getHitRatio(ranklist20, target)
        tmp_NDCG20 = getNDCG(ranklist20, target)
        hr20.append(tmp_hr20)
        NDCG20.append(tmp_NDCG20)

    #RMSE, MAE
    try:
        rmse = sqrt(mean_squared_error(tmp_pred, tmp_target))
        mae = mean_absolute_error(tmp_pred, tmp_target)
    except:
        rmse = 9999999999
        mae = 9999999999

    return np.mean(hr1), np.mean(NDCG1), np.mean(hr5), np.mean(NDCG5), np.mean(hr10), np.mean(NDCG10), np.mean(hr15), np.mean(NDCG15), np.mean(hr20), np.mean(NDCG20), rmse, mae

def interest_attention_visualize():
    args = Config()
    dataset = args.dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    use_cuda = False
    if torch.cuda.is_available() and args.use_cuda:
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    if args.pickle:
        path_data = "data/" + args.dataset + ".pickle"
        with open(path_data, "rb") as f:
            history_u_lists, history_ur_lists, history_ut_lists, \
            history_v_lists, history_vr_lists, history_vt_lists, \
            history_va_lists, \
            social_adj_lists, ratings_list, history_af_lists, \
            train_u, train_v, train_r, train_a, test_u, test_v, test_r, testNeg = pickle.load(f)
        print(args.dataset, "dataset loaded.")
    else:
        ds = DataSet(args.rating_filename, args.item_attr_filename, args.friend_filename)
        history_u_lists, history_ur_lists, history_ut_lists, \
        history_v_lists, history_vr_lists, history_vt_lists, \
        history_va_lists, \
        social_adj_lists, ratings_list, history_af_lists, \
        train_u, train_v, train_r, train_a, test_u, test_v, test_r, testNeg = ds.get_data()
        print(args.dataset, "dataset loaded.")


    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r) / max(ratings_list)
                                              )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                               drop_last=True)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    index_set = set()
    for i in list(history_vt_lists.values()):
        index_set.update(i)
    for i in list(history_ut_lists.values()):
        index_set.update(i)
    num_index = index_set.__len__()  # index的个数

    attr_set = set()
    for i in list(history_va_lists.values()):
        attr_set.update(i)
    num_attr = attr_set.__len__()  # index的个数


    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    t2e = nn.Embedding(num_index, embed_dim).to(device)
    a2e = nn.Embedding(num_attr, embed_dim).to(device)
    f2e = nn.Embedding(args.famous_labels_num, embed_dim).to(device)

    # item model:  attr -> item
    agg_av_history = VA_Aggregator(v2e, r2e, u2e, a2e, f2e, embed_dim, cuda=device, va=True)
    enc_av_history = VA_Encoder(v2e, embed_dim, history_va_lists, history_af_lists, agg_av_history, cuda=device,
                                va=True)

    # item model:  user -> item
    agg_uv_history = UV_Aggregator(v2e, r2e, u2e, t2e, embed_dim, cuda=device, uv=False)
    enc_uv_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, history_vt_lists, agg_uv_history,
                                cuda=device, uv=False)

    # item model:  item merge
    item_merge = Merge(embed_dim)
    enc_item = Merge_Encoder(item_merge, enc_av_history, enc_uv_history)

    # user model:  item -> user
    agg_vu_history = UV_Aggregator(v2e, r2e, u2e, t2e, embed_dim, cuda=device, uv=True)
    enc_vu_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, history_ut_lists, agg_vu_history,
                                cuda=device, uv=True)

    # user model:  user -> user
    agg_u_social = Social_Aggregator(u2e, embed_dim, cuda=device)
    enc_user_social = Social_Encoder(enc_vu_history, embed_dim, social_adj_lists, agg_u_social, cuda=device)

    # model
    if args.social:
        if args.attr:
            graphrec = GraphRec(enc_user_social, enc_item, r2e).to(device)
        else:
            graphrec = GraphRec(enc_user_social, enc_uv_history, r2e).to(device)
    else:
        if args.attr:
            graphrec = GraphRec(enc_vu_history, enc_item, r2e).to(device)
        else:
            graphrec = GraphRec(enc_vu_history, enc_uv_history, r2e).to(device)
    optimizer = torch.optim.Adam(graphrec.parameters(), lr=args.lr)

    model_load_filename = "save/kuai_big_HCL128_RS12_ED256_BS128_SMTrue_TMPTrue_SCLTrue_INTboth_ATHTrue_ATTTrue/kuai_big#33.model"
    graphrec.load_state_dict(torch.load(model_load_filename))

    graphrec.eval()
    graphrec.enc_u.forward(torch.tensor([60]).to(device))

def main():

    args = Config()
    dataset = args.dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    use_cuda = False
    if torch.cuda.is_available() and args.use_cuda:
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    if args.pickle:
        path_data = "data/" + args.dataset + ".pickle"
        with open(path_data, "rb") as f:
            history_u_lists, history_ur_lists, history_ut_lists, \
            history_v_lists, history_vr_lists, history_vt_lists, \
            history_va_lists, \
            social_adj_lists, ratings_list, history_af_lists, \
            train_u, train_v, train_r, train_a, test_u, test_v, test_r, testNeg = pickle.load(f)
        print(args.dataset, "dataset loaded.")
    else:
        ds = DataSet(args.rating_filename, args.item_attr_filename, args.friend_filename)
        history_u_lists, history_ur_lists, history_ut_lists, \
        history_v_lists, history_vr_lists, history_vt_lists, \
        history_va_lists, \
        social_adj_lists, ratings_list, history_af_lists, \
        train_u, train_v, train_r, train_a, test_u, test_v, test_r, testNeg = ds.get_data()
        print(args.dataset, "dataset loaded.")

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r) / max(ratings_list)
                                              )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)


    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    index_set = set()
    for i in list(history_vt_lists.values()):
        index_set.update(i)
    for i in list(history_ut_lists.values()):
        index_set.update(i)
    num_index = index_set.__len__()

    attr_set = set()
    for i in list(history_va_lists.values()):
        attr_set.update(i)
    num_attr = attr_set.__len__()


    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    t2e = nn.Embedding(num_index, embed_dim).to(device)
    a2e = nn.Embedding(num_attr, embed_dim).to(device)
    f2e = nn.Embedding(args.famous_labels_num, embed_dim).to(device)

    # item model:  attr -> item
    agg_av_history = VA_Aggregator(v2e, r2e, u2e, a2e, f2e, embed_dim, cuda=device, va=True)
    enc_av_history = VA_Encoder(v2e, embed_dim, history_va_lists, history_af_lists, agg_av_history, cuda=device, va=True)

    # item model:  user -> item
    agg_uv_history = UV_Aggregator(v2e, r2e, u2e, t2e, embed_dim, cuda=device, uv=False)
    enc_uv_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, history_vt_lists, agg_uv_history,
                                cuda=device, uv=False)

    # item model:  item merge
    item_merge = Merge(embed_dim)
    enc_item = Merge_Encoder(item_merge, enc_av_history, enc_uv_history)

    # user model:  item -> user
    agg_vu_history = UV_Aggregator(v2e, r2e, u2e, t2e, embed_dim, cuda=device, uv=True)
    enc_vu_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, history_ut_lists, agg_vu_history,
                                cuda=device, uv=True)

    # user model:  user -> user
    agg_u_social = Social_Aggregator(u2e, embed_dim, cuda=device)
    enc_user_social = Social_Encoder(enc_vu_history, embed_dim, social_adj_lists, agg_u_social, cuda=device)

    # model
    if args.social:
        if args.attr:
            graphrec = GraphRec(enc_user_social, enc_item, r2e).to(device)
        else:
            graphrec = GraphRec(enc_user_social, enc_uv_history, r2e).to(device)
    else:
        if args.attr:
            graphrec = GraphRec(enc_vu_history, enc_item, r2e).to(device)
        else:
            graphrec = GraphRec(enc_vu_history, enc_uv_history, r2e).to(device)
    optimizer = torch.optim.Adam(graphrec.parameters(), lr=args.lr)

    dir_path = "result/" + str(datetime.datetime.now()).split()[0]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    result_filename = dir_path + "/" + str(dataset) \
                    + "_HCL" + str(args.hash_code_length) \
                    + "_RS" + str(args.random_seed) \
                    + "_ED" + str(args.embed_dim) \
                    + "_BS" + str(args.batch_size) \
                    + "_SM" + str(args.sample_model) \
                    + "_TMP" + str(args.temporal) \
                    + "_SCL" + str(args.social) \
                    + "_INT" + str(args.long_short) \
                    + "_ATH" + str(args.atanh) \
                    + "_ATT" + str(args.attr) \
                    + "_BAL" + str(args.balance) \
                    + "_DEC" + str(args.decorrelation) \
                    + str(int(time.time())) \
                    + ".log"

    save_path = "save/" + str(dataset) \
                + "_HCL" + str(args.hash_code_length) \
                + "_RS" + str(args.random_seed) \
                + "_ED" + str(args.embed_dim) \
                + "_BS" + str(args.batch_size) \
                + "_SM" + str(args.sample_model) \
                + "_TMP" + str(args.temporal) \
                + "_SCL" + str(args.social) \
                + "_INT" + str(args.long_short) \
                + "_ATH" + str(args.atanh) \
                + "_ATT" + str(args.attr) \
                + "_BAL" + str(args.balance) \
                + "_DEC" + str(args.decorrelation)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    best_hr1 = 0
    best_hr1_epoch = 0
    best_hr5 = 0
    best_hr5_epoch = 0
    best_hr10 = 0
    best_hr10_epoch = 0
    best_hr15 = 0
    best_hr15_epoch = 0
    best_hr20 = 0
    best_hr20_epoch = 0
    best_ndcg1 = 0
    best_ndcg1_epoch = 0
    best_ndcg5 = 0
    best_ndcg5_epoch = 0
    best_ndcg10 = 0
    best_ndcg10_epoch = 0
    best_ndcg15 = 0
    best_ndcg15_epoch = 0
    best_ndcg20 = 0
    best_ndcg20_epoch = 0

    best_rmse = 9999.0
    best_rmse_epoch = 0
    best_mae = 9999.0
    best_mae_epoch = 0

    best_epoch_loss = 99999999
    best_epoch_loss_epoch = 0

    pre_epoch_loss = 99999

    epoch_start = 1

    best_filename = save_path + "/" + str(dataset) + "_best.pickle"
    if os.path.exists(best_filename) and args.load_model:
        with open(best_filename, "rb") as f:
            best_hr1, best_hr5, best_hr10, best_hr15, best_hr20, \
            best_ndcg1, best_ndcg5, best_ndcg10, best_ndcg15, best_ndcg20, \
            best_hr1_epoch, best_hr5_epoch, best_hr10_epoch, best_hr15_epoch, best_hr20_epoch, \
            best_ndcg1_epoch, best_ndcg5_epoch, best_ndcg10_epoch, best_ndcg15_epoch, best_ndcg20_epoch, best_rmse, best_mae, \
            best_epoch_loss, best_epoch_loss_epoch, epoch_start = pickle.load(f)
        if args.given_epoch != -1 and args.given_epoch < epoch_start:
            epoch_start = args.given_epoch
        model_load_filename = save_path + "/" + str(dataset) + "#" + str(epoch_start) + '.model'
        graphrec.load_state_dict(torch.load(model_load_filename))

        print("load model success.")
        print("start epoch:", epoch_start)
        print("best_HR1:", best_hr1, "\tbest_NDCG1:", best_ndcg1)
        print("best_HR5:", best_hr5, "\tbest_NDCG5:", best_ndcg5)
        print("best_HR10:", best_hr10, "\tNDCG10:", best_ndcg10)
        print("best_HR15:", best_hr15, "\tNDCG15:", best_ndcg15)
        print("best_HR20:", best_hr20, "\tNDCG20:", best_ndcg20)
        print("best_RMSE:", best_rmse, "\tMAE:", best_mae)
        print("best_epoch loss:", best_epoch_loss)

    continue_epoch_num = 1
    for epoch in range(epoch_start, args.epochs + 1):

        print("epoch:", epoch, "lr:", optimizer.param_groups[0]['lr'])
        print("batch total num:", int(len(train_u) / args.batch_size))

        epoch_loss = train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        delta_loss = epoch_loss - pre_epoch_loss
        pre_epoch_loss = epoch_loss

        hr1, ndcg1, hr5, ndcg5, hr10, ndcg10, hr15, ndcg15, hr20, ndcg20, rmse, mae = evaluate_fast(graphrec, device, testNeg, num_users, num_items, args)
        print("epoch:", epoch)
        print("HR1:", hr1, "\tNDCG1:", ndcg1)
        print("HR5:", hr5, "\tNDCG5:", ndcg5)
        print("HR10:", hr10, "\tNDCG10:", ndcg10)
        print("HR15:", hr15, "\tNDCG15:", ndcg15)
        print("HR20:", hr20, "\tNDCG20:", ndcg20)
        print("RMSE:", rmse, "\tMAE:", mae)
        print("epoch loss:", epoch_loss)
        print("delta loss:", delta_loss)


        if hr1 > best_hr1:
            best_hr1 = hr1
            best_hr1_epoch = epoch
        if hr5 > best_hr5:
            best_hr5 = hr5
            best_hr5_epoch = epoch
        if hr10 > best_hr10:
            best_hr10 = hr10
            best_hr10_epoch = epoch
        if hr15 > best_hr15:
            best_hr15 = hr15
            best_hr15_epoch = epoch
        if hr20 > best_hr20:
            best_hr20 = hr20
            best_hr20_epoch = epoch
        if ndcg1 > best_ndcg1:
            best_ndcg1 = ndcg1
            best_ndcg1_epoch = epoch
        if ndcg5 > best_ndcg5:
            best_ndcg5 = ndcg5
            best_ndcg5_epoch = epoch
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            best_ndcg10_epoch = epoch
        if ndcg15 > best_ndcg15:
            best_ndcg15 = ndcg15
            best_ndcg15_epoch = epoch
        if ndcg20 > best_ndcg20:
            best_ndcg20 = ndcg20
            best_ndcg20_epoch = epoch
        if best_rmse > rmse:
            best_rmse = rmse
            best_rmse_epoch = epoch
        if best_mae > mae:
            best_mae = mae
            best_mae_epoch = epoch
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch_loss_epoch = epoch

        print("best_HR1:", best_hr1, "\tbest_NDCG1:", best_ndcg1)
        print("best_HR5:", best_hr5, "\tbest_NDCG5:", best_ndcg5)
        print("best_HR10:", best_hr10, "\tbest_NDCG10:", best_ndcg10)
        print("best_HR15:", best_hr15, "\tbest_NDCG15:", best_ndcg15)
        print("best_HR20:", best_hr20, "\tbest_NDCG20:", best_ndcg20)
        print("best_rmse:", best_rmse, "\tbest_mae:", best_mae)
        print("best_epoch loss:", best_epoch_loss)
        print(best_hr1_epoch, best_hr5_epoch, best_hr10_epoch, best_hr15_epoch, best_hr20_epoch, \
              best_ndcg1_epoch, best_ndcg5_epoch, best_ndcg10_epoch, best_ndcg15_epoch, best_ndcg20_epoch, best_rmse_epoch, best_mae_epoch, \
              best_epoch_loss_epoch)

        with open(result_filename, 'a') as fw:
            line = "Epoch_" + str(epoch) \
                   + '\t' + str(hr1) \
                   + '\t' + str(ndcg1) \
                   + '\t' + str(hr5) \
                   + '\t' + str(ndcg5) \
                   + '\t' + str(hr10) \
                   + '\t' + str(ndcg10) \
                   + '\t' + str(hr15) \
                   + '\t' + str(ndcg15) \
                   + '\t' + str(hr20) \
                   + '\t' + str(ndcg20) \
                   + '\t' + str(rmse) \
                   + '\t' + str(mae) \
                   + '\t' + str(epoch_loss) \
                   + '\n'
            fw.write(line)
            line = "Epoch_" + str(epoch) \
                   + '\t' + str(best_hr1) \
                   + '\t' + str(best_ndcg1) \
                   + '\t' + str(best_hr5) \
                   + '\t' + str(best_ndcg5) \
                   + '\t' + str(best_hr10) \
                   + '\t' + str(best_ndcg10) \
                   + '\t' + str(best_hr15) \
                   + '\t' + str(best_ndcg15) \
                   + '\t' + str(best_hr20) \
                   + '\t' + str(best_ndcg20) \
                   + '\t' + str(best_rmse) \
                   + '\t' + str(best_mae) \
                   + '\t' + str(best_epoch_loss) \
                   + '\n'
            fw.write(line)
            line = "Epoch_" + str(epoch) \
                   + '\t' + str(best_hr1_epoch) \
                   + '\t' + str(best_ndcg1_epoch) \
                   + '\t' + str(best_hr5_epoch) \
                   + '\t' + str(best_ndcg5_epoch) \
                   + '\t' + str(best_hr10_epoch) \
                   + '\t' + str(best_ndcg10_epoch) \
                   + '\t' + str(best_hr15_epoch) \
                   + '\t' + str(best_ndcg15_epoch) \
                   + '\t' + str(best_hr20_epoch) \
                   + '\t' + str(best_ndcg20_epoch) \
                   + '\t' + str(best_rmse_epoch) \
                   + '\t' + str(best_mae_epoch) \
                   + '\t' + str(best_epoch_loss_epoch) \
                   + '\t' + str(delta_loss) \
                   + '\n'
            fw.write(line)

        if args.dynamic_lr:
            best_epoches = [best_hr1_epoch, best_hr5_epoch, best_hr10_epoch, best_hr15_epoch, best_hr20_epoch,\
                            best_ndcg1_epoch, best_ndcg5_epoch, best_ndcg10_epoch, best_ndcg15_epoch, best_ndcg20_epoch]
            delta = epoch - max(best_epoches)
            if continue_epoch_num <= args.lr_adjust_patience:
                continue_epoch_num += 1
                for param_group in optimizer.param_groups:
                    if args.lr ** continue_epoch_num < args.lr_min:
                        param_group['lr'] = args.lr_min
                    else:
                        param_group['lr'] = args.lr ** continue_epoch_num
            elif delta >= args.lr_adjust_delta:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                continue_epoch_num = 1
            else:
                continue_epoch_num += 1
                for param_group in optimizer.param_groups:
                    if args.lr ** continue_epoch_num < args.lr_min:
                        param_group['lr'] = args.lr_min
                    else:
                        param_group['lr'] = args.lr ** continue_epoch_num


        model_save_filename = save_path + "/" + str(dataset) + "#" + str(epoch) + '.model'
        torch.save(graphrec.state_dict(), model_save_filename)


        with open(best_filename, "wb") as f:
            pickle.dump((best_hr1, best_hr5, best_hr10, best_hr15, best_hr20, \
                         best_ndcg1, best_ndcg5, best_ndcg10, best_ndcg15, best_ndcg20, \
                         best_hr1_epoch, best_hr5_epoch, best_hr10_epoch, best_hr15_epoch, best_hr20_epoch, \
                         best_ndcg1_epoch, best_ndcg5_epoch, best_ndcg10_epoch, best_ndcg15_epoch, best_ndcg20_epoch, \
                         best_rmse_epoch, best_mae_epoch, best_epoch_loss, best_epoch_loss_epoch, epoch), f)


if __name__ == "__main__":
    main()
