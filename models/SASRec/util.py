import sys
import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('/content/gdrive/My Drive/SASRec/data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
    

def lpoint_data_partition():
    
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    file_path = '/content/gdrive/My Drive/Sequential_Recommendation/datasets/구매_상품정보.csv'
    data = pd.read_csv(file_path, encoding='cp949') 
    
    data.rename(columns={'TRANS_ID':'user','TRANS_SEQ':'timestamps','PD_C':'item'}, inplace=True)
    
    p = data.groupby('item')['user'].count().reset_index().rename(columns={'user':'item_count'})
    data = pd.merge(data,p,how='left',on='item')
    data = data[data['item_count'] > 5].drop(['item_count'],axis=1)
    
    data = data[['user','timestamps', 'item']]
    data = data.sort_values(by=['user','timestamps'])
    data.reset_index(inplace=True, drop=True)
    # ReMap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1)))
    item_map[0] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[0] = 0
    all_user_count = len(user_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])
    
    for u in np.unique(data.user.values):  
      User[u].append(data[data.user == u].item.values.tolist())
      User[u] = User[u][0]

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, all_user_count, all_item_count]


def evaluate(model, dataset, args, sess):
    d_train, d_test, d_info = copy.deepcopy(dataset)
    usernum, itemnum, items_usr_clicked, _, _ = d_info

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    user_list = [*items_usr_clicked]

    if usernum>15000:
        users = random.sample(user_list, 10000)
    else:
        users = user_list
        
    for u in users:
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(items_usr_clicked[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(items_usr_clicked[u][:-1])
        rated.add(0)
        item_idx = [items_usr_clicked[u][-1]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.'),
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
    
def all_item_evaluate(model, dataset,all_items, args, sess):
    d_train, d_test, d_info = copy.deepcopy(dataset)
    usernum, itemnum, items_usr_clicked, _, _ = d_info
    item_set = all_items
    
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    user_list = [*items_usr_clicked]

    if usernum>15000:
        users = random.sample(user_list, 10000)
    else:
        users = user_list
        
    for u in users:
        itemset = item_set.copy()
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(items_usr_clicked[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(items_usr_clicked[u][:-1])
        rated.add(0)
        
        item_idx = [items_usr_clicked[u][-1]]
        itemset.remove(item_idx[0])
        item_idx = item_idx + itemset

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.'),
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
