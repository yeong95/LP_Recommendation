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

def feature_evaluate(model, item_dataset, all_items, cate1_dataset, cate2_dataset, price_dataset, args, sess):
    d_train, d_test, d_info = copy.deepcopy(item_dataset)
    cate1_train, cate1_test, cate1_info = copy.deepcopy(cate1_dataset)
    cate2_train, cate2_test, cate2_info = copy.deepcopy(cate2_dataset)
    price_train, price_test, price_info = copy.deepcopy(price_dataset)
    usernum, itemnum, items_usr_clicked, _, _ = d_info
    num_usr, num_cate1, cate1_usr_clicked, _, cate1_map = cate1_info
    num_usr, num_cate2, cate2_usr_clicked, _, cate2_map = cate2_info
    num_usr, num_price, price_usr_clicked, _, price_map = price_info
    
    item_set = all_items

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    user_list = [*items_usr_clicked]

    if usernum>10000:
        users = random.sample(user_list, 10000)
    else:
        users = user_list
    
    for u in users:
        itemset = item_set.copy()
        seq = np.zeros([args.maxlen], dtype=np.int32)
        cate1_seq = np.zeros([args.maxlen], dtype=np.int32)
        cate2_seq = np.zeros([args.maxlen], dtype=np.int32)
        price_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for a,b,c,d in zip(reversed(items_usr_clicked[u][:-1]), reversed(cate1_usr_clicked[u][:-1]), reversed(cate2_usr_clicked[u][:-1]), reversed(price_usr_clicked[u][:-1])):
            seq[idx] = a
            cate1_seq[idx] = b
            cate2_seq[idx] = c
            price_seq[idx] = d
            idx -= 1
            if idx == -1: break
        rated = set(items_usr_clicked[u][:-1])
        rated.add(0)
        item_idx = [items_usr_clicked[u][-1]]
        itemset.remove(item_idx[0])
        item_idx = item_idx + itemset

        predictions = -model.predict(sess, [u], [seq], [cate1_seq], [cate2_seq], [price_seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.ranking:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user

def cate_evaluate(model, item_dataset, cate1_dataset, cate2_dataset, args, sess):
    d_train, d_test, d_info = copy.deepcopy(item_dataset)
    cate1_train, cate1_test, cate1_info = copy.deepcopy(cate1_dataset)
    cate2_train, cate2_test, cate2_info = copy.deepcopy(cate2_dataset)
    usernum, itemnum, items_usr_clicked, _, _ = d_info
    num_usr, num_cate1, cate1_usr_clicked, _, cate1_map = cate1_info
    num_usr, num_cate2, cate2_usr_clicked, _, cate2_map = cate2_info

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    user_list = [*items_usr_clicked]

    if usernum>10000:
        users = random.sample(user_list, 10000)
    else:
        users = user_list
        
    for u in users:
        seq = np.zeros([args.maxlen], dtype=np.int32)
        cate1_seq = np.zeros([args.maxlen], dtype=np.int32)
        cate2_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for x,y,z in zip(reversed(items_usr_clicked[u][:-1]), reversed(cate1_usr_clicked[u][:-1]), reversed(cate2_usr_clicked[u][:-1]              )):
            seq[idx] = x
            cate1_seq[idx] = y
            cate2_seq[idx] = z
            idx -= 1
            if idx == -1: break
        rated = set(items_usr_clicked[u][:-1])
        rated.add(0)
        item_idx = [items_usr_clicked[u][-1]]
        for _ in range(100):
            t = np.random.randint(1, itemnum )
            while t in rated: t = np.random.randint(1, itemnum )
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], [cate1_seq], [cate2_seq], item_idx)
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
    

    
def price_evaluate(model, item_dataset, price_dataset, args, sess):
    d_train, d_test, d_info = copy.deepcopy(item_dataset)
    price_train, price_test, price_info = copy.deepcopy(price_dataset)
    usernum, itemnum, items_usr_clicked, _, _ = d_info
    num_usr, num_price, price_usr_clicked, _, price_map = price_info

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    user_list = [*items_usr_clicked]

    if usernum>10000:
        users = random.sample(user_list, 10000)
    else:
        users = user_list
        
    for u in users:
        seq = np.zeros([args.maxlen], dtype=np.int32)
        price_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for x,y in zip(reversed(items_usr_clicked[u][:-1]), reversed(price_usr_clicked[u][:-1])):
            seq[idx] = x
            price_seq[idx] = y
            idx -= 1
            if idx == -1: break
        rated = set(items_usr_clicked[u][:-1])
        rated.add(0)
        item_idx = [items_usr_clicked[u][-1]]
        for _ in range(100):
            t = np.random.randint(1, itemnum )
            while t in rated: t = np.random.randint(1, itemnum )
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], [price_seq], item_idx)
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


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
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
