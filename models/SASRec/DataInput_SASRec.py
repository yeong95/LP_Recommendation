import pandas as pd
import numpy as np
import random
import pickle
from tqdm import tqdm
from make_datasets import make_datasets


# trainIterator = DataIterator('train',d_train, batch_size, neg_sample,
#                                  all_items, items_usr_clicked, shuffle=True)

class DataIterator:

    def __init__(self,
                 mode,
                 data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = (self.datasize - self.idx) - (self.datasize - self.idx) % 3

        cur = self.data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u, seq, in zip(cur['user'],cur['seq']):
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set,len(seq)))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)
        
class CATE1_DataIterator:

    def __init__(self,
                 mode,
                 data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 all_cates = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.all_cates = all_cates
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = (self.datasize - self.idx) - (self.datasize - self.idx) % 3

        cur = self.data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u, seq, in zip(cur['user'],cur['seq']):
                user_cate_set = set(self.all_cates) - set(self.item_usr_clicked[u][:-1])
                a = np.random.choice(list(user_cate_set), len(seq)-1)
                user_item_set = set(self.all_items) - set([self.item_usr_clicked[u][-1]])
                b = np.random.choice(list(user_item_set),1)
                batch_neg.append(np.append(a,b))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)
class Cate_DataIterator:

    def __init__(self,
                 mode,
                 data,
                 cate1_data,
                 cate2_data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.cate1_data = cate1_data
        self.cate2_data = cate2_data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = (self.datasize - self.idx) - (self.datasize - self.idx) % 3

        cur = self.data.iloc[self.idx:self.idx+nums]
        cur1 = self.cate1_data.iloc[self.idx:self.idx+nums]
        cur2 = self.cate2_data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)
        
        batch_cate1_seq = []
        for seq in cur1['seq'].values:
            batch_cate1_seq.append(seq)
            
        batch_cate2_seq = []
        for seq in cur2['seq'].values:
            batch_cate2_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u, seq, in zip(cur['user'],cur['seq']):
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set,len(seq)))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_cate1_seq, batch_cate2_seq, batch_pos, batch_neg)
        
class Feature_DataIterator:

    def __init__(self,
                 mode,
                 data,
                 cate1_data,
                 cate2_data,
                 price_data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.cate1_data = cate1_data
        self.cate2_data = cate2_data
        self.price_data = price_data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = (self.datasize - self.idx) - (self.datasize - self.idx) % 3

        cur = self.data.iloc[self.idx:self.idx+nums]
        cur1 = self.cate1_data.iloc[self.idx:self.idx+nums]
        cur2 = self.cate2_data.iloc[self.idx:self.idx+nums]
        cur3 = self.price_data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)
        
        batch_cate1_seq = []
        for seq in cur1['seq'].values:
            batch_cate1_seq.append(seq)
            
        batch_cate2_seq = []
        for seq in cur2['seq'].values:
            batch_cate2_seq.append(seq)
            
        batch_price_seq = []
        for seq in cur3['seq'].values:
            batch_price_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u, seq, in zip(cur['user'],cur['seq']):
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set,len(seq)))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_cate1_seq, batch_cate2_seq, batch_price_seq, batch_pos, batch_neg)
        
class Price_DataIterator:

    def __init__(self,
                 mode,
                 data,
                 price_data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.price_data = price_data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = (self.datasize - self.idx) - (self.datasize - self.idx) % 3

        cur = self.data.iloc[self.idx:self.idx+nums]
        cur3 = self.price_data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)
            
        batch_price_seq = []
        for seq in cur3['seq'].values:
            batch_price_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u, seq, in zip(cur['user'],cur['seq']):
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set,len(seq)))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_price_seq, batch_pos, batch_neg)

if __name__ == '__main__':

    d_train, d_test, d_info = make_datasets(5, 3, 4)
    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator

    trainIterator = DataIterator('train', d_train, 21, 5,
                                 all_items, items_usr_clicked, shuffle=True)
    for epoch in range(6):
        for data in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
            batch_usr, batch_seq, batch_pos, batch_neg = data




