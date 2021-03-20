import sys
import os
from tqdm import tqdm
import pandas as pd
import argparse
sys.path.append("..")
sys.path.append("../..")

os.environ["CUDA_VISIBLE_DEVICES"]='0'
import tensorflow as tf
import numpy as np

from model import Model
from make_datasets_SASRec import make_datasets
from DataInput_SASRec import DataIterator
from evaluation import SortItemsbyScore,Metric_HR,Metric_MRR
from modules import *
from util import *
import time


def parse_args():
    parser = argparse.ArgumentParser(description='SASRec')
    parser.add_argument('--dataset', default='lpoint', type=str)
    parser.add_argument('--train_dir', default='train', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_len', default=50, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    return parser.parse_args()




if __name__ == '__main__':

    # Get Params
    args = parse_args()

    # make datasets

    file_path = '/content/drive/MyDrive/Sequential_Recommendation/datasets/제6회 L.POINT Big Data Competition-분석용데이터-02.거래 정보.csv'
    data = pd.read_csv(file_path, encoding='utf-8')
    removed_data = data[~(data.pd_c == 'unknown')]
    removed_data = removed_data[~(removed_data.buy_ct == 0)]
    removed_data = removed_data.sort_values(by=['trans_id','trans_seq'])
    removed_data = removed_data.astype({'pd_c':'int'})
    new_data = removed_data[['trans_id','trans_seq', 'pd_c']]
    new_data.rename(columns={'trans_id':'user','trans_seq':'timestamps','pd_c':'item'}, inplace=True)
    item_dataset = make_datasets(new_data, args.max_len)
    d_train, d_test, d_info = item_dataset


    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator
    trainIterator = DataIterator('train',d_train, args.batch_size, args.max_len,
                                 all_items, items_usr_clicked, shuffle=True)

    # Define Model
    model = Model(usernum=num_usr,
                    itemnum=num_item,
                    args=args)
    # log
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'SASRec_log.txt'), 'w')

    # train SASRec Model
    T = 0.0
    t0 = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, args.num_epochs):

            #train
            cost_list = []
            for train_input in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
                batch_usr, batch_seq, batch_pos, batch_neg = train_input
                auc, cost, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: batch_usr, model.input_seq: batch_seq,
                                    model.pos: batch_pos, model.neg: batch_neg, model.is_training: True})
                cost_list.append(cost)
            mean_cost = np.mean(cost_list)

            #saver.save(sess, FLAGS.save_path)

            if epoch % 20 == 0:
              t1 = time.time() - t0
              T += t1
              t_test = all_item_evaluate(model, item_dataset, all_items, args, sess)
              print('epoch:%d, time: %f(s), test (NDCG@10: %.4f, HR@10: %.4f)' % ( epoch, T, t_test[0], t_test[1]))
              f.write(str(epoch)+'epoch:'+' ' + '(NDCG@10,HR@10)' + ' ' + str(t_test) + '\n')
              f.flush()

              t0 = time.time()
