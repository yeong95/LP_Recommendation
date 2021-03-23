import sys
import os
from tqdm import tqdm
import pandas as pd
import argparse
# os.chdir(r'C:\Users\CHOYEONGKYU\Desktop\Sequential_Recommendation\models\GRU4Rec')
sys.path.append('../..')
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import tensorflow as tf
import numpy as np
from model_GRU4rec import GRU4Rec
from make_datasets import make_datasets
from DataInput import DataIterator
from evaluation import SortItemsbyScore,Metric_HR,Metric_MRR,Metric_NDCG


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--dataset', default='lpoint', type=str)
    parser.add_argument('--train_dir', default='train', type=str)
    parser.add_argument('--num_epochs', type=int, default=201)
    parser.add_argument('--emb_size', type=int, default=50)
    parser.add_argument('--len_Seq', type=int, default=2)
    parser.add_argument('--len_Tag', type=int, default=1)
    parser.add_argument('--len_Pred', type=int, default=1)
    parser.add_argument('--neg_sample', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--keep_prob', type=float, default=0.7)
    parser.add_argument('--loss_fun', type=str, default='top1')
    parser.add_argument('--l2_lambda', type=float, default=1e-6)
    return parser.parse_args()

if __name__ == '__main__':

    # Get Params
    args = parse_args()
    len_Seq = args.len_Seq   #序列的长度
    len_Tag = args.len_Tag   #训练时目标的长度
    len_Pred = args.len_Pred     #预测时目标的长度
    batch_size = args.batch_size
    emb_size = args.emb_size
    neg_sample = args.neg_sample
    keep_prob = args.keep_prob
    layers = args.layers
    loss_fun = args.loss_fun
    l2_lambda = args.l2_lambda
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # make datasets

    print('==> make datasets <==')
    # file_path = '/content/drive/MyDrive/Sequential_Recommendation/datasets/제6회 L.POINT Big Data Competition-분석용데이터-02.거래 정보.csv'
    file_path = 'C:\\Users\\CHOYEONGKYU\\Desktop\\Sequential_Recommendation\\datasets\\RAW_Transaction_data.csv'
    data = pd.read_csv(file_path, encoding='utf-8')
    removed_data = data[~(data.pd_c == 'unknown')]
    removed_data = removed_data[~(removed_data.buy_ct == 0)]
    removed_data = removed_data.sort_values(by=['trans_id','trans_seq'])
    removed_data = removed_data.astype({'pd_c':'int'})
    new_data = removed_data[['trans_id','trans_seq', 'pd_c']]
    new_data.rename(columns={'trans_id':'user','trans_seq':'timestamps','pd_c':'item'}, inplace=True)
    d_train, d_test, d_info = make_datasets(new_data, len_Seq, len_Tag, len_Pred)
    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator

    trainIterator = DataIterator('train',d_train, batch_size, neg_sample,
                                 all_items, items_usr_clicked, shuffle=True)
    testIterator = DataIterator('test',d_test, batch_size,  shuffle=False)

    # Define Model

    model = GRU4Rec(emb_size, num_usr, num_item, len_Seq, 1, layers)
    loss = model.loss
    input_Seq = model.input_Seq
    input_NegT = model.input_NegT
    input_PosT = model.input_PosT
    input_keepprob = model.input_keepprob
    score_pred = model.predict()

    # Define Optimizer

    global_step = tf.Variable(0, trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # log
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'GRU4Rec_log.txt'), 'w')
    # Training and test for every 20 epoch 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, num_epochs):

            #train
            cost_list = []
            for train_input in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
                batch_usr, batch_seq, batch_pos, batch_neg = train_input
                feed_dict = {input_Seq: batch_seq,input_PosT: batch_pos
                    , input_NegT: batch_neg,input_keepprob: keep_prob}
                _, step, cost= sess.run([train_op, global_step, loss],feed_dict)
                cost_list += list(cost)
            mean_cost = np.mean(cost_list)
            #saver.save(sess, FLAGS.save_path)

            # test
            if epoch % 20 == 0:
                pred_list = []
                next_list = []
                user_list = []

                for test_input in testIterator:
                    batch_usr, batch_seq, batch_pos, batch_neg = test_input
                    feed_dict = {input_Seq: batch_seq, input_keepprob: 1.0}
                    pred = sess.run(score_pred, feed_dict)  # , options=options, run_metadata=run_metadata)

                    pred_list += pred.tolist()
                    next_list += list(batch_pos)
                    user_list += list(batch_usr)

                sorted_items,sorted_score = SortItemsbyScore(all_items,pred_list,reverse=True,remove_hist=False
                                                        ,usr=user_list,usrclick=items_usr_clicked)
            
                hr10 = Metric_HR(10, next_list, sorted_items)
                NDCG = Metric_NDCG(next_list, sorted_items)
                print(" epoch {}, mean_loss{:g}, test HR@10: {:g} NDCG@10: {:g}"
                    .format(epoch + 1, mean_cost, hr10, NDCG))
                f.write(str(epoch)+'epoch:'+' ' + '(NDCG@10,HR@10)' + ' ' + '('+ str(NDCG) + ',' + str(hr10) + ')' + '\n')
                f.flush()
        f.close()