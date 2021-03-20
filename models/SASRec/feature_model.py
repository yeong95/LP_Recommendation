from modules import *


class Feature_Model():
    def __init__(self, usernum, itemnum, cate1num, cate2num, pricenum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.cate1_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.cate2_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.price_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.weight =tf.get_variable('weight_',
                           dtype=tf.float32,
                           initializer= tf.constant(0.8, dtype=tf.float32))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1) # 0인것은 0으로, 나머지는 1로 

        with tf.variable_scope("Item_SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum ,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="item_dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
            
        ### feature level self-attention
            
        with tf.variable_scope("Feature_SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.first_seq, cate1_emb_table = embedding(self.cate1_seq,
                                                 vocab_size=cate1num ,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="cate1_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
                                                 
            self.second_seq, cate2_emb_table = embedding(self.cate2_seq,
                                                 vocab_size=cate2num ,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="cate2_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
                                                 
            self.third_seq, price_emb_table = embedding(self.price_seq,
                                                 vocab_size=pricenum ,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="price_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
                                                 
            self.cate_seq = tf.concat([self.first_seq, self.second_seq, self.third_seq],2)   # [Batch, seq, hidden units*3]
            fc1 = fc_layer(self.cate_seq, 150, 'FC1', use_relu=False)
            fc1 = tf.reshape(fc1, [-1, tf.shape(fc1)[2]])
            fc1 = tf.nn.softmax(fc1, axis=-1)
            self.attention_fc1 = fc1 * tf.reshape(self.cate_seq, [-1, tf.shape(self.cate_seq)[2]])
            self.attention_fc1 = tf.reshape(self.attention_fc1, [tf.shape(self.cate_seq)[0], tf.shape(self.cate_seq)[1], -1])
            
            # Positional Encoding
            cate_t, cate_pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.attention_fc1)[1]), 0), [tf.shape(self.attention_fc1)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units*3,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="feature_dec_pos",
                reuse=reuse,
                with_t=True
            )
            
            self.attention_fc1 += cate_t
            
            self.attention_fc1 = tf.layers.dropout(self.attention_fc1,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
                                         
            cate_mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
            self.attention_fc1 *= cate_mask
            
            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("feature_num_blocks_%d" % i):

                    # Self-attention
                    self.attention_fc1 = multihead_attention(queries=normalize(self.attention_fc1),
                                                   keys=self.attention_fc1,
                                                   num_units=args.hidden_units*3,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="feature_self_attention")

                    # Feed forward
                    self.attention_fc1 = feedforward(normalize(self.attention_fc1), 
                                        num_units=[args.hidden_units*3, args.hidden_units*3], dropout_rate=args.dropout_rate,                                                   is_training=self.is_training)
                    self.attention_fc1 *= cate_mask

            self.attention_fc1 = normalize(self.attention_fc1)
        
        self.attention_fc1 = fc_layer(self.attention_fc1, 50, 'att_dimension_reduce', use_relu=False)    
        fc2 = self.seq * self.weight + self.attention_fc1 * (1 - self.weight) # [batch, maxlen, 50]
        # concat_seq = tf.concat([self.seq, self.attention_fc1], 2)  [batch, maxlen, 50+150]
        
        

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(fc2, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(itemnum))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, cate1_seq, cate2_seq, price_seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.cate1_seq: cate1_seq, self.cate2_seq: cate2_seq,
                         self.price_seq: price_seq, self.test_item: item_idx, self.is_training: False})