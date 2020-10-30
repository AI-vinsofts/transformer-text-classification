from modules import *
import numpy as np

import tensorflow as tf


class TransformerDecoder:
    def __init__(self, is_training=True, args=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=(None, args.maxlen))

            if is_training:
                self.y = tf.placeholder(tf.int32, shape=(None, 1))
            if args.embeddings_path:
                print("Using word2vec embedding")
            # Decoder
            with tf.variable_scope("decoder"):
                self.dec, self.lookup_table = embedding(self.x,
                                                        vocab_size=args.vocab_size,
                                                        num_units=args.hidden_units,
                                                        scale=True,
                                                        zero_pad=False,
                                                        scope="dec_embed",
                                                        init_values=np.load(
                                                            args.embeddings_path) if args.embeddings_path else None)
                if args.sinusoid:
                    self.dec += positional_encoding(self.x,
                                                    num_units=args.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0),
                                                  [tf.shape(self.x)[0], 1]),
                                          vocab_size=args.maxlen,
                                          num_units=args.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe")[0]

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=args.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=args.hidden_units,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="self_attention")

                        self.dec = feedforward(self.dec, num_units=[4 * args.hidden_units, args.hidden_units])
                self.proj = tf.get_variable("proj", [args.num_classes, args.hidden_units * args.maxlen])
                self.logits = tf.matmul(tf.reshape(self.dec, [-1, args.hidden_units * args.maxlen]),
                                        self.proj, transpose_b=True)

            # self.logits = tf.layers.dense(self.dec, len(word2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            if is_training:
                one_hot_depth = args.num_classes
                # Loss
                self.labels = tf.reshape(
                    tf.stop_gradient(tf.one_hot(self.y, depth=one_hot_depth)), [-1, one_hot_depth])
                if args.class_weights is not None:
                    class_weights = tf.constant(np.array(args.class_weights, dtype=np.float32))
                    weight_map = tf.multiply(self.labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
                if args.class_weights is not None:
                    weighted_loss = tf.multiply(self.loss, weight_map)
                    self.mean_loss = tf.reduce_mean(weighted_loss)
                else:
                    self.mean_loss = tf.reduce_mean(self.loss)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter(args.logdir, self.graph)
