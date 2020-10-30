from __future__ import print_function
import argparse
import os

from data_load import next_batch, load_vocab, load_train_data
from modules import *
import pickle
import numpy as np
from models import TransformerDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='corpora/data')
    parser.add_argument('--embeddings_path', type=str, default=None)
    parser.add_argument('--vocab_path', type=str, default='./corpora/vocab.txt')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--logdir', type=str, default='./ckpt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_batches', type=int, default=30000)
    parser.add_argument('--save_every', type=int, default=3000)
    parser.add_argument('--maxlen', type=int, default=128)
    parser.add_argument('--weight_tying', type=int, default=0)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--weighted_loss', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=9)
    parser.add_argument('--sinusoid', type=int, default=0)

    args = parser.parse_args()
    word2idx, idx2word = load_vocab(args.vocab_path)
    X, class_weights = load_train_data(args.train_path)
    args.vocab_size = len(word2idx)
    if args.weighted_loss:
        args.class_weights = class_weights
    with open(os.path.join(args.logdir, "args.pkl"), 'wb') as f:
        pickle.dump(args, f)

    # Construct graph
    model = TransformerDecoder(is_training=True, args=args)
    print("Graph loaded")

    # Start session
    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        if ckpt:
            print("restoring from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        gs = sess.run(model.global_step)
        for step in range(args.num_batches):
            x_step, y_step = next_batch(X, args.batch_size, args.maxlen)
            [_, mean_loss, loss, preds] = sess.run([model.train_op, model.mean_loss, model.merged, model.preds],
                                                   feed_dict={
                                                       model.x: x_step,
                                                       model.y: y_step
                                                   })
            if step % 10 == 0:
                acc = np.count_nonzero((preds == y_step[:, 0]).astype(np.int)) / preds.shape[0]
                model.train_writer.add_summary(loss, gs + step)
                print("acc = {:.4f}".format(acc))
            print("step = {}/{}, loss = {:.4f}".format(step + 1, args.num_batches, mean_loss))
            if (step + 1) % args.save_every == 0 or step + 1 == args.num_batches:
                gs = sess.run(model.global_step)
                saver.save(sess, args.logdir + '/model_gs_%d' % (gs))

    print("Done")
