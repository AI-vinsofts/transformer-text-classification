from __future__ import print_function

import argparse
import pickle
import tensorflow as tf
import numpy as np
from data_load import load_vocab, basic_tokenizer
from models import TransformerDecoder
import codecs
import regex as re
from utils import url_marker
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

CLASS_NAMES = os.listdir("mydata/test")


def _format_line(line):
    line = re.sub(
        url_marker.WEB_URL_REGEX,
        "<link>", line)
    line = re.sub("[\.]+", ".", line)
    line = re.sub("[0-9]*\,[0-9]+", "<num>", line)
    line = re.sub("[0-9]*\.[0-9]+", "<num>", line)
    line = re.sub("[0-9]+", "<num>", line)
    line = re.sub("[\.\?\!]", " <eos> ", line)
    return basic_tokenizer(line)


def _classify(data):
    original_len = len(data)
    batch_size = min(args.max_samples, len(data) // saved_args.maxlen)
    if batch_size == 0:
        prime = np.array(data[:saved_args.maxlen])
        prime = np.atleast_2d(np.tile(prime, saved_args.maxlen // len(prime) + 1)[:saved_args.maxlen])
        # prime = np.atleast_2d(
        #     np.lib.pad(prime, [0, saved_args.maxlen - len(prime)], 'constant', constant_values=pad_idx))

    else:
        prime = data[:saved_args.maxlen * batch_size]
        prime = np.reshape(np.array(prime), [batch_size, saved_args.maxlen])
    preds, dec, proj = sess.run((softmax, model.dec, model.proj), feed_dict={model.x: prime})
    dec = dec[0].flatten()
    proj = proj[np.argmax(preds)]
    attns = np.sum(np.reshape(dec * proj, [saved_args.maxlen, saved_args.hidden_units]), 1)[:original_len]
    return np.argmax(preds)


def _compute_acc_for_class(input_dir, class_id):
    files = os.listdir(input_dir)
    total = 0.0
    correct = 0.0
    y_p = []
    for file in tqdm(files):
        with codecs.open(os.path.join(input_dir, file), "r", "utf8") as f:
            lines = f.readlines()
            data = []
            for line in lines:
                line = _format_line(line)
                if len(line) > 3:
                    l = []
                    for w in line:
                        if w[0] == "<" or w.isalpha():
                            l.append(w)
                    data.extend(l)
        data = [word2idx.get(w, unk_idx) for w in data]
        max = _classify(data)
        if max == class_id:
            correct += 1.0
        total += 1.0
        y_p.append(max)
    return total, correct, y_p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default='../classification/data/VNTC/corpus/test/')
    parser.add_argument('--prime', type=str, default="ronaldo")
    parser.add_argument('--max_samples', type=int, default=20)
    parser.add_argument('--mode', type=str, default="clf")
    parser.add_argument('--ckpt_path', type=str, default="./ckpt")
    parser.add_argument('--vocab_path', type=str, default="./corpora/vocab.txt")
    parser.add_argument('--saved_args_path', type=str, default="./ckpt/args.pkl")
    args = parser.parse_args()

    test = open("test.txt", 'r', encoding='utf-8').read()

    word2idx, idx2word = load_vocab(args.vocab_path)
    with open(args.saved_args_path, 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.embeddings_path = None
    model = TransformerDecoder(is_training=False, args=saved_args)
    pad_idx = word2idx.get("<eos>")
    unk_idx = word2idx.get("<unk>")
    with tf.Session(graph=model.graph) as sess:
        total = 0.0
        correct = 0.0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if args.ckpt_path:
            saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
        softmax = tf.reduce_mean(tf.nn.softmax(model.logits), axis=0)
        # if args.mode == "test":
        #     input_dirs = os.listdir(args.input_path)
        #     total_examples = 0
        #     for ipd in args.input_path:
        #         total_examples += len(os.path.join(args.input_path, ipd))
        #     y_true = []
        #     y_pred = []
        #     for idx, input_dir in enumerate(input_dirs):
        #         t, c, y_p = _compute_acc_for_class(os.path.join(args.input_path, input_dir), idx)
        #         y_true.extend([idx] * len(os.listdir(os.path.join(args.input_path, input_dir))))
        #         y_pred.extend(y_p)
        #         print("accuracy {:.4f} for class {}".format(c / t, CLASS_NAMES[idx]))
        #         total += t
        #         correct += c
        #     # print("accuracy {}".format(correct / total))
        #     print("f1 score {}".format(f1_score(y_true, y_pred, average="micro")))
        #     print(confusion_matrix(y_true, y_pred))
        # elif args.mode == "clf":
        print(CLASS_NAMES[_classify([word2idx.get(w, unk_idx) for w in _format_line(args.prime)])])
