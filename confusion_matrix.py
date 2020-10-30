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
from imutils import paths
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

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
    parser.add_argument('--prime', type=str,
                        default="Giao thông được hỗ trợ bởi công nghệ mới chính là chìa khoá giúp cải thiện hơn nữa chất lượng cuộc sống của người dân./")
    parser.add_argument('--max_samples', type=int, default=20)
    parser.add_argument('--mode', type=str, default="clf")
    parser.add_argument('--ckpt_path', type=str, default="./ckpt")
    parser.add_argument('--vocab_path', type=str, default="./corpora/vocab.txt")
    parser.add_argument('--saved_args_path', type=str, default="./ckpt/args.pkl")
    args = parser.parse_args()

    DATASET_PATH = "mydata/test"
    LABELS = os.listdir(DATASET_PATH)

    word2idx, idx2word = load_vocab(args.vocab_path)
    with open(args.saved_args_path, 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.embeddings_path = None
    model = TransformerDecoder(is_training=False, args=saved_args)
    pad_idx = word2idx.get("<eos>")
    unk_idx = word2idx.get("<unk>")
    y_test = []
    y_pred = []
    filePaths = list(paths.list_files(DATASET_PATH))
    with tf.Session(graph=model.graph) as sess:
        total = 0.0
        correct = 0.0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if args.ckpt_path:
            saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
        softmax = tf.reduce_mean(tf.nn.softmax(model.logits), axis=0)
        for filePath in tqdm(filePaths):
            text = open(filePath, 'r', encoding='utf-8').read()
            true_label = filePath.split(os.path.sep)[-2]
            y_test.append(true_label)

            pred_label = LABELS[_classify([word2idx.get(w, unk_idx) for w in _format_line(text)])]
            y_pred.append(pred_label)

    array = confusion_matrix(y_test, y_pred)
    print(sum(array[ii][ii] for ii in range(len(array))) / len(filePaths))

    df_cm = pd.DataFrame(array, index=LABELS, columns=LABELS)
    # plt.figure(figsize=(10,7))
    # df_norm_col = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

    sn.set(font_scale=1)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap="Blues")  # font size
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.savefig('cf_18class.png', pdi=300)
    plt.show()
