from __future__ import print_function

import argparse
import pickle
import sys

import tensorflow as tf
import numpy as np
from data_load import load_vocab, basic_tokenizer
from models import TransformerDecoder
import regex as re
from utils import url_marker
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

CLASS_NAMES = ['chinh tri xa hoi', 'doi song', 'khoa hoc', 'kinh doanh', 'phap luat', 'suc khoe', 'the gioi',
               'the thao', 'van hoa', 'vi tinh']


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
    attns = attns
    return np.argmax(preds), preds, attns


parser = argparse.ArgumentParser()
parser.add_argument('--max_samples', type=int, default=20)
parser.add_argument('--mode', type=str, default="clf")
parser.add_argument('--ckpt_path', type=str, default="./ckpt")
parser.add_argument('--vocab_path', type=str, default="./corpora/vocab.txt")
parser.add_argument('--saved_args_path', type=str, default="./ckpt/args.pkl")
args = parser.parse_args()

word2idx, idx2word = load_vocab(args.vocab_path)
with open(args.saved_args_path, 'rb') as f:
    saved_args = pickle.load(f)
saved_args.embeddings_path = None
model = TransformerDecoder(is_training=False, args=saved_args)
pad_idx = word2idx.get("<eos>")
unk_idx = word2idx.get("<unk>")
with model.graph.as_default():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if args.ckpt_path:
        saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
    softmax = tf.reduce_mean(tf.nn.softmax(model.logits), axis=0)


@app.route('/clf', methods=['POST'])
def classify():
    content = request.get_json()
    print(content["data"], file=sys.stderr)
    formatted_line = _format_line(content["data"])
    line_ = []
    for w in formatted_line:
        if w[0] == "<" or w.isalpha():
            line_.append(w)
    pred, probs, attns = _classify([word2idx.get(w, unk_idx) for w in line_])
    data = {
        "pred": str(pred),
        "probs": probs.tolist(),
        "attns": attns.tolist(),
        "sent": line_
    }
    return json.dumps(data)


if __name__ == '__main__':
    app.run()
