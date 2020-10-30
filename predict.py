import argparse
import pickle
import tensorflow as tf
import numpy as np
from data_load import load_vocab, basic_tokenizer
from models import TransformerDecoder
import regex as re
from utils import url_marker

CLASS_NAMES = ['Bóng đá', 'Bất động sản', 'Chính trị - Quân sự', 'Công nghệ', 'Du lịch', 'Game', 'Giáo dục', 'Kinh doanh', 'Lao động - Việc làm', 'Pháp luật', 'Showbiz', 'Sức khỏe', 'Thế giới động vật', 'Thời tiết', 'Thời trang', 'Tình yêu', 'Ô tô', 'Ẩm thực']

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
    batch_size = min(args.max_samples, len(data) // saved_args.maxlen)
    if batch_size == 0:
        prime = np.array(data[:saved_args.maxlen])
        prime = np.atleast_2d(np.tile(prime, saved_args.maxlen // len(prime) + 1)[:saved_args.maxlen])
    else:
        prime = data[:saved_args.maxlen * batch_size]
        prime = np.reshape(np.array(prime), [batch_size, saved_args.maxlen])
    preds, dec, proj = sess.run((softmax, model.dec, model.proj), feed_dict={model.x: prime})
    return np.argmax(preds)

parser = argparse.ArgumentParser()
parser.add_argument('--max_samples', type=int, default=20)
parser.add_argument('--ckpt_path', type=str, default="./ckpt")
parser.add_argument('--vocab_path', type=str, default="./corpora/vocab.txt")
parser.add_argument('--saved_args_path', type=str, default="./ckpt/args.pkl")
args = parser.parse_args()

word2idx, idx2word = load_vocab(args.vocab_path)
unk_idx = word2idx.get("<unk>")

#laod model
with open(args.saved_args_path, 'rb') as f:
    saved_args = pickle.load(f)
saved_args.embeddings_path = None
model = TransformerDecoder(is_training=False, args=saved_args)
with model.graph.as_default():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if args.ckpt_path:
        saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
    softmax = tf.reduce_mean(tf.nn.softmax(model.logits), axis=0)

#predict
def predict(text):
    return (CLASS_NAMES[_classify([word2idx.get(w, unk_idx) for w in _format_line(text)])])
