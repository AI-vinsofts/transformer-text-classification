from __future__ import print_function
import numpy as np
import codecs
import regex as re
from tqdm import tqdm
import glob
import os

_WORD_SPLIT = re.compile("([.,!?\"/':;)(])")
_DIGIT_RE = re.compile(br"\d")
STOP_WORDS = "\" \' [ ] . , ! : ; ?".split(" ")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    return [w.lower() for w in words if w != '' and w != ' ']


def load_vocab(path):
    vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_train_data(train_path):
    classes = len(glob.glob(train_path+"/*.npy"))
    print("num classes = {}".format(classes))
    class_weights = np.zeros(classes)
    result = []
    for cls in range(classes):
        result.append(np.load(os.path.join(train_path,"{}.npy".format(cls))))
        class_weights[cls] = len(result[cls])
    total_tokens = np.sum(class_weights)
    class_weights = total_tokens / class_weights
    class_weights = class_weights / np.mean(class_weights)
    return result, class_weights


def _select_examples(X, maxlen):
    begin = np.random.randint(len(X) - maxlen)
    return X[begin: begin + maxlen]
    pass


def next_batch(X, batch_size, maxlen):
    x = np.zeros(shape=[batch_size, maxlen + 1])
    choices = np.random.randint(len(X), size=batch_size)
    for idx, choice in enumerate(choices):
        x[idx, :-1] = _select_examples(X[choice], maxlen)
        x[idx, -1] = choice
    Y = np.expand_dims(x[:, -1], 1).copy()
    x = x[:, :-1].copy()
    return x, Y
