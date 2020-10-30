import os
import numpy as np
import codecs
from data_load import load_vocab
from tqdm import tqdm

base_path = '../corpora/classes/{}.txt'
seq_length = 128
word2idx, idx2word = load_vocab('../corpora/vocab.txt')

all_data = None
total_examples = 0
for i in tqdm(range(10)):
    with codecs.open(base_path.format(i), "r", "utf8") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            tokens = line.split()
            if len(tokens) > 3:
                data.extend(word2idx[token] for token in tokens)
        num_examples = len(data) // seq_length
        total_examples += num_examples
        data = data[:num_examples * seq_length]
        data = np.reshape(np.asarray(data), [num_examples, seq_length])
        label = np.zeros([num_examples, 1]) + i
        data = np.concatenate((data, label), 1)
        if all_data is None:
            all_data = data
        else:
            all_data = np.concatenate((all_data, data), 0)
np.save("corpora/train.npy", all_data)
