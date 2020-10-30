from __future__ import print_function

from data_load import load_vocab, basic_tokenizer
import regex as re
from utils import url_marker
import os
from imutils import paths
import numpy as np
from tqdm import tqdm

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



if __name__ == '__main__':

    word2idx, idx2word = load_vocab("./corpora/vocab.txt")
    unk_idx = word2idx.get("<unk>")

    root = "mydata/train/"
    folders = os.listdir(root)
    id = 0
    for folder in tqdm(folders):
        path = root + folder
        filepaths = list(paths.list_files(path))
        data = ""
        for filepath in filepaths:
            text = open(filepath, 'r', encoding='utf-8').read()
            data += text + " "
        # hhh = data.split(".")
        # print(len(hhh)/270)
        result = np.asarray([word2idx.get(w, unk_idx) for w in _format_line(data)])
        np.save("corpora/data/"+str(id)+".npy", result)
        id +=1
