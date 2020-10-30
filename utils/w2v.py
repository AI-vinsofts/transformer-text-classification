import gensim
from gensim.models import Word2Vec
import codecs
from data_load import basic_tokenizer, load_vocab
import numpy as np

sentences = []
with codecs.open("corpora/train2.txt", "r","utf8") as f:
    lines = f.readlines()
    for line in lines:
        sentences.append(line.split( ))
model = Word2Vec(sentences, size=256, window=5, min_count=5, workers=8)
model.save("w2v.model")
model = Word2Vec.load("w2v.model")
word2idx, idx2word = load_vocab("corpora/vocab.txt")

embeddings = np.zeros(shape=[len(word2idx), 256])
for word,idx in word2idx.items():
    embeddings[idx, :] = model.wv[word]
np.save("corpora/embeddings.npy", embeddings)