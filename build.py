import json
import pickle as pk

import nltk
from nltk import MLEProbDist

from gensim.models import Word2Vec


embed_len = 200
min_freq = 5

path_word_vec = 'feat/word_vec.pkl'
path_cfd = 'feat/cfd.pkl'
path_cpd = 'feat/cpd.pkl'


def word2vec(sents, path_word_vec):
    sent_words = [sent.split() for sent in sents]
    model = Word2Vec(sent_words, size=embed_len, window=3, min_count=min_freq, negative=5, iter=10)
    word_vecs = model.wv
    with open(path_word_vec, 'wb') as f:
        pk.dump(model, f)
    if __name__ == '__main__':
        words = ['几', '元', '天']
        for word in words:
            print(word_vecs.most_similar(word))


def fit(path_train, update):
    with open(path_train, 'r') as f:
        sents = json.load(f)
    if update:
        word2vec(sents, path_word_vec)
    all_words = ' '.join(sents).split()
    bigrams = list(nltk.ngrams(all_words, 2))
    cfd = nltk.ConditionalFreqDist(bigrams)
    cpd = nltk.ConditionalProbDist(cfd, MLEProbDist)
    with open(path_cfd, 'wb') as f:
        pk.dump(cfd, f)
    with open(path_cpd, 'wb') as f:
        pk.dump(cpd, f)


if __name__ == '__main__':
    path_train = 'data/train.json'
    fit(path_train, update=False)
