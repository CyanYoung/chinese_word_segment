import json
import pickle as pk

import nltk

import math

from util import map_item


def divide_smooth(cond):
    if cond not in cfd:
        prob = 1 / len(cfd)
    else:
        prob = 1 / sum(cfd[cond].values())
    return prob


def neural_smooth(cond, word, cand, depth, thre):
    if depth > thre:
        return divide_smooth(cond)
    if cond not in cfd:
        cond_subs = word_vecs.most_similar(cond)[:cand]
        cond_flag = False
        for cond_sub in cond_subs:
            if cond_sub in cfd:
                cond, cond_flag = cond_sub, True
                break
        if not cond_flag:
            return divide_smooth(cond)
        else:
            neural_smooth(cond, word, cand, depth + 1, thre)
    if word not in cfd[cond]:
        word_subs = word_vecs.most_similar(word)[:cand]
        word_flag = False
        for word_sub in word_subs:
            if word_sub in cfd[cond]:
                word, word_flag = word_sub, True
                break
        if not word_flag:
            return divide_smooth(cond)
    return cpd[cond].prob(word)


path_vocab_freq = 'stat/vocab_freq.json'
path_cfd = 'feat/cfd.pkl'
path_cpd = 'feat/cpd.pkl'
path_word_vec = 'feat/word_vec.pkl'
with open(path_vocab_freq, 'rb') as f:
    vocabs = json.load(f)
with open(path_cfd, 'rb') as f:
    cfd = pk.load(f)
with open(path_cpd, 'rb') as f:
    cpd = pk.load(f)
with open(path_word_vec, 'rb') as f:
    word_vecs = pk.load(f)

funcs = {'divide': divide_smooth,
         'neural': neural_smooth}


def for_match(sent, max_len):
    words = list()
    sent_len, head = len(sent), 0
    while head < len(sent):
        tail = min(head + max_len, sent_len)
        while head < tail - 1:
            if sent[head:tail] in vocabs:
                words.append(sent[head:tail])
                break
            else:
                tail = tail - 1
        if head == tail - 1:
            words.append(sent[head])
            head = head + 1
        else:
            head = tail
    return words


def back_match(sent, max_len):
    words = list()
    sent_len, tail = [len(sent)] * 2
    while tail > 0:
        head = max(tail - max_len, 0)
        while tail - 1 > head:
            if sent[head:tail] in vocabs:
                words.append(sent[head:tail])
                break
            else:
                head = head + 1
        if tail - 1 == head:
            words.append(sent[tail - 1])
            tail = tail - 1
        else:
            tail = head
    return list(reversed(words))


def get_log(words, name):
    smooth = map_item(name, funcs)
    bigrams = list(nltk.ngrams(words, 2))
    log_sum = 0
    for cond, word in bigrams:
        prob = cpd[cond].prob(word)
        if prob == 0.0:
            if name == 'neural':
                prob = smooth(cond, word, cand=5, depth=1, thre=3)
            else:
                prob = smooth(cond)
        log_sum = log_sum + math.log(prob)
    return log_sum / len(words)


def predict(text, name, max_len):
    sent = text.strip()
    word1s, word2s = for_match(sent, max_len), back_match(sent, max_len)
    if word1s == word2s:
        return ' '.join(word1s)
    else:
        sent1_log, sent2_log = get_log(word1s, name), get_log(word2s, name)
        if sent1_log > sent2_log:
            return ' '.join(word1s)
        else:
            return ' '.join(word2s)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('divide: %s' % predict(text, 'divide', max_len=7))
        print('neural: %s' % predict(text, 'neural', max_len=7))
