import json
import pickle as pk

import nltk

import math

from util import map_item


def divide_smooth(cond, word):
    pass


def neural_smooth(cond, word):
    pass


path_vocab_freq = 'stat/vocab_freq.json'
path_cpd = 'feat/cpd.pkl'
path_word_vec = 'feat/word_vec.pkl'
with open(path_vocab_freq, 'rb') as f:
    vocabs = json.load(f)
with open(path_cpd, 'rb') as f:
    cpd = pk.load(f)
with open(path_word_vec, 'rb') as f:
    word_vec = pk.load(f)

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
        if prob > 0:
            log_sum = log_sum + math.log(prob)
        else:
            log_sum = log_sum + smooth(cond, word)
    return log_sum


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
