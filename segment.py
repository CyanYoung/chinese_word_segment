import json
import pickle as pk

import nltk


path_vocab_freq = 'stat/vocab_freq.json'
path_cpd = 'feat/cpd.pkl'
path_word_vec = 'feat/word_vec.pkl'
with open(path_vocab_freq, 'rb') as f:
    vocabs = json.load(f)
with open(path_cpd, 'rb') as f:
    cpd = pk.load(f)
with open(path_word_vec, 'rb') as f:
    word_vec = pk.load(f)


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


def get_prob(words):
    pass


def predict(text, name, max_len):
    sent = text.strip()
    word1s, word2s = for_match(sent, max_len), back_match(sent, max_len)
    if word1s == word2s:
        return ' '.join(word1s)
    else:
        sent1_prob, sent2_prob = get_prob(word1s), get_prob(word2s)
        if sent1_prob > sent2_prob:
            return ' '.join(word1s)
        else:
            return ' '.join(word2s)



if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('const:  %s' % predict(text, 'const', max_len=7))
        print('neural: %s' % predict(text, 'neural', max_len=7))
