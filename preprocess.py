import json

import re

from random import shuffle


def save(path, texts):
    with open(path, 'w') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)


def clean(text):
    text = re.sub('\d{8}-\d{2}-\d{3}-\d{3}', '', text)
    text = re.sub('\[', '', text)
    return re.sub('/\S+', '', text)


def prepare(path_univ, path_train, path_test_sent, path_test_label):
    texts = list()
    with open(path_univ, 'r') as f:
        for line in f:
            text = clean(line).strip()
            if text:
                text = re.sub('  ', ' ', text)
                texts.append(text)
    shuffle(texts)
    bound = int(len(texts) * 0.9)
    train_sents, test_labels = texts[:bound], texts[bound:]
    test_sents = list()
    for label in test_labels:
        test_sents.append(re.sub(' ', '', label))
    save(path_train, train_sents)
    save(path_test_sent, test_sents)
    save(path_test_label, test_labels)


if __name__ == '__main__':
    path_univ = 'data/univ.txt'
    path_train = 'data/train.json'
    path_test_sent = 'data/test_sent.json'
    path_test_label = 'data/test_label.json'
    prepare(path_univ, path_train, path_test_sent, path_test_label)
