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


def prepare(path_univ, path_train, path_test, path_label):
    texts = list()
    with open(path_univ, 'r') as f:
        for line in f:
            text = clean(line).strip()
            if text:
                text = re.sub('  ', ' ', text)
                texts.append(text)
    shuffle(texts)
    bound = int(len(texts) * 0.9)
    train_texts, labels = texts[:bound], texts[bound:]
    test_texts = list()
    for label in labels:
        test_texts.append(re.sub(' ', '', label))
    save(path_train, train_texts)
    save(path_label, labels)
    save(path_test, test_texts)


if __name__ == '__main__':
    path_univ = 'data/univ.txt'
    path_train = 'data/train.json'
    path_test = 'data/test.json'
    path_label = 'data/label.json'
    prepare(path_univ, path_train, path_test, path_label)
