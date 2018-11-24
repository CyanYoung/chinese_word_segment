import json

from segment import predict


max_len = 7

path_sent = 'data/test_sent.json'
path_label = 'data/test_label.json'
with open(path_sent, 'rb') as f:
    sents = json.load(f)
with open(path_label, 'rb') as f:
    labels = json.load(f)


def get_cut_ind(text):
    inds, count = set(), 0
    for i in range(len(text)):
        if text[i] == ' ':
            count = count + 1
            inds.add(i - count)
    return inds


def test(name, sents, labels):
    count, pred_num, label_num = [0] * 3
    for sent, label in zip(sents, labels):
        pred = predict(sent, name, max_len)
        pred_inds, label_inds = get_cut_ind(pred), get_cut_ind(label)
        for pred_ind in pred_inds:
            if pred_ind in label_inds:
                count = count + 1
        pred_num = pred_num + len(pred_inds)
        label_num = label_num + len(label_inds)
    prec, rec = count / pred_num, count / label_num
    f1 = 2 * prec * rec / (prec + rec)
    print('\n%s - prec: %.2f - rec: %.2f - f1: %.2f' % (name, prec, rec, f1))


if __name__ == '__main__':
    test('divide', sents, labels)
    test('neural', sents, labels)
