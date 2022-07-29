#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

from locale import normalize
from re import A
import numpy as np

def get_score(actual, predicted):
    score = 0.0
    for (a, p) in zip(actual, predicted):
        if a == p:
            score += 0.25
            if a != 'unrelated':
                score += 0.50
        if a in RELATED and p in RELATED:
            score += 0.25
    return score

def get_confusion(actual, predicted):
    d1 = len(set(actual))
    cm = np.zeros((d1, d1))

    for (a, p) in zip(actual, predicted):
        cm[LABELS.index(a)][LABELS.index(p)] += 1
    tot = cm.sum(axis=1)[:, None]
    norm_cm = cm / tot
    return cm, norm_cm

def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(actual, predicted):
    confusion, norm_conf = get_confusion(actual, predicted)
    
    score = get_score(actual,predicted)
    best_score  = get_score(actual,actual)

    print_confusion_matrix(confusion)
    print_confusion_matrix(norm_conf)

    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")

    return score*100/best_score
