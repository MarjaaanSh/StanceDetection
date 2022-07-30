import numpy as np

#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

# LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS = ['unrelated', 'related']
LABELS_RELATED = ['unrelated', 'related']
RELATED = ['agree', 'disagree', 'discuss']


def score_submission(actual, predicted):
    score = 0.0
    cls = len(set(actual))
    cm = np.zeros((cls, cls))
    for a, p in zip(actual, predicted):
        if a == p:
            score += 0.25
            if a != 'unrelated':
                score += 0.50
        # if g in RELATED and t in RELATED:
        if a == p and a == 'related':
            score += 0.25
        # print(a, p, LABELS.index(a), LABELS.index(p))
        cm[LABELS.index(a), LABELS.index(p)] += 1
        # print(cm)
    return score, cm


def print_confusion_matrix(cm):
    lines = []
    # header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    header = "|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        # lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i], *row))
        lines.append("|{:^11}|{:^11}|{:^11}|".format(LABELS[i], *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(actual,predicted):
    predicted = [LABELS[int(p)] for p in predicted]
    actual = [LABELS[int(a)] for a in actual]

    score, cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual, actual)

    denum = cm.sum(axis=1)[:, None]
    normalized_cm = cm / denum
    normalized_cm = np.round(normalized_cm, 2)
    print_confusion_matrix(cm)
    print_confusion_matrix(normalized_cm)
    print("Score: " + str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score