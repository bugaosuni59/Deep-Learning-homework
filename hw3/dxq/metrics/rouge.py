# coding=utf-8

import numpy as np
import pdb


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


def compute_rouge(candidate, refs, beta=1.2):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    prec = []
    rec = []

    # split into tokens
    token_c = candidate

    for reference in refs:
        # split into tokens
        token_r = reference
        # compute the longest common subsequence
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs / float(len(token_c)))
        rec.append(lcs / float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)

    if (prec_max != 0 and rec_max != 0):
        score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
    else:
        score = 0.0
    return score

def cal_rouge_score(pred, label):
    '''
      :param pred: ndarray: [timestep, batch]
      :param label: bdarray: [timestep, batch]
      :return:
      '''

    T, B = pred.shape

    sum = 0
    for i in range(B):
        sum += compute_rouge(pred[:, i], [label[:, i]])
    return sum / B

if __name__ == "__main__":
    hypothesis = [0, 1, 2, 3, 4, 5, 18,
                  6, 7, 8, 9, 17,
                  16, 8, 10, 15, 8, 11]
    reference = [0, 1, 2, 3, 4, 5, 7,
                 6, 7, 8, 9, 13, 14,
                 12, 11, 10]
    import numpy as np

    pred = np.tile(np.array(hypothesis), (20, 1))
    label = np.tile(np.array(reference), (20, 1))

    label[0] = np.array(reference) + 2
    label[-1] = np.array(reference) - 3

    # print(calc_score([reference1], hypothesis1))
    print(compute_rouge(hypothesis,[reference]))
    print(cal_rouge_score(pred.transpose((1,0)), label.transpose((1,0))))