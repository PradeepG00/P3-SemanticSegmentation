from __future__ import print_function, division
from multiprocessing import Pool
from multiprocessing import sharedctypes

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(predictions, gts, num_classes):
    """Function for evaluating the collection of predictions given the set of ground-truths

    :param predictions:
    :param gts:
    :param num_classes:
    :return:
    """
    con_matrix = np.zeros((num_classes, num_classes))
    labels = np.arange(num_classes).tolist()
    for lp, lt in zip(predictions, gts):
        lp[lt == 255] = 255
        # lt[lt < 0] = -1
        con_matrix += confusion_matrix(lt.flatten(), lp.flatten(), labels=labels)

    M, N = con_matrix.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = con_matrix[i, i]
        fp[i] = np.sum(con_matrix[:, i]) - tp[i]
        fn[i] = np.sum(con_matrix[i, :]) - tp[i]

    precision = tp / (tp + fp)  # = tp/col_sum
    recall = tp / (tp + fn)
    f1_score = 2 * recall * precision / (recall + precision)

    ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix
    acc = np.diag(con_matrix).sum() / con_matrix.sum()
    acc_cls = np.diag(con_matrix) / con_matrix.sum(axis=ax_p)
    acc_cls = np.nanmean(acc_cls)
    iu = tp / (tp + fp + fn)
    mean_iu = np.nanmean(iu)
    freq = con_matrix.sum(axis=ax_p) / con_matrix.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, np.nanmean(f1_score)


def multiprocess_evaluate(predictions, gts, num_classes):
    """Function for evaluating the collection of predictions given the set of ground-truths

    :param predictions:
    :param gts:
    :param num_classes:
    :return:
    """
    con_matrix = np.zeros((num_classes, num_classes))
    labels = np.arange(num_classes).tolist()
    for lp, lt in zip(predictions, gts):
        lp[lt == 255] = 255
        # lt[lt < 0] = -1
        con_matrix += confusion_matrix(lt.flatten(), lp.flatten(), labels=labels)

    M, N = con_matrix.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = con_matrix[i, i]
        fp[i] = np.sum(con_matrix[:, i]) - tp[i]
        fn[i] = np.sum(con_matrix[i, :]) - tp[i]

    precision = tp / (tp + fp)  # = tp/col_sum
    recall = tp / (tp + fn)
    f1_score = 2 * recall * precision / (recall + precision)

    ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix
    acc = np.diag(con_matrix).sum() / con_matrix.sum()
    acc_cls = np.diag(con_matrix) / con_matrix.sum(axis=ax_p)
    acc_cls = np.nanmean(acc_cls)
    iu = tp / (tp + fp + fn)
    mean_iu = np.nanmean(iu)
    freq = con_matrix.sum(axis=ax_p) / con_matrix.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, np.nanmean(f1_score)


class AverageMeter(object):
    def __init__(self):
        """

        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
