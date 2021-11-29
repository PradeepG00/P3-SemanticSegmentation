import torch
import torch.nn as nn
import torch.nn.functional as F


class ACWLoss(nn.Module):
    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5, ignore_index=255):
        """Adaptive Class Weighting Loss is the loss function class for handling the highly imbalanced distribution
        of images Multi-class adaptive class loss function

        **Adaptive Class Weighting Loss**

        .. math::

            L_{acw}=\\frac{1}{|Y|}\\sum_{i\\in Y}\\sum_{j\\in C}{\\tilde{w}_{ij}\\times p_{ij} -log{( \\text{
            MEAN}\\{ d_j| j\\in C \\} )} }

        **Dice coefficient**

        .. math::

            d_j = \\frac{ 2\\sum_{i\\in Y}y_{ij} \\tilde{y}_{ij}} {\\sum_{ij}y_{ij} + \\sum_{i\\in Y}\\tilde{
            y}_{ij} }

        :param ini_weight:
        :param ini_iteration:
        :param eps:
        :param ignore_index:z
        """
        super(ACWLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        # one = torch.ones_like(err)

        pnc = self.pnc(err)
        loss_pnc = torch.sum(acw * pnc, 1)

        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def pnc(self, err):
        """Apply positive-negative class balanced function (PNC)

        **PNC**
        .. math::
            p = e - \\log(\\frac{1-e}{1+e})\\mid e=(y-\\tilde{y})^2

        :param err:
        :return:
        """
        return err - ((1.0 - err + self.eps) / (1.0 + err + self.eps)).log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        """Adaptive Class Weighting (ACW) computed based on the iterative batch-wise
        class derived from the median frequency to balance weights.

        **ACW**

        .. math::
            \\tilde{w}_{ij}=\\frac{ w^t_j}
            { \\sum_{j\\in C}(w^t_j) }\\times (1 + y_{ij} + \\tilde{y}_{ij})

        **Iterative Median Frequency Class Weights**

        .. math::

            w^t_j=\\frac{\\text{MEDIAN(\\{   f^t_j|j\\in C\\})}}
            {f^t_j+\\epsilon}\\mid\\epsilon=10^{-5}

        **Pixel Frequency**

        .. math::
            f^t_j=\\frac{\\hat{f^t_j}+(t-1)\\times f^{t-1}_j}
            {t} \\mid t\\in \\{1,2,...,\\infty\\}


        :param pred:
        :param one_hot_label:
        :param mask:
        :return:
        """
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        acw = (1.0 + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None
