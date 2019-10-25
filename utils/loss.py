# coding=utf-8

import torch
import torch.nn as nn


class CrossEntropyLoss(object):
    """the normal cross entropy loss"""
    def __init__(self, ignore_index=255, accumulation_steps=1):
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=self.ignore_index,
                                        reduction='elementwise_mean')
        self.accumulation_steps = accumulation_steps

    def __call__(self, logit, target):
        """call method"""
        #n, c, h, w = logit.size()
        loss = self.criterion(logit, target.long())
        #loss = torch.div(loss, accumulation_steps * 1.0)
        return loss

    def cuda(self, main_gpu=0):
        self.criterion = self.criterion.cuda(main_gpu)


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index, reduction=reduction)

    def forward(self, *inputs):
        return super(SegmentationLosses, self).forward(*inputs)


if __name__ == "__main__":
    pass
