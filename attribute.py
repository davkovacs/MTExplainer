#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import score

class ProbDifference(nn.Module):
   # nn returning difference inn probability of two possible targets
    def __init__(self, src):
        super(ProbDifference,self).__init__()
        self.scorer = score(src)  # The function of William returning the two gold scores

    def forward(self, x):
        x = self.scorer(x)
        return x