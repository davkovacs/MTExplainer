#!/usr/bin/env python

import torch.nn as nn
from translate_IG import main as score

class ProbDifference(nn.Module):
   # nn returning difference inn probability of two possible targets
    def __init__(self):
        super(ProbDifference,self).__init__()
        self.scorer = score()  # The function of William returning the two gold scores

    def forward(self, x):
        x = self.scorer(x)
        return x
