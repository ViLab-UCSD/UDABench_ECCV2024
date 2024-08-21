import torch
from torch import nn
import torch.nn.functional as F


class fewShotCls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, temperature=0.05):
        super(fewShotCls, self).__init__()
        self.n_class = n_class
        self.temperature = temperature

        assert len(feat_size) == 1 , "For few shot classifier, MLP is not allowed."

        self.out = nn.Linear(feat_size[-1], n_class, bias=False)
        torch.nn.init.xavier_normal_(self.out.weight)


    def forward(self, x, feat=False):

        assert feat == False

        x = F.normalize(x, p=2, dim=1)

        y = self.out(x) / self.temperature

        if feat:
            return y, x
        return y


def fscls(**kwargs):
    model = fewShotCls(**kwargs)
    return model