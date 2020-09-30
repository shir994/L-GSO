import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, condition=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](z=x, condition=condition, reverse=reverse)
            return x
        else:
            for i in inds:
                # print(self.chain[i].__class__)
                x, logpx = self.chain[i](z=x, logpz=logpx, condition=condition, reverse=reverse)
            return x, logpx
