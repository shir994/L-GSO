from scipy.stats import ks_2samp, skew
import torch


class Metrics(object):
    def __init__(self, bins_range, n_bins):
        self.bins_range = bins_range
        self.n_bins = n_bins
        self.epsilon = 1e-10

    def _bin_histogram(self, p):
        hist = torch.histc(p, self.n_bins, *self.bins_range)
        hist[hist == 0] = self.epsilon
        return hist / hist.sum()

    def _KL(self, p, q):
        return - (p * (q / p).log()).sum()

    def compute_KL(self, p, q):
        p = self._bin_histogram(p)
        q = self._bin_histogram(q)
        return self._KL(p, q)

    def compute_KSStat(self, p, q):
        stat, p_val = ks_2samp(p.reshape(-1), q.reshape(-1))
        return stat

    def compute_JS(self, p, q):
        p = self._bin_histogram(p)
        q = self._bin_histogram(q)
        M = 0.5 * (p + q)
        return 0.5 * (self._KL(p, M) + self._KL(q, M))

    #     def compute_moment(self, p, order):
    #         hist = self._bin_histogram(p)
    #         hist[hist == self.epsilon] = 0
    #         print(len(hist))
    #         bin_step = (self.bins_range[1] - self.bins_range[0]) / self.n_bins
    #         bins_x_val = torch.arange(self.bins_range[0], self.bins_range[1], bin_step) + bin_step / 2
    #         print(bins_x_val)
    #         expectation = (bins_x_val * hist).sum()
    #         if order == 1:
    #             return expectation
    #         else:
    #             return (torch.pow(bins_x_val - expectation, order) * hist).sum()

    def compute_moment(self, p, order):
        if order == 1:
            return p.mean()
        if order == 2:
            return p.var()
        if order == 3:
            return skew(p.detach().cpu().numpy())[0]