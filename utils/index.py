import numpy as np

from utils.converters import to_numpy
from functools import reduce


def rate_of_return(seq):
    seq_ = to_numpy(seq)
    if len(seq_.shape) == 1:
        return 100. * (seq_[1:] / seq_[:-1] - 1.)
    elif len(seq_.shape) == 2:
        return 100. * (seq_[:, 1:] / seq_[:, -1] - 1.)
    raise ValueError("Rate of return supports only 1-dim or 2-dim arrays.")


def cumulative_rate_of_return(seq):
    seq_ = to_numpy(seq)
    if len(seq_.shape) == 1:
        rr = seq_[1:] / seq_[:-1]
        return 100. * (reduce(lambda crr_, rr_: np.hstack((crr_, np.array(crr_[-1] * rr_))), rr, np.ones(1)) - 1.)
    elif len(seq_.shape) == 2:
        rr = seq_[:, 1:] / seq_[:, -1]
        return 100. * (reduce(lambda crr_, rr_: np.hstack((crr_, np.array(crr_[:, -1] * rr_).reshape((-1, 1)))),
                              [rr[:, i] for i in range(rr.shape[1])], np.ones((rr.shape[0], 1))) - 1.)
    raise ValueError("Cumulative rate of return supports only 1-dim or 2-dim arrays.")


rr = rate_of_return
crr = cumulative_rate_of_return
