import numpy as np


class Portfolio:

    def __init__(self, method, head=10):
        """
        Params:
            - method: string, method used to calculate stock portfolio.
                - "head_weighted": return weighted stocks for top-10 scored stocks
                - "global_weighted": return equal weight for all stock
                - "pos_weighted": return weighted weights for all stocks which have positive score
                - "pos_equal": return equal weights for all stocks which have positive score
        """
        self._method = method
        self.head = head

    def get_portfolio(self, score, weight_func):
        """
        Calculate weight for all stocks according to the score.

        Params:
            - score: 1-dim array.

        Return:
            1-dim array of weight for stocks, can be negative.
        """
        if self._method == 'head':
            top_score = sorted(score)[-self.head]
            score_ = np.array([s if s >= top_score else 0 for s in score])
        elif self._method == 'global':
            score_ = score
        elif self._method == 'pos':
            score_ = np.array([s if s > 0 else 0 for s in score])
        else:
            raise ValueError(f'Method {self._method} not implemented')

        return weight_func(score_)

identity_weight = lambda x: x
equal_weight = lambda x: np.ones_like
