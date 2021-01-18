import numpy as np


class Portfolio:

    def __init__(self, method, *args):
        """
        Params:
            - method: string, method used to calculate stock portfolio.
                - "head_weighted": return weighted stocks for top-10 scored stocks
                - "head_equal": return equal weight for top-10 scored stocks
                - "global_weighted": return equal weight for all stock
                - "global_equal": return weight calculated by score
                - "pos_weighted": return weighted weights for all stocks which have positive score
                - "pos_equal": return equal weights for all stocks which have positive score
        """
        self.method = method
        self.method_params = args

    def get_portfolio(self, score):
        """
        Calculate weight for all stocks according to the score.

        :param score: 1-dim array.
        :return: 1-dim array of weight for stocks, can be negative.
        """


class ESGScoreTest:

    def __init__(self, methods):
        self.methods = methods

    def test(self, score):
        pass

    def
