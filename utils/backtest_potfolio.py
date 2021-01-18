import numpy as np


def return_rate(history, portfolio):
    """
    Params:
        - history: pd.Dataframe in shape(n, m), sample:
            |STOCK CODE|  CLOSE_1  |  ...  |  CLOSE_M  |
            |------------------------------------------|
            |  xxxxxx  |           |       |           |
            |  xxxxxx  |           |       |           |
        - portfolio:
            np.array in shape(n,), weight for portfolio
            with same order of stocks in history
    Return:
        Return rate of portfolio at datetimes in history.
    """
    wa = np.sum(history.values * portfolio.reshape((history.shape[0], )), axis=0)
    return wa / wa[0]

