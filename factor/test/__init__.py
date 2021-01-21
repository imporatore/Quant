import numpy as np
import pandas as pd

from utils.plot import *


def weighted_factor_value(factor_score, free_shares, returns, mindex, n_top=50):
    """
    Compute market free shares weighted value for stocks with

    top-k & bottom-k factor scores, all parameters should have
    corresponding rank of axis datetime & stock.

    Params:
        - factor_score: pd.DataFrame, dir: +, sample:
        STOCK_CODE        DATE1        DATE2        DATE3
         xxxxxx            xxx          xxx          xxx
         xxxxxx            xxx          xxx          xxx

        - free_shares: pd.DataFrame, sample:
        STOCK_CODE        DATE1        DATE2        DATE3
         xxxxxx            xxx          xxx          xxx
         xxxxxx            xxx          xxx          xxx

        - returns: pd.DataFrame, sample:
        STOCK_CODE        DATE1        DATE2        DATE3
         xxxxxx            xxx          xxx          xxx
         xxxxxx            xxx          xxx          xxx

        - mindex: pd.DataFrame, sample:
        DATETIME        INDEX
         xxxxx           xxx
         xxxxx           xxx
    """
    dates = pd.to_datetime(mindex.DATETIME)

    if isinstance(factor_score, pd.DataFrame):
        factor_score = factor_score.values

    if isinstance(free_shares, pd.DataFrame):
        free_shares = free_shares.values


    if isinstance(returns, pd.DataFrame):
        returns = returns.values

    n_stocks = factor_score.shape[0]
    top_k_returns = list()
    bot_k_returns = list()

    for date in range(factor_score.shape[1] - 1):
        factor_rank = sorted(np.arange(n_stocks),
                             key=lambda i: factor_score[i, date],
                             reverse=True)
        top_k = (np.array(factor_rank) < n_top)
        bot_k = (np.array(factor_rank) >= n_stocks - n_top)
        top_k_returns.append(np.sum(returns[top_k, date] * free_shares[top_k, date] /
                                    np.sum(free_shares[top_k])))
        bot_k_returns.append(np.sum(returns[bot_k, date] * free_shares[bot_k, date] /
                                    np.sum(free_shares[bot_k])))

    plt.plot(dates, top_k_returns, label=f"top {n_top} score")
    plt.plot(dates, mindex, label=f"mother index")
    plt.plot(dates, bot_k_returns, labels=f"bottom {n_top} score")
    plt.xlabel('date')

