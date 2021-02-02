import numpy as np
import pandas as pd

from utils import to_numpy
from utils.plot import *


def fs_weighted_top_factor_return_rate(factor_score, free_shares, returns, mindex, n_top=20, mname=''):
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
    if not mname:
        mname = mindex.columns[-1]

    factor_score = to_numpy(factor_score)
    free_shares = to_numpy(free_shares)
    returns = to_numpy(returns)

    top_k_returns = [1., ]
    bot_k_returns = [1., ]

    for date in range(factor_score.shape[1] - 2):
        mask = (free_shares[:, date+1] > 0)
        factor_rank = sorted(np.arange(np.sum(mask)),
                             key=lambda i: factor_score[mask, date+1][i],
                             reverse=True)
        top_k = factor_rank[:n_top]
        bot_k = factor_rank[-n_top:]
        top_k_returns.append(np.sum(returns[mask, date+2][top_k] * free_shares[mask, date+1][top_k]) /
                             np.sum(returns[mask, date+1][top_k] * free_shares[mask, date+1][top_k]) * top_k_returns[-1])
        bot_k_returns.append(np.sum(returns[mask, date+2][bot_k] * free_shares[mask, date+1][bot_k]) /
                             np.sum(returns[mask, date+1][bot_k] * free_shares[mask, date+1][bot_k]) * bot_k_returns[-1])

    plt.plot(dates, 100. * (np.array(top_k_returns) - 1), label=f"{mname}高评分{n_top}组合")
    plt.plot(dates, 100. * (mindex.values[:, 1] / mindex.values[0, 1] - 1.), label=f"{mname}")
    plt.plot(dates, 100. * (np.array(bot_k_returns) - 1), label=f"{mname}低评分{n_top}组合")
    plt.xlabel('日期')
    plt.ylabel("区间累计回报（%）")
    plt.legend()


if __name__ == "__main__":
    import os

    from ESG.config import TEST_DATA_DIR, RAW_DATA_DIR, FIGURE_DIR, RESULT_DIR

    score_df = pd.read_excel(os.path.join(TEST_DATA_DIR, 'esg_score_202012.xlsx'),
                             sheet_name='all_score',
                             dtype={'STOCK_CODE': str})[['STOCK_CODE', 'REPORT_YEAR', 'ESG_SCORE', 'INDUSTRY']]

    score_df = score_df[score_df['STOCK_CODE'].apply(len) == 6].reset_index(drop=True)

    zz_stock_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'stock300_500.csv'))
    sz_stock_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'stock300.csv'))

    total_score_df = score_df

    zz_stock_df['STOCK_CODE'] = zz_stock_df['STOCK_CODE'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    zz_score_df = pd.merge(zz_stock_df, score_df, how='left', on='STOCK_CODE')

    sz_stock_df['STOCK_CODE'] = sz_stock_df['STOCK_CODE'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    sz_score_df = pd.merge(sz_stock_df, score_df, how='left', on='STOCK_CODE')

    zz_800_df = pd.read_csv(os.path.join(r"D:\QuantData\中证指数", '中证800指数（2017-2019）.csv'))
    dates = pd.to_datetime(zz_800_df['DATETIME'])

    zz_800_close = pd.read_csv(os.path.join(r"D:\QuantData\中证800成分股", '中证800股收盘价（2017-2019）.csv'))
    zz_800_fs = pd.read_csv(os.path.join(r"D:\QuantData\中证800成分股", '中证800股自由流通市值（2017-2019）.csv'))

    stock_codes = list(map(lambda sc: sc.split('.')[0], zz_800_close['STOCK_CODE'].values))
    set(list(map(lambda sc: '0'*(6-len(str(sc))) + str(sc), zz_score_df['STOCK_CODE'].values))) - set(stock_codes)

    factor_scores = zz_800_close[['STOCK_CODE']].copy()
    zz_score_df = zz_score_df.sort_values(['STOCK_CODE']).reset_index(drop=True)

    for d in dates:
        # factor_scores[d._date_repr] = zz_score_df[zz_score_df['REPORT_YEAR']==d.year]['ESG_SCORE'].values
        factor_scores.loc[:, d._date_repr] = zz_score_df.loc[zz_score_df['REPORT_YEAR']==d.year, 'ESG_SCORE'].values

    plt.figure(figsize=(20, 8), dpi=60)
    fs_weighted_top_factor_return_rate(factor_scores, zz_800_fs, zz_800_close, zz_800_df, mname='中证800')
    plt.savefig(os.path.join(FIGURE_DIR, "中证800ESG投资组合累计收益.png"))

