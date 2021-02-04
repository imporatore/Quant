import numpy as np
import pandas as pd

from utils import to_numpy
from utils.plot import *


def weighted_top_factor_return_rate(factor_score, weights, returns, mindex, n_top=20, mname=''):
    """
    Compute weighted value for stocks with

    top-k & bottom-k factor scores, all parameters should have
    corresponding rank of axis datetime & stock.

    Params:
        - factor_score: pd.DataFrame, dir: +, sample:
        STOCK_CODE        DATE1        DATE2        DATE3
         xxxxxx            xxx          xxx          xxx
         xxxxxx            xxx          xxx          xxx

        - weights: pd.DataFrame, sample:
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
    weights = to_numpy(weights)
    returns = to_numpy(returns)

    top_k_returns = [1., ]
    bot_k_returns = [1., ]

    for date in range(factor_score.shape[1] - 2):
        mask = (weights[:, date + 1] > 0)
        factor_rank = sorted(np.arange(np.sum(mask)),
                             key=lambda i: factor_score[mask, date+1][i],
                             reverse=True)
        top_k = factor_rank[:n_top]
        bot_k = factor_rank[-n_top:]
        top_k_returns.append(np.sum(returns[mask, date+2][top_k] * weights[mask, date + 1][top_k]) /
                             np.sum(returns[mask, date+1][top_k] * weights[mask, date + 1][top_k]) * top_k_returns[-1])
        bot_k_returns.append(np.sum(returns[mask, date+2][bot_k] * weights[mask, date + 1][bot_k]) /
                             np.sum(returns[mask, date+1][bot_k] * weights[mask, date + 1][bot_k]) * bot_k_returns[-1])

    plt.plot(dates, 100. * (np.array(top_k_returns) - 1), label=f"{mname}高评分{n_top}组合")
    plt.plot(dates, 100. * (mindex.values[:, 1] / mindex.values[0, 1] - 1.), label=f"{mname}")
    plt.plot(dates, 100. * (np.array(bot_k_returns) - 1), label=f"{mname}低评分{n_top}组合")
    plt.xlabel('日期')
    plt.ylabel("区间累计回报（%）")
    plt.legend()


def weighted_ind_neutral_bins_return_rate(factor_score, industry, weights, returns,
                                                mindex, n_bins=5, mname=''):
    """
    Compute weighted value for stocks with

    top-k & bottom-k factor scores, all parameters should have
    corresponding rank of axis datetime & stock.

    Params:
        - factor_score: pd.DataFrame, dir: +, sample:
        STOCK_CODE        DATE1        DATE2        DATE3
         xxxxxx            xxx          xxx          xxx
         xxxxxx            xxx          xxx          xxx

        - weights: pd.DataFrame, sample:
        STOCK_CODE        DATE1        DATE2        DATE3
         xxxxxx            xxx          xxx          xxx
         xxxxxx            xxx          xxx          xxx

        - industry: pd.DataFrame, sample:
        STOCK_CODE        INDUSTRY
         xxxxxx             xxx
         xxxxxx             xxx

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
    weights = to_numpy(weights)
    returns = to_numpy(returns)

    n_returns = [[1., ] for _ in range(n_bins)]

    for date in range(factor_score.shape[1] - 2):
        mask = (weights[:, date + 1] > 0)
        factor_rank = sorted(np.arange(np.sum(mask)),
                             key=lambda i: factor_score[mask, date+1][i],
                             reverse=True)
        industry_factor_rank = [np.array(factor_rank)[i] for i in
                                industry.loc[mask, :].groupby(by=['INDUSTRY']).indices.values()]
        industry_k = [len(i) / n_bins for i in industry_factor_rank]

        for i in range(n_bins):
            cur = list()
            weight = list()

            for r, j in zip(industry_factor_rank, industry_k):
                cur.extend(r[range(int(np.floor(j*i)), int(np.ceil(j*(i+1))))])
                if j*(i+1) > np.ceil(j*i + 1e-10):
                    weight.append(np.ceil(j*i + 1e-10) - j*i)
                    weight.extend([1] * (len(range(int(np.floor(j*i)), int(np.ceil(j*(i+1)))))-2))
                    weight.append(j*(i+1) - np.floor(j*(i+1) - 1e-10))
                else:
                    weight.append(j)

            n_returns[i].append(np.sum(returns[mask, date+2][cur] * weights[mask, date + 1][cur] * np.array(weight)) /
                                np.sum(returns[mask, date+1][cur] * weights[mask, date + 1][cur] * np.array(weight))
                                * n_returns[i][-1])

    for i in range(n_bins):
        plt.plot(dates, 100. * (np.array(n_returns[i]) - 1), label=f"{mname}第{i+1}高评分组合")
    plt.plot(dates, 100. * (mindex.values[:, 1] / mindex.values[0, 1] - 1.), label=f"{mname}")
    plt.xlabel('日期')
    plt.ylabel("区间累计回报（%）")
    plt.legend()


if __name__ == "__main__":
    # pass
    import os

    from ESG.config import TEST_DATA_DIR, RAW_DATA_DIR, FIGURE_DIR, RESULT_DIR

    new_score_df = pd.read_csv(os.path.join(TEST_DATA_DIR, 'scores_new.csv'), dtype={'STOCK_CODE': str})
    new_score_df['STOCK_CODE'] = new_score_df['STOCK_CODE'].apply(lambda x: '0' * (6 - len(x)) + x)
    new_score_df = new_score_df.sort_values(by=['STOCK_CODE']).reset_index(drop=True)

    zz_800_df = pd.read_csv(os.path.join(r"D:\QuantData\指数", '指数收盘价（18-20）.csv'), parse_dates=['DATETIME']).iloc[:,
                [0, 2]]
    zz_800_close = pd.read_csv(os.path.join(r"D:\QuantData\中证800成分股", '中证800股收盘价（2018-2020）.csv'))
    zz_800_fs = pd.read_csv(os.path.join(r"D:\QuantData\中证800成分股", '中证800股自由流通市值（2018-2020）.csv'))
    zz_800_ind = pd.read_csv(os.path.join(r"D:\QuantData\中证800成分股", '中证800股行业.csv'))

    new_score_df['STOCK_CODE'] = zz_800_close['STOCK_CODE'].copy()

    new_sz_score_df = pd.read_csv(os.path.join(TEST_DATA_DIR, 'scores_new_300.csv'), dtype={'STOCK_CODE': str})
    new_sz_score_df['STOCK_CODE'] = new_sz_score_df['STOCK_CODE'].apply(lambda x: '0' * (6 - len(x)) + x)
    new_sz_score_df = new_sz_score_df.sort_values(by=['STOCK_CODE']).reset_index(drop=True)

    mask = (zz_800_close['STOCK_CODE'].apply(lambda x: x.split('.')[0] in new_sz_score_df['STOCK_CODE'].values))
    sz_300_df = pd.read_csv(os.path.join(r"D:\QuantData\指数", '指数收盘价（18-20）.csv'), parse_dates=['DATETIME']).iloc[:,
                [0, 1]]
    sz_300_close = zz_800_close.loc[mask, :].copy().reset_index(drop=True)
    sz_300_ind = zz_800_ind.loc[mask, :].copy().reset_index(drop=True)

    new_sz_score_df['STOCK_CODE'] = sz_300_close['STOCK_CODE'].copy()
    dates = pd.to_datetime(sz_300_df['DATETIME'])

    sz_300_weight_ = pd.read_csv(os.path.join(r"D:\QuantData\指数", '沪深300指数权重（18-20）.csv'), parse_dates=['TRADE_DT'])

    factor_scores = sz_300_close[['STOCK_CODE']].copy()

    for d in dates:
        factor_scores.loc[:, d._date_repr] = new_sz_score_df.loc[:, 'ESG_SCORE_' + str(d.year - 1)].values

    sz_300_weight = sz_300_close[['STOCK_CODE']].copy()

    def fetch_data(sc, dates):
        df = sz_300_weight_.loc[sz_300_weight_['STOCK_CODE'] == sc, :]
        dat = list()
        cur = 0

        for d in dates:
            if d < df['TRADE_DT'].values[cur]:
                dat.append(float('nan'))
            else:
                if d >= df['TRADE_DT'].values[cur + 1]:
                    cur += 1
                dat.append(df['I_WEIGHT'].values[cur])
        return dat

    for sc in sz_300_weight['STOCK_CODE'].values:
        sz_300_weight.loc[sz_300_weight['STOCK_CODE'] == sc, dates] = fetch_data(sc, dates)

    plt.figure(figsize=(20, 8), dpi=60)
    weighted_top_factor_return_rate(factor_scores, sz_300_weight, sz_300_close, sz_300_df, mname='沪深300')
    plt.savefig(os.path.join(FIGURE_DIR, "沪深300ESG投资组合累计收益.png"))

    plt.figure(figsize=(20, 8), dpi=60)
    weighted_ind_neutral_bins_return_rate(factor_scores, sz_300_ind, sz_300_weight, sz_300_close, sz_300_df,
                                          mname='沪深300')
    plt.savefig(os.path.join(FIGURE_DIR, "沪深300行业中性ESG投资组合累计收益.png"))



