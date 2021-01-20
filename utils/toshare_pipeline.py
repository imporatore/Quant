import tushare as ts

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings

from utils.config import USER_TOKEN


def fetch_stocks_history(stock_codes, token=USER_TOKEN, *args):
    """
    Params:
        stock_codes: iterable, either list/numpy.array/pd.Series.

    Return:
        pd.Dataframe, historical data of stocks, sample:
           stock_code       date      open    high   close     low     volume    p_change  ma5    ...
             600848      2012-01-11   6.880   7.380   7.060   6.880   14129.96     2.62   7.060   ...
             600848      2012-01-12   7.050   7.100   6.980   6.900    7895.19    -1.13   7.020   ...
             600848      2012-01-13   6.950   7.000   6.700   6.690    6611.87    -4.01   6.913   ...
             600848      2012-01-16   6.680   6.750   6.510   6.480    2941.63    -2.84   6.813   ...
             600849      2012-01-11   6.880   7.380   7.060   6.880   14129.96     2.62   7.060   ...
             600849      2012-01-12   7.050   7.100   6.980   6.900    7895.19    -1.13   7.020   ...
             600849      2012-01-13   6.950   7.000   6.700   6.690    6611.87    -4.01   6.913   ...
             600849      2012-01-16   6.680   6.750   6.510   6.480    2941.63    -2.84   6.813   ...
    """
    stock_codes = np.array(stock_codes, dtype='str').reshape(-1)
    stock_history = pd.DataFrame()
    pro = ts.pro_api(token)
    for code in tqdm(stock_codes):
        if len(code) > 6:
            raise ValueError(f'Stock code {code} invalid.')
        elif len(code) < 6:
            code = '0' * (6-len(code)) + code
        try:
            history = pro.daily_basic(ts_code=code, *args)
        except Exception:
            history = ts.get_hist_data(code, *args)
        try:
            history['stock_code'] = np.repeat(code, history.shape[0])
            stock_history = stock_history.append(history)
        except AttributeError:
            warnings.warn(f"Stock code {code} not found.", stacklevel=2)

    return stock_history


if __name__ == '__main__':
    daily_history = fetch_stocks_history(pd.read_csv(r'D:\QuantData\ESG\stock300_500.csv'))
    daily_history.to_csv(r'D:\QuantData\daily_history.csv')
