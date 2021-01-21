from WindPy import w

import warnings


def wind_api_decorator(func):
    def wrapper(*args, **kwargs):
        w.start()

        if not w.isconnected():
            raise RuntimeError('Wind API not connected.')

        wind_result = func(*args, **kwargs)

        if type(wind_result) == w.WindData:
            err_code = wind_result.ErrorCode

            if err_code == -40522007:
                raise ValueError("Field index not supported.")
            elif err_code != 0:
                raise RuntimeWarning("Wind API request error.")

        return wind_result
    return wrapper


@wind_api_decorator
def stock_basic_series(codes, fields, begin=None, end=None, day='Trading',
                       period='D', priceadj='', options="", usedf=False, *args, **kwargs):
    opts = options.split(';')
    if day != 'D' and 'Days' not in options:
        opts.append(f"Days={day}")
    if period != 'D' and 'period' not in options:
        opts.append(f"Period={period}")
    if priceadj and 'PriceAdj' not in options:
        opts.append(f"PriceAdj={priceadj}")
    options = ';'.join(opts)

    return w.wsd(codes, fields, beginTime=begin, endTime=end, options=options, usedf=usedf, *args, **kwargs)


if __name__ == '__main__':
    s = stock_basic_series('596', ["CLOSE", "MKT_CAP_FLOAT"], "2019-01-01", "2019-10-10")
pass
