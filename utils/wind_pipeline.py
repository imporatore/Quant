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

            if err_code != 0:
                raise RuntimeWarning("Wind API request error.")

        return wind_result
    return wrapper


@wind_api_decorator




def


if __name__ == '__main__':
    pass
