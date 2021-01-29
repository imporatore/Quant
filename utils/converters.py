import pandas as pd
import numpy as np


def to_numpy(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data

    return np.array(data)
