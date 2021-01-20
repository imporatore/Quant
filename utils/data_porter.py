import os

import pandas as pd


def save_to_csv(df, file_dir, fname, index=False):
    df.to_csv(os.path.join(file_dir, fname + '.csv'), index=index, encoding="utf_8_sig")


if __name__ == '__main__':
    pass
