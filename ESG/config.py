import os
from utils.config import DATA_DIR

ESG_DATA_DIR = os.path.join(DATA_DIR, 'ESG')
RAW_DATA_DIR = os.path.join(ESG_DATA_DIR, 'raw_data')
TEST_DATA_DIR = os.path.join(ESG_DATA_DIR, 'test_data')
FIGURE_DIR = os.path.join(ESG_DATA_DIR, 'figure')
RESULT_DIR = os.path.join(ESG_DATA_DIR, 'result')

if __name__ == '__main__':
    pass
