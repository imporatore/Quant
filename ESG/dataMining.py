# -*- coding: utf-8 -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r'D:\QuantData\ESG\esg_3levels_basevalue.csv')
df_hs300 = pd.read_csv(r'D:\QuantData\ESG\stock300_500.csv')
df_indicator = pd.read_csv(r'D:\QuantData\ESG\indicator_code.csv')

strcols = df.columns[[0, 1, 2, 3, 5, 7, 8, 9]]
df[strcols] = pd.DataFrame(df[strcols], dtype='str')
df['LEVEL3_CODE'] = df['CSR_INDICATOR_CODE'].apply(lambda x: x[:7] if len(x) >= 7 else x)
df_hs300 = df_hs300.merge(df, left_on='STOCK_CODE', right_on='TICKER', how='left')

df_indicator['编码'] = df_indicator['编码'].apply(lambda x: x[2:] if type(x)==str else np.nan)

level1_codes = df['LEVEL1_CODE'].unique().tolist()
level2_codes = df['LEVEL2_CODE'].unique().tolist()
level3_codes = df['LEVEL3_CODE'].unique().tolist()
#codes = level1_codes + level2_codes + level3_codes
indicator_codes = df['CSR_INDICATOR_CODE'].unique().tolist()
codes = indicator_codes
codes.sort()

###############################################################################
df_stat = pd.DataFrame(columns=['CODE', 'none_zero_pct', 'none_zero_pct_in_hs300'])
df_stat['CODE'] = codes

for i in range(df_stat.shape[0]):
    code = df_stat.loc[i, 'CODE']
    df_temp1 = df[df['CSR_INDICATOR_CODE'] == code]
    df_temp2 = df_temp1[df_temp1['BASE_VALUE'] > -10000]
    df_temp2 = df_temp2[df_temp2['BASE_VALUE'] != 0]
    n = df_temp1.shape[0]
    m = df_temp2.shape[0]   
    df_stat.loc[i, 'none_zero_pct'] = m/n
    
    df_temp1 = df_hs300[df_hs300['CSR_INDICATOR_CODE'] == code]
    df_temp2 = df_temp1[df_temp1['BASE_VALUE'] > -10000]
    df_temp2 = df_temp2[df_temp2['BASE_VALUE'] != 0]
    n = df_temp1.shape[0]
    m = df_temp2.shape[0]
    df_stat.loc[i, 'none_zero_pct_in_hs300'] = m/n

df_indicator = df_indicator.merge(df_stat, left_on='编码', right_on='CODE', how='left')

###############################################################################


industries = df['INDUSTRY1'].unique().tolist()

df_stat = pd.DataFrame(columns=['CODE']+industries)
df_stat['CODE'] = codes

for industry in industries:
    print(industry)
    for i in range(df_stat.shape[0]):
        code = df_stat.loc[i, 'CODE']
        df_temp1 = df_hs300[df_hs300['CSR_INDICATOR_CODE'] == code]
        df_temp1 = df_temp1[df_temp1['INDUSTRY1'] == industry]
        df_temp2 = df_temp1[df_temp1['BASE_VALUE'] > -10000]
        df_temp2 = df_temp2[df_temp2['BASE_VALUE'] != 0]
        n = df_temp1.shape[0]
        m = df_temp2.shape[0]   
        df_stat.loc[i, industry] = m/n

df_indicator = df_indicator.merge(df_stat, left_on='编码', right_on='CODE', how='left')

#df_indicator.to_csv('D:\\ebscn\\data\\output\\df_stat_industry_hs300.csv')

df_indicator2 = df_indicator[df_indicator['none_zero_pct_in_hs300'] > 0.125].reset_index(drop=True)
hq_codes = list(df_indicator2['编码'].unique())

ESG_codes = [[] for i in range(3)]
for code in hq_codes:
    i = int(code[0])
    ESG_codes[i-1].append(code)
E_codes, S_codes, G_codes = ESG_codes

df_score = pd.DataFrame(columns=['STOCK_CODE'])
df_score['STOCK_CODE'] = stock_codes    

df_hs300['BASE_VALUE'] = df_hs300['BASE_VALUE'].apply(lambda x: x if x>-10000 else np.nan)

'''
def cal_score(base_value, quantiles):
    if base_value == np.nan:
        return 0
    n = len(quantiles) - 1
    lo = -(n-1)/2
    for i in range(n):
        if quantiles[i] <= base_value < quantiles[i+1]:
            return lo + i

count = 0
for code in hq_codes:
    print(count)
    df_temp1 = df_hs300[df_hs300['CSR_INDICATOR_CODE'] == code]
    
    base_values = np.array(df_temp1['BASE_VALUE'])
    base_values = base_values[~np.isnan(base_values)]
    df_temp2 = df_temp1[['STOCK_CODE', 'BASE_VALUE']]
    if set(base_values) == set([0,1]):
        df_temp2['BASE_VALUE'] = df_temp2['BASE_VALUE'].apply(lambda x: 0.5 if x==1 else -0.5)
    else:
        quantiles = []
        for i in range(6):
            q = np.quantile(base_values, i*0.2)
            quantiles.append(q)
        df_temp2['BASE_VALUE'] = df_temp2['BASE_VALUE'].apply(cal_score, quantiles=quantiles)
    df_score = df_score.merge(df_temp2, on='STOCK_CODE', how='left')
    df_score = df_score.rename(columns={'BASE_VALUE': code})
    count = count + 1
'''


def cal_score_by_normalize(df):
    indices = df[~np.isnan(df['BASE_VALUE'])].index.tolist()
    base_values = np.array(df.loc[indices, 'BASE_VALUE']).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(base_values)
    scores = scaler.transform(base_values).reshape((len(indices),))
    df['SCORE'] = np.nan
    df.loc[indices, 'SCORE'] = scores
    return df

for code in hq_codes:
    df_temp1 = df_hs300[df_hs300['CSR_INDICATOR_CODE'] == code]
    df_temp2 = df_temp1[['STOCK_CODE', 'BASE_VALUE']]
    df_temp2['SCORE'] = 0
    base_values = set(df_temp2['BASE_VALUE'])
    if base_values <= set([-1,0,1,np.nan]):
        df_temp2['SCORE'] = df_temp2['BASE_VALUE'].apply(lambda x: x-0.5)
    else:
        df_temp2 = cal_score_by_normalize(df_temp2)
    df_score = df_score.merge(df_temp2[['STOCK_CODE', 'SCORE']], on='STOCK_CODE', how='left')
    df_score = df_score.rename(columns={'SCORE': code})
    
    
    




    
    
df_score.fillna(0, inplace=True)

df_score['E_score'] = np.mean(df_score[E_codes], axis=1)
df_score['S_score'] = np.mean(df_score[S_codes], axis=1)
df_score['G_score'] = np.mean(df_score[G_codes], axis=1)
df_score['ESG_score'] = np.mean(df_score[['E_score', 'S_score', 'G_score']], axis=1)



scores = ['E_score', 'S_score', 'G_score']
for industry in industries:
    stocks = df_hs300[df_hs300['INDUSTRY1'] == industry]['STOCK_CODE'].unique()
    df_temp = pd.DataFrame(columns=['STOCK_CODE'])
    df_temp['STOCK_CODE'] = stocks
    df_temp = df_temp.merge(df_score, on='STOCK_CODE', how='left')
    fig, ax = plt.subplots(2,2)
    fig.tight_layout()
    ax[0,0].hist(df_temp['E_score'])
    ax[0,0].set_title('E_score')
    ax[0,1].hist(df_temp['S_score'])
    ax[0,1].set_title('S_score')
    ax[1,0].hist(df_temp['G_score'])
    ax[1,0].set_title('G_score')
    ax[1,1].hist(df_temp['ESG_score'])
    ax[1,1].set_title('ESG_score')
    fig.savefig('D:\\ebscn\\data\\output\\temp_plots1\\'+industry+'.png')
    


###############################################################################

df_temp = df_hs300[df_hs300['INDUSTRY1'] == '休闲服务']
stock_temp = df_temp['STOCK_CODE'].unique().tolist()
df_temp = pd.DataFrame()
df_temp['STOCK_CODE'] = stock_temp
df_temp = df_temp.merge(df_score, on='STOCK_CODE', how='left')

