#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:53:07 2020

@author: Sally
"""

import pandas as pd

data = pd.read_excel('https://github.com/SelinaDing/Sentiment-Analysis-on-NYSE-NASDAQ-Delisted-China-Stocks/blob/main/cik.xlsx?raw=true',index_col=None)
#print(data.head(3))
new_data = data.filter(['Ticker','CIK'],axis=1)
#cleaned dataset
#print(new_data.head(3))

#Method 1
CIK_dict1 = new_data.set_index('Ticker')['CIK'].to_dict()
print(CIK_dict1)
#Method 2
CIK_dict2 = dict(new_data.values.tolist())
print(CIK_dict2)
