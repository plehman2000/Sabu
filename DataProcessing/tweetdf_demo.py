# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:58:49 2021

@author: 19522
"""

import csv
import twint
import nest_asyncio
import pandas as pd

    
nest_asyncio.apply()
c = twint.Config() 
c.Search = "from:@nbcnews" 
c.Store_object = True 
c.Limit=20
c.Pandas=True
#c.Store_csv = True
#c.Output = 'nbc.csv'
data = twint.run.Search(c) 
Tweets_df = twint.storage.panda.Tweets_df;
    
