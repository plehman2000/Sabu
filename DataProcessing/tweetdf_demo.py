# -*- coding: utf-8 -*-
"""
Simple Demonstration of using Twint to gather tweets from a particular account
and store the tweets in a Pandas dataframe.
"""

import csv
import twint
import nest_asyncio
import pandas as pd

# Required to use Twint    
nest_asyncio.apply()
# Create Twint object for a web-scrape query
c = twint.Config() 
# Designate the search to take place from the @nbcnews twitter account
c.Search = "from:@nbcnews" 
# We are storing the results in an object
c.Store_object = True 
# Limit the search to 20 results (most recent in this case)
c.Limit=20
# Necessary to incorporate pandas dataframe functionality
c.Pandas=True
#c.Store_csv = True
#c.Output = 'nbc.csv'
# Run the search on our 'c' twint object
data = twint.run.Search(c) 
# Store the search results in a 'Tweets_df' pandas df
Tweets_df = twint.storage.panda.Tweets_df;
