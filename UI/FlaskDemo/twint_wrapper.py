# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:27:03 2021

Test Twint Wrapper

@author: 19522
"""
import nest_asyncio
import twint
import pandas
# Required to use Twint
#nest_asyncio.apply()
# Create Twint object for a web-scrape query

# Store_object stores Tweets in list

# Remove non-English Tweets


def get_tweets(search, limit, popular_tweets = False):
    c = twint.Config()
    c.Store_object = True
    c.Hide_output = True
    c.Pandas = True
    c.Lang = "en"
    """Get Tweets that match given search"""
    c.Search = search
    #NOTE:"So even when you specify the limit below 100 it will still return 100, try to specify the limit in multiples of 100."
    c.Limit = limit
    # Popular_tweets scrapes popular tweets if True, most recent if False
    c.Popular_tweets = popular_tweets
    # Run the search on the Twint object
    twint.run.Search(c)
    # Return the search results in a pandas DataFrame
    #list comprehension hard limits number of tweets
    Tweets_df = twint.storage.panda.Tweets_df
    #var = []
    #var = [dict(tweet.__dict__) for i,tweet in enumerate(twint.output.tweets_list) if i < limit]

    return Tweets_df

