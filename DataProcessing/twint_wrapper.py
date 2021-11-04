# -*- coding: utf-8 -*-

"""
Twint wrapper module
"""

import nest_asyncio
import twint

# Required to use Twint
nest_asyncio.apply()
# Create Twint object for a web-scrape query
c = twint.Config()
# Store_object stores Tweets in list
c.Store_object = True
c.Hide_output = True
# Remove non-English Tweets
c.Lang = "en"

def get_tweets(search, limit, popular_tweets = False):
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
    return [tweet.__dict__ for i,tweet in enumerate(twint.output.tweets_list) if i < limit]
