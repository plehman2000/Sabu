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

def get_tweets(search, limit, popular_tweets = False):
    """Get Tweets that match given search"""
    c.Search = search
    c.Limit = limit
    # Popular_tweets scrapes popular tweets if True, most recent if False
    c.Popular_tweets = popular_tweets
    # Run the search on the Twint object
    twint.run.Search(c)
    # Return the search results in a pandas DataFrame
    return [tweet.__dict__ for tweet in twint.output.tweets_list]
