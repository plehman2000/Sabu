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
# Necessary to incorporate pandas DataFrame functionality
c.Pandas = True

def get_tweets(search, limit, popular_tweets = False):
    """Get Tweets that match given search"""
    c.Search = search
    c.Limit = limit
    # popular_tweets scrapes popular tweets if True, most recent if False
    c.Popular_tweets = popular_tweets
    # Run the search on the Twint object
    twint.run.Search(c)
    # Return the search results in a pandas DataFrame
    return twint.storage.panda.Tweets_df
