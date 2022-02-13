# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:27:03 2021

Test Twint Wrapper

@author: 19522
"""

import twint
import pandas
# Required to use Twint
#nest_asyncio.apply()
# Create Twint object for a web-scrape query

# Store_object stores Tweets in list

# Remove non-English Tweets
OPTIONS = ["Username", # scrapes tweets from the specified user
           "User_id", # scrapes tweets from the user with the specified ID
           "Search", # scrapes tweets associated with the specified search terms
           "Geo", # format lat,lon,radius with lat, lon in decimal degrees and radius ending with
           "Near", # toponym, scrapes tweets near the specified location
           "Year", # scrapes tweets before the specified year
           "Since", # format YYYY-MM-DD, scrapes tweets after the specified date
           "Until", # format YYYY-MM-DD, scrapes tweets before the specified date
           "Verified", # scrapes tweets by verified users
           "Limit", # NOTE: even when you specify the limit below 20 it will still return 20
           "To", # scrapes tweets sent to the specified user
           "All", # scrapes tweets sent to or from or mentioning the specified user
           "Images", # scrapes tweets with images
           "Videos", # scrapes tweets with videos
           "Media", # scrapes tweets with images or videos
           "Popular_tweets", # scrapes popular tweets if True, most recent if False
           "Native_retweets", # scrapes native retweets
           "Min_likes", # scrapes tweets with at least the specified number of likes
           "Min_retweets", # scrapes tweets with at least the specified number of retweets
           "Min_replies", # scrapes tweets with at least the specified number of replies
           "Links", # includes tweets with links if "include", excludes if "exclude"
           "Source", # scrapes tweets sent with the specified source client
           "Members_list", # list ID, scrapes tweets by users in the specified list
           "Filter_retweets"] # scrapes non-retweets

def get_tweets(**kwargs):
    """Get tweets that match given search"""
    try:
        # Create Twint object for a web-scrape query
        config = twint.Config()
        # Store_object stores tweets in list
        config.Store_object = True
        config.Hide_output = True
        # Remove non-English tweets
        config.Lang = "en"
        # Tweets are scraped in batches of 20
        config.Limit = 20
        # If Twitter says there is no data, Twint retries to scrape Retries_count times
        config.Retries_count = 0
        for option in OPTIONS:
            if option in kwargs:
                vars(config)[option] = kwargs[option]
        #print(config)
        config.Pandas = True
        twint.output.tweets_list = []
        # Run the search on the Twint object
        twint.run.Search(config)
        # List comprehension hard-limits number of tweets
        Tweets_df = twint.storage.panda.Tweets_df
        #return [tweet.__dict__ afor i,tweet in enumerate(twint.output.tweets_list) if i < config.Limit]
    except:
        Tweets_df =pandas.DataFrame
    return Tweets_df

