# -*- coding: utf-8 -*-

"""
Twint wrapper module
"""

import sys
import os
import html
import nest_asyncio
import regex
from nltk import word_tokenize as tokenize
import twint

sys.path.append(os.getcwd() + r'\DataProcessing')

HASHTAG_CHARS = (r"\p{L}\p{M}\p{Nd}_\u200c\u200d\ua67e\u05be\u05f3\u05f4\uff5e\u301c\u309b\u309c"
                 r"\u30a0\u30fb\u3003\u0f0b\u0f0c\u00b7")

OPTIONS = ["Username", # scrapes tweets from the specified user
           "User_id", # scrapes tweets from the user with the specified ID
           "Search", # scrapes tweets associated with the specified search terms
           "Geo", # format lat,lon,radius with lat, lon in decimal degrees and radius ending with
                  # "km" or "mi", scrapes tweets within radius of (lat, lon)
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

# Required to use Twint
nest_asyncio.apply()

def get_tweets(**kwargs):
    """Get tweets that match given search"""
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
    return Tweets_df

def process_tweet(tweet):
    """Remove all URLs (e.g. www.xyz.com), hash tags (e.g. #topic), targets (@username)"""
    processed = tweet.copy()

    processed["tweet"] = regex.sub(r"https?://t\.co/[a-zA-Z0-9]+",
                                "", processed["tweet"])

    processed["tweet"] = regex.sub(r"(?:([^\w!#$%&*@＠]|^)|(?:^|[^\w+~.-])(?:rt|rT|Rt|RT):?)[@＠]"
                                   r"(\w{1,20}(?:/[a-zA-Z][\w-]{0,24})?)",
                                   r"\1\2", processed["tweet"])

    processed["tweet"] = regex.sub(r"(^|\ufe0e|\ufe0f|[^&" +
                                HASHTAG_CHARS +
                                r"])[#＃]((?!\ufe0f|\u20e3)[" +
                                HASHTAG_CHARS +
                                r"]*[\p{L}\p{M}][" +
                                HASHTAG_CHARS +
                                r"]*)",
                                r"\1\2", processed["tweet"])

    processed["tweet"] = regex.sub(r"\n+",
                                "\n", processed["tweet"])

    processed["tweet"] = regex.sub(r"\s+",
                                " ", processed["tweet"]).strip()

    processed["tweet"] = html.unescape(processed["tweet"])

    return processed


def detect_profanity(processed_tweet_str):
    """ Detect profane words in list; output words and their number of occurences"""

    # Need to update profane_words list; only need lowercase words.
    profane_words = []
    output_dict = {}
    tokens = tokenize(processed_tweet_str)

    for token in tokens:
        if token in profane_words:
            if token not in output_dict:
                output_dict.update({token : 1})
            else:
                output_dict[token] += 1

    return output_dict
