# -*- coding: utf-8 -*-

"""
Twint wrapper module
"""

import nest_asyncio
import twint
import html
import regex
import nltk
from nltk import word_tokenize as tokenize
#from twint_wrapper import get_tweets
#from process import process_tweet
import spacy
import random
#from wikiUtil import wikiExplainer
import requests


#TODO fix imports

HASHTAG_CHARS = r"\p{L}\p{M}\p{Nd}_\u200c\u200d\ua67e\u05be\u05f3\u05f4\uff5e\u301c\u309b\u309c\u30a0\u30fb\u3003\u0f0b\u0f0c\u00b7"

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




def process_tweet(tweet):
    """Remove all URLs (e.g. www.xyz.com), hash tags (e.g. #topic), targets (@username)"""
    processed = tweet.copy()

    processed["tweet"] = regex.sub(r"https?://t\.co/[a-zA-Z0-9]+",
                                "", processed["tweet"])

    processed["tweet"] = regex.sub(r"(?:([^\w!#$%&*@＠]|^)|(?:^|[^\w+~.-])(?:rt|rT|Rt|RT):?)[@＠](\w{1,20}(?:/[a-zA-Z][\w-]{0,24})?)",
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
          if token not in output_dict.keys():
            output_dict.update({token : 1})
          else:
            output_dict[token] += 1

    return output_dict


