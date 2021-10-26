# -*- coding: utf-8 -*-

"""
Test process
"""

from twint_wrapper import get_tweets
from process import process_tweet

for tweet in get_tweets('@potus', 20):
    print(process_tweet(tweet)["tweet"])
    print(tweet['tweet'])
    break