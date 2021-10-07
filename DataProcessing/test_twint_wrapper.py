# -*- coding: utf-8 -*-

"""
Test twint_wrapper
"""

import json
from twint_wrapper import get_tweets

print(json.dumps(get_tweets('from:@nbcnews', 20), indent=4))
