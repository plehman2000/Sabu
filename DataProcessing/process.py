# -*- coding: utf-8 -*-
# pylint: disable=line-too-long,trailing-whitespace

"""
Tweet processing module
"""

import html
import regex

HASHTAG_CHARS = r"\p{L}\p{M}\p{Nd}_\u200c\u200d\ua67e\u05be\u05f3\u05f4\uff5e\u301c\u309b\u309c\u30a0\u30fb\u3003\u0f0b\u0f0c\u00b7"

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
