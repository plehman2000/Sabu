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



def colored(color_rgb, text):
    try:
        colors = "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(color_rgb[0], color_rgb[1], color_rgb[2], text)
    except:
        print("Exception")
        colors=[]
    return colors
    

def highlightNouns(tweet):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(tweet)
    
    colors = []
    nouns=[]

            
    sentenceLimit = 3
    token_sents = [sent_id for sent_id, sent in enumerate(doc.sents) for token in sent]

            
    sF=11
    print("\n\n")
    for token in doc:

        if token.pos_=="NOUN":# or token.pos_=="PNOUN..":
            nouns.append(token.text)
            color_rgb = [random.randint(10,15) * sF, random.randint(10,15) * sF, random.randint(10,15) * sF]
            colors.append(color_rgb)
            print(colored(color_rgb, token.text), end=" ")
        else:
            print(token.text, end=' ')
        
    print("\n")
    return colors, nouns
        




def analyzeTweetNouns(tweet, explainerLength=3):
    nlp = spacy.load('en_core_web_sm')    
    all_noun_dictionaries = []
    
    tempDict = {}
    processed_tweet=process_tweet(tweet)["tweet"]
    #processed_tweet=input("Enter Input: ")

    colors, nouns = highlightNouns(processed_tweet)
    for i, noun in enumerate(nouns):

        doc = nlp(wikiExplainer(noun))
        sentence_IDs = [sent_id for sent_id, sent in enumerate(doc.sents) for token in sent]                                       

        explainer = ""
        for j,sentence in enumerate(doc.sents):
            if (j+1) > explainerLength:
                break
            explainer += str(sentence.text) + " "
        if explainer != "":
            tempDict[noun] = explainer
            noun_context =  " - " + explainer
            print(colored([255,0,0], noun), colored(colors[i], noun_context)) 
    all_noun_dictionaries.append(tempDict)
        
  
    return all_noun_dictionaries

    

def wikiExplainer(title, removeEscapeChars=True):
        
    response = requests.get(
         'https://en.wikipedia.org/w/api.php',
         params={
             'action': 'query',
             'format': 'json',
             'titles': title,
             'prop': 'extracts',
             'exintro': True,
             'explaintext': True,
         }).json()
    response = requests.get("https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&exintro&titles=" + title.replace(" ", "_") + "|" + title.replace(" ", "_") + "&redirects=").json()
    page = next(iter(response['query']['pages'].values()))
    if 'extract' in page:
        page = page['extract']
        if removeEscapeChars:
            page = ''.join(c for c in page if c.isalnum() or c==' ')
    else:
        page = ""
    return page
