# -*- coding: utf-8 -*-

"""
Test process
"""
import spacy
import random
from wikiUtil import wikiExplainer

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
        


from twint_wrapper import get_tweets
from process import process_tweet


def analyzeTweetNouns(user, numTweets, explainerLength=3):
    nlp = spacy.load('en_core_web_sm')    
    
    all_noun_dictionaries = []
    for tweet in get_tweets('@potus', 1):
        tempDict = {}
        processed_tweet=process_tweet(tweet)["tweet"]
    
        
        colors, nouns = highlightNouns(processed_tweet)
        for i, noun in enumerate(nouns):
            
            doc = nlp(wikiExplainer(noun))
            sentence_IDs = [sent_id for sent_id, sent in enumerate(doc.sents) for token in sent]                                       
            
            explainer = ""
            for j,sentence in enumerate(doc.sents):
                if (j+1) > explainerLength:
                    break
                explainer += str(sentence.text)
            if explainer != "":
                tempDict[noun] = explainer
                noun_context =  " - " + explainer
                print(colored([255,0,0], noun), colored(colors[i], noun_context)) 
        all_noun_dictionaries.append(tempDict)
        
  
    return all_noun_dictionaries

        
dictionaries = analyzeTweetNouns(user='@potus', numTweets=1)
#print(dictionaries[0])
    #fix 'may refer to' edge case, example is 'networks'
        


    

    

    
