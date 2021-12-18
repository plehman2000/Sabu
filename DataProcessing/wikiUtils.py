import spacy
import random
import requests





nlp = spacy.load('en_core_web_sm')

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
        




def analyzeTweetNouns(processed_tweet, explainerLength=3):
    #nlp = spacy.load('en_core_web_sm')    nlp is now defined globally
    all_noun_dictionaries = []
    
    tempDict = {}

    colors, nouns = highlightNouns(processed_tweet)
    for i, noun in enumerate(nouns):

        doc = nlp(wikiExplainer(noun))

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

    

def wikiExplainer(title, removeEscapeChars=True, explainerLength=3):
    
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
    explainer = next(iter(response['query']['pages'].values()))
    if 'extract' in explainer:
        explainer = explainer['extract']
        if removeEscapeChars:
            explainer = ''.join(c for c in explainer if c.isalnum() or c==' ')
            explainer = explainer.replace("\n", " ")
    else:
        explainer = ""


    doc = nlp(explainer)
    explainer = ""
    for j,sentence in enumerate(doc.sents):
        if(j+1 > explainerLength):
            break
        else:
            explainer += str(sentence.text) + " "
    return explainer
