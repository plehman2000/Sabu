import requests

def wikiExplainer(title):
        
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
    else:
        page = ""
    return page


