from flask import Flask, render_template, request, flash
from twint_wrapper import get_tweets
from wikiUtil import wikiExplainer
import twint
import numpy as np
import nest_asyncio
import pandas
from getToxicity import getToxicity
from getSentiment import *
from getToxicity import *



#nest_asyncio.apply()

# SET FLASK ENV VARIABLE FOR TEST RUN:
# $env:FLASK_APP = "market.py"
# FLASK DEBUG MODE:
# $env:FLASK_ENV = "development"

app = Flask(__name__)
app.secret_key = "my_password888888"

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')
@app.route('/search', methods=['POST', 'GET'])
def query_results():
    if request.method == 'POST':
        popular_tweets = False
        limit = 5 # Default limit of 20
        if ('popular_tweets' in request.form):
            popular_tweets = True

        print(str(request.form['name_input']))
        data_dict = get_tweets(str(request.form['name_input']), 20, True)
    else:
        return render_template('search.html')
    # Convert dataframe to dictionary since Jinja is not compatible with DF's
    items = data_dict.tweet.to_dict()
    print(items)
    #tweets = [tweet['tweet'] for tweet in data_dict]
    #list(items.values())
    # Test chart.js
    data = [
        ("01-01-2020", 1597),
        ("02-01-2020", 1456),
        ("03-01-2020", 1908),
        ("04-01-2020", 896),
        ("05-01-2020", 755),
        ("06-01-2020", 453),
        ("07-01-2020", 1100),
        ("08-01-2020", 1235),
        ("09-01-2020", 1478),
    ]

    scores = getSentimentAverage(list(items.values()))
    search_query = str(request.form['name_input'])
    print(f'Tweets found using {search_query} were {100*scores[0]:.2f}% Negative, {100*scores[1]:.2f}% Neutral, {100*scores[2]:.2f}% Positive')



    # get Toxcity test
    sentences = np.asarray(list(items.values()))

    """
    sentence_tox_pairs = getToxicity(sentences)

    for pair in sentence_tox_pairs:
        print(f'Sentence: {pair[0]}')
        if len(pair[1]) > 0:
            print(pair[1], pair[2])
    """
    #actual_tweets = items.values()
    #print(actual_tweets)
    #items_np = np.fromiter(items.values(),dtype='S128')
    #print(items_np)
    return render_template('search.html', items=items)

@app.route('/search')
def search_page():
    items = []
    return render_template('search.html', items=items)

@app.route('/tweet')
def show_tweet():
    return render_template('tweet.html')

@app.route('/about')
def about_page():
    return render_template('about.html')


# Below is an example of a dynamic page, will be useful when creating custom pages for gathered tweets.
# Creates page about '<username>' dynamic route creates /'John' when referenced
#@app.route('/about/<username>')
#def about_page(username):
#    return f'<h1>This is the about page of {username}</h1>'

