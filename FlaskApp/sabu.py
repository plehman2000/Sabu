from flask import Flask, render_template, request, session
from twint_wrapper import get_tweets
from DataModeling.getEmotions.getEmotions import emotionInference
from DataModeling.getSentiment.getSentiment import sentimentInference
from DataModeling.getTox.getToxicity import toxInference

app = Flask(__name__)


#nest_asyncio.apply()

# First time setup
# 1. pip install flask
# 2. Windows version:
# 2. set FLASK_APP=sabu.py
# 2. Linux version:
# 2. export FLASK_APP=sabu.py

# 3. git clone --depth=1 https://github.com/twintproject/twint.git
# 4. cd twint
# 5. pip3 install . -r requirements.txt
# 6. pip3 install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint

# SET FLASK ENV VARIABLE FOR TEST RUN:
# $env:FLASK_APP = "sabu.py"
# FLASK DEBUG MODE (optional):
# $env:FLASK_ENV = "development"

# Needed for session data
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'

# Set default html render page (index.html)
@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def search_page():

    if request.method == 'POST':
        input_config = {"Search": str(request.form['search_input'])}

        if ('popular_tweets' in request.form):
            input_config["Popular_tweets"] = True
        if ('verified' in request.form):
            input_config["Verified"] = True
        if ('username' in request.form):
            input_config["Username"] = str(request.form['search_input'])
        if ('filter_retweets' in request.form):
            input_config["Filter_retweets"] = True
        if ('links' in request.form):
            input_config["Links"] = True
        if ('sabu_' in request.form):
            # future use case?
            print('sabu')

        if (request.form['from'] != ''):
            input_config['Since'] = str(request.form['from'])
        if (request.form['until'] != ''):
            input_config['Until'] = str(request.form['until'])


        # get tweet limit
        slice_limit = request.form['limit']

        data_dict = get_tweets(**input_config)
    else:
        # Anger, Disgust, Fear, Joy, Sadness
        data = {'joy': 'nan', 'anger': 'nan', 'sadness':'nan', 'disgust':'nan', 'fear':'nan'}
        add_msg = "Enter a search query above."
        toxClass = ['', '']
        senClass = ['', '', '']
        isAnalyzed = False

        return render_template('index.html', data=data, add_msg=add_msg, toxClass=toxClass, senClass=senClass, isAnalyzed=isAnalyzed)

    # Convert dataframe to dictionary since Jinja is not compatible with DF's
    if hasattr(data_dict, 'tweet') and hasattr(data_dict, 'id'):
        items = data_dict[:(int(slice_limit))].tweet.to_dict()
        tweet_ids = data_dict[:(int(slice_limit))].id.to_dict()
        add_msg = "Select a tweet below."
    else:
        items = {"Nan": "Nan"}
        tweet_ids = {"Nan": "Nan"}
        add_msg = "Error: No Tweets found that match the search query"

    data = {'joy': 'nan', 'anger': 'nan', 'sadness':'nan', 'disgust':'nan', 'fear':'nan'}
    # send session data to analyzed page so we can have persistent data
    session["tweets"] = items
    session["tweet_ids"] = tweet_ids

    # filler data before render
    toxClass = ['', '']
    senClass = ['', '', '']
    isAnalyzed = False

    return render_template('index.html', items=items, data=data, add_msg=add_msg, toxClass=toxClass, senClass=senClass, isAnalyzed=isAnalyzed, tweet_ids=tweet_ids)


@app.route('/analyzed', methods=['POST', 'GET'])
def analyzed():
    # find the tweet they want to analyze
    tweet_index = request.form['projectFilepath']
    add_msg = ""

    # gather tweet data
    items = session.get("tweets", None)
    tweet_ids = session.get("tweet_ids", None)

    toxicity = toxInference([items[f"{tweet_index}"]])
    sentiment = sentimentInference([items[f"{tweet_index}"]])
    emotion = emotionInference([items[f"{tweet_index}"]])

    # format sentiment scores
    senClass = [f"{(sentiment[0][2]*100):.2f}", f"{(sentiment[0][1]*100):.2f}", f"{(sentiment[0][0]*100):.2f}"]

    # Check if it is a Toxic tweet
    if (toxicity[0][0] == 'not toxic'):
        # If not assign descriptors
        toxClass = ['Not Toxic', 'No Toxicity Classifications']
    else:
        # Else check for classification types
        if (toxicity[0][0][9:] == ''):
            toxClass = ['Toxic', "No Toxicity Classifications"]
        else:
            toxClass = ['Toxic', f"[{toxicity[0][0][9:]}"]

    isAnalyzed = True

    # get chart data
    sum = (emotion[0][3]) + (emotion[0][0]) + (emotion[0][4]) + (emotion[0][1]) + (emotion[0][2])
    data = {'joy': (emotion[0][3])/sum, 'anger': (emotion[0][0])/sum, 'sadness': (emotion[0][4])/sum, 'disgust': (emotion[0][1])/sum, 'fear': (emotion[0][2])/sum}

    return render_template('index.html', data=data, items=items, add_msg=add_msg, toxClass=toxClass, senClass=senClass, isAnalyzed=isAnalyzed, tweet_index=tweet_index, tweet_ids=tweet_ids)






# TODO
#def about_page():
