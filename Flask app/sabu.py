from flask import Flask, render_template, request, session
from twint_wrapper import get_tweets

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
# $env:FLASK_APP = "market.py"
# FLASK DEBUG MODE (optional):
# $env:FLASK_ENV = "development"

app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'

# Set default html render page (index.html)
@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def search_page():


    if request.method == 'POST':
        #popular_tweets = False
        # set default parameters (none)
        popular_tweets = False
        verified = False
        username = False
        filer_retweets = False
        links = False
        sabu_ = False


        limit = 5  # Default limit of 20
        if ('popular_tweets' in request.form):
            popular_tweets = True
            print('pop')
        if ('verified' in request.form):
            verified = True
            print('veri')
        if ('username' in request.form):
            username = True
            print('user')
        if ('filter_retweets' in request.form):
            filter_retweets = True
            print('filter r')
        if ('links' in request.form):
            links = True
            print('linky')
        if ('sabu_' in request.form):
            sabu_ = True
            print('sabu')

        # gather from until vals


        # gather tweet limit val
        print('Limit entered: ' + str(request.form['limit']))


        print(str(request.form['search_input']))
        data_dict = get_tweets(str(request.form['search_input']), 20, True)
    else:
        # Anger, Disgust, Fear, Joy, Sadness
        data = {'happy': 'nan', 'angry': 'nan', 'sad':'nan', 'confused':'nan', 'funny':'nan'}
        add_msg = "Enter a search query above."
        return render_template('index.html', data=data, add_msg=add_msg)
    # Convert dataframe to dictionary since Jinja is not compatible with DF's
    items = data_dict.tweet.to_dict()
    #print(items)
    # tweets = [tweet['tweet'] for tweet in data_dict]
    # list(items.values())
    # Test chart.js
    add_msg = "Select a tweet below."
    """
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
    ]"""

    """
    sentiment_scores = getSentimentAverage(list(items.values()))
    search_query = str(request.form['name_input'])

    negativity = "{:.2f}".format(100 * sentiment_scores[0])
    neutrality = "{:.2f}".format(100 * sentiment_scores[1])
    positivity = "{:.2f}".format(100 * sentiment_scores[2])

    print("Tweets found using " + search_query + " were " + negativity + "% Negative, " + neutrality + "% Neutral, " + positivity + "% Positive")
    """

    """
    # get Toxcity test
    sentences = np.asarray(list(items.values()))


    sentence_tox_pairs = getToxicity(sentences)

    toxic_num = 0

    for pair in sentence_tox_pairs:
        if (pair > 0.5):
            toxic_num += 1
    print('TOXIC NUM: ' + str(toxic_num))
    print('TOXIC COUNT: ' + str(len(sentence_tox_pairs)))
    toxic_rating = "{: .2f}".format(float(toxic_num / len(sentence_tox_pairs)))
    nontoxic = "{: .2f}".format(float(1 - (toxic_num / len(sentence_tox_pairs))))
    """

    # actual_tweets = items.values()
    # print(actual_tweets)
    # items_np = np.fromiter(items.values(),dtype='S128')
    # print(items_np)

    data = {'happy': 'nan', 'angry': 'nan', 'sad':'nan', 'confused':'nan', 'funny':'nan'}
    session["tweets"] = items
    #session["second"] = "yes"


    return render_template('index.html', items=items, data=data, add_msg=add_msg)


@app.route('/analyzed', methods=['POST', 'GET'])
def analyzed():


    projectpath = request.form['projectFilepath']
    add_msg = ""
    print(projectpath)
    # gather tweet data
    items = session.get("tweets", None)
    print(session.get("second", None))
    print("Hello World!")
    print(items)
    data = {'happy': 0.8, 'angry': 0.8, 'sad':0.8, 'confused':0.8, 'funny':0.8}
    return render_template('index.html', data=data, items=items, add_msg=add_msg)






# TODO
#def about_page():
