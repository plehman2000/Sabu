from flask import Flask, render_template, request, flash
import twint
import nest_asyncio
import pandas
#nest_asyncio.apply()

app = Flask(__name__)
app.secret_key = "my_password888888"

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')
@app.route('/search', methods=['POST', 'GET'])
def query_results():
    c = twint.Config()
    c.Pandas = True
    if request.method == 'POST':
        c.Search = str(request.form['name_input'])
        if ('popular_tweets' in request.form):
            c.Popular_tweets = True
    else:
        return render_template('search.html')
    c.Store_object = True
    c.Limit = 20
    twint.run.Search(c)

    Tweets_df = twint.storage.panda.Tweets_df;
    # Convert dataframe to dictionary since Jinja is not compatible with DF's
    items = Tweets_df.tweet.to_dict()
    return render_template('search.html', items=items)

@app.route('/search')
def search_page():
    items = []
    return render_template('search.html', items=items)

# Below is an example of a dynamic page, will be useful when creating custom pages for gathered tweets.
# Creates page about '<username>' dynamic route creates /'John' when referenced
#@app.route('/about/<username>')
#def about_page(username):
#    return f'<h1>This is the about page of {username}</h1>'