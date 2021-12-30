from DataProcessing.TwintUtils import *
from DataProcessing.wikiUtils import *
from DataModelling.getSentiment.getSentiment import *
from DataModelling.getTox.getToxicity import *
from IPython.display import clear_output
clear_output(wait=True)



search_query = 'from:@FoxNews'
#need to add try-except to get_tweets

tweets = get_tweets(search_query, 5)
        

processed_tweets = [process_tweet(tweet)['tweet'] for tweet in tweets]
print(processed_tweets)
