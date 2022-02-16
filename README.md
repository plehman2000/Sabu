# Sabu -> Sentiment Analysis By User for Twitter
<p align="center">
<img src="https://github.com/plehman2000/Sabu/blob/main/gh_resources/1.png?raw=true" width="900" height="auto">
  </p>
  <br/>
    Sabu enables you to search Twitter by topic or username and run several NLP models on tweets of interest. This project uses 2 fine-tuned BERT PyTorch-based Models and a premade Tensorflow-based model for emotion detection, sentiment analysis and toxicity classification, respectively. TWINT is used to gather and filter tweets while surpassing Twitter API rate limits and Flask was used to build the webapp. 
<br/>
<br/>
    To use, simply navigate to http://45.33.26.141/ and input your search, checking the boxes associated with your desired filters. When you find a tweet to analyze, simply click 'Analyze Tweet.'

<br/>
<br/>
<br/>

 <img src="https://github.com/plehman2000/Sabu/blob/main/gh_resources/3.png?raw=true" width="auto" height="350"><img src="https://github.com/plehman2000/Sabu/blob/main/gh_resources/2.png?raw=true" width="auto" height="350">
 <br/>
    Sample tweet query results for Username: 'elonmusk'
 <img src="https://github.com/plehman2000/Sabu/blob/main/gh_resources/tweet_output.png?raw=true" width="auto" height="350">



## Architecture Diagrams for Sentiment and Emotion Models
<img src="https://github.com/plehman2000/Sabu/blob/main/gh_resources/emotionNet.png?raw=true" width="auto" height="300"><img src="https://github.com/plehman2000/Sabu/blob/main/gh_resources/sentimentNet.png?raw=true" width="auto" height="300">

## Example Notebook of Model Usage
![alt text](https://github.com/plehman2000/Sabu/blob/main/gh_resources/testNB.png?raw=true)


Produced by the Thought Modeling and Analysis Team for the GAITOR Club at UF
