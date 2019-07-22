# Sentiment Analysis
This project get the desired input data and tries to model the sentiment. \
Generally, the procedure is listed as below: 

1. Data preprocessing
   - removing punctuations
   - tokenizing
   - cleaning stop-words
   - stemming words
2. Convert text to numeric (word embedding)
    - applying word2vec 
3. Apply ML model to decide the output
    - Classifer like: RandomForest model
  
Hint: since we study tweets consisting emojis and special characters,
it is better to eliminate punctuation cleaning phase. :)

### Data Source
#### Twitter US Airline Sentiment

The original data exists in [kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). \
It contains sentiments such as positive, neutral and negative about 6 US airlines; \
Besides, the data was crawled in February 2015. \
It should be noted that the current version is the reduced one to control the size of ai-platform. \
We are interested in the below columns of data:
- airline_sentiment: True labels
- text: The original tweet
   
### Running Instruction 
```commandline
cd /tasks/natural-language-processing/sentiment-analysis
mlflow run . -P data_path=Tweets.csv
``` 