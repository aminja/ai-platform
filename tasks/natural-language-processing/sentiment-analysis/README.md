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
### Running Instruction 
```commandline
cd /tasks/natural-language-processing/sentiment-analysis
mlflow run . -P data_path=Tweets.csv
``` 