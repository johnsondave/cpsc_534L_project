import csv
import gensim
import nltk
import io
from sklearn.preprocessing import Normalizer
import glob
import json

def proc(filelocations): 

  filenames = glob.glob(filelocations)
  test = []
  for i in filenames:
      with open(i) as f:
          data = json.load(f)
          test.append(data["text"])
  print(test[0])

  
  token_tweets = []
  for i in test:
      if i != "":
          words = nltk.word_tokenize(i)
      token_tweets.append([word.lower() for word in words if word.isalpha()])
  print(token_tweets[:3])

  model = gensim.models.Word2Vec(token_tweets, size=100, min_count=1)
  model.init_sims(replace=True)

  features = []
  token_sentences = []
  for i in token_tweets:
      temp = []
      for word in i:
          temp.append(model.wv[word])
      features.append(temp)
      if len(i) >20:
          token_sentences.append(i[:20])
      else:
          token_sentences.append(i)
  transformed_features = []
  for i in features:
      temp = []
      if len(i) > 20:
          transformed_features.append(i[:20])
      else:
          #temp.append(i)
          
          while len(i) < 20:
              tempzeroes = [0 for x in range(100)]
              i.append(tempzeroes)
              
              #temp.append(tempzeroes)
              #print(len(temp))
              #print("#############")
              #temp.append([0 for i in range(100)])
          transformed_features.append(i)

  return transformed_features, token_sentences

