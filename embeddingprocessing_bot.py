import csv
import gensim
import nltk
import io
from sklearn.preprocessing import Normalizer


def proc(filename):
  headings = []
  text = []
  print (filename)

  with open(filename, 'r') as f:
  #with codecs.open(filename, 'rb', 'utf-8') as f:
    s = csv.reader((x.replace('\0', '') for x in f), delimiter=',')
    for row in s:
      #for i in row:
          #i.replace('\0', '')

      #print(row)
      text.append(row)
  headings = text[0]
  text = text[1:]
  tweets = []
  for i in range(15000): #just grab 15,000 examples for now 
    tweets.append(text[i][1])
  token_tweets = []
  for i in tweets:
      if i != "":
          words = nltk.word_tokenize(i.decode('utf-8'))
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


