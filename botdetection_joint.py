from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import botprocessing
import userprocessing

import embeddingprocessing
import embeddingprocessing_bot

from sklearn import metrics
from sklearn.utils import shuffle

import cPickle as pickle

import heatmap
import nltk


# NOTE: This is for LRP prediction tests...this function can be ignored :)

def lrp1(news_sentences, final_predictions_news):
  t = y_news_test[40:60]
  x = x_news_test[40:60]
  final_predictions_news =  final_predictions_news[40:60]
  news_sentences = news_sentences[40:60]

  output1 = shared_layer.eval({X:x, Y1:t})
  output2 = Y1_hidden1_layer.eval({X:x, Y1: t})
  output3 = Y1_hidden2_layer.eval({X:x, Y1: t})
  output4 = Y1_hidden3_layer.eval({X:x, Y1: t})
  output5 = Y1_hidden4_layer.eval({X:x, Y1: t})
  output6 = Y1_logits.eval({X:x, Y1: t})
  output6 = tf.cast(output6, tf.float32)

  outputresult = output5 * tf.reduce_sum(tf.expand_dims(Y1_output_weights,0) * tf.expand_dims(output6/(output6+1e-3),1),-1)
  hiddenlayerresult1 = output4 * tf.reduce_sum(tf.expand_dims(Y1_hidden4_weights,0) * tf.expand_dims(outputresult/(output5+1e-3),1),-1)
  hiddenlayerresult2 = output3 * tf.reduce_sum(tf.expand_dims(Y1_hidden3_weights,0) * tf.expand_dims(hiddenlayerresult1/(output4+1e-3),1),-1)
  hiddenlayerresult3 = output2 * tf.reduce_sum(tf.expand_dims(Y1_hidden2_weights,0) * tf.expand_dims(hiddenlayerresult2/(output3+1e-3),1),-1)
  hiddenlayerresult4 = output1 * tf.reduce_sum(tf.expand_dims(Y1_hidden1_weights,0) * tf.expand_dims(hiddenlayerresult3/(output2+1e-3),1),-1)
  
  lrp_result = X * tf.reduce_sum(tf.expand_dims(shared_layer_weights,0) * tf.expand_dims(hiddenlayerresult4/(output1+1e-3),1),-1)
  lrp = lrp_result.eval({X:x, Y1:t})

  lrpheatmapinput = []
  for i in range(20): #number of examples passed in as a batch
    temp_count = 0
    temp = []
    for embedding in range(2000): 
      if embedding != 0 and embedding != 1999:
        if embedding%100!=0: 
          temp_count += lrp[i][embedding]
        else:
          if temp_count != 0:
            temp.append(temp_count)
            temp_count = 0
          temp_count += lrp[i][embedding]
      elif embedding == 1999:
        temp_count += lrp[i][embedding]
        if temp_count != 0:
            temp.append(temp_count)
      else:
        temp_count += lrp[i][embedding]
    lrpheatmapinput.append(temp)

  print(len(x[0]))
  print(len(lrpheatmapinput[i]))
  print(len(lrpheatmapinput))


  '''token_tweets = []
  for i in news_sentences:
      if i != "":
          words = nltk.word_tokenize(i)
      token_tweets.append([word.lower() for word in words if word.isalpha()])'''
  

  '''fake_news_features, fake_sentences = embeddingprocessing.proc('/home/dwj/Documents/school/534/pheme-rnr-dataset/*/rumours/*/so*/*')
  real_news_features, real_sentences = embeddingprocessing.proc('/home/dwj/Documents/school/534/pheme-rnr-dataset/*/non-rumours/*/so*/*')
  sents_for_lrp = []

  fake_news_reshaped = []
  for i in range(len(fake_news_features)):
    temp = []
    for a in range(len(fake_news_features[i])):
     for z in range(len(fake_news_features[i][a])):
      temp.append(fake_news_features[i][a][z])
    fake_news_reshaped.append(temp)
    
  real_news_reshaped = []
  for i in range(len(real_news_features)):
    temp = []
    for a in range(len(real_news_features[i])):
     for z in range(len(real_news_features[i][a])):
      temp.append(real_news_features[i][a][z])
    real_news_reshaped.append(temp)
  
  print(len(x[0]))
  print(len(x))
  print(len(fake_news_reshaped[0]))
  print(x[0][:10])
  print(fake_news_reshaped[0][:10])
  print(type(x[0][0]))
  print(type(fake_news_reshaped[0][0]))
  x = np.asarray(x, dtype=np.float64)
  fake_news_reshaped = np.asarray(fake_news_reshaped, dtype=np.float64)
  real_news_reshaped = np.asarray(real_news_reshaped, dtype=np.float64)
  print(x[0])
  print(fake_news_reshaped[0])
  for num in range(len(fake_news_reshaped)):
    #print (x[1][0], fake_news_reshaped[num][0])
    #print (x[1][0], real_news_reshaped[num][0])
    if x[1][0] == fake_news_reshaped[num][0] or x[1][0] == real_news_reshaped[num][0]:
      print("yes")'''
      
  '''
  for h in x:
    for y in range(len(fake_news_reshaped)):
      #if h.all() == fake_news_reshaped[y].all():
      if False not in (np.equal(h, fake_news_reshaped[y])):
        print("yes")
        sents_for_lrp.append(fake_sentences[y][:20])
  for h in x:
    for y in range(len(real_news_reshaped)):
      #if h.all() == real_news_reshaped[y].all():
      if False not in (np.equal(h, real_news_reshaped[y])):
        print("yes")
        sents_for_lrp.append(real_sentences[y][:20])'''

  print(len(lrpheatmapinput))
  print(len(news_sentences))
  for h in range(len(lrpheatmapinput)):
    #print("Correct:", correct[i])
    print(heatmap.html_heatmap(news_sentences[h], lrpheatmapinput[h]))
    print("predicted label: ", final_predictions_news[h])
    print("correct label: ", t[h])


# NOTE: This is for LRP prediction tests...this function can be ignored :)
def lrp2(bot_sentences, final_predictions):
  t = y_bot_test[40:60]
  x = x_bot_test[40:60]
  final_predictions = final_predictions[40:60]
  bot_sentences = bot_sentences[40:60]

  output1 = shared_layer.eval({X:x, Y1:t})
  output2 = Y2_hidden1_layer.eval({X:x, Y1: t})
  output3 = Y2_hidden2_layer.eval({X:x, Y1: t})
  output4 = Y2_hidden3_layer.eval({X:x, Y1: t})
  output5 = Y2_hidden4_layer.eval({X:x, Y1: t})
  output6 = Y2_logits.eval({X:x, Y1: t})
  output6 = tf.cast(output6, tf.float32)

  outputresult = output5 * tf.reduce_sum(tf.expand_dims(Y2_output_weights,0) * tf.expand_dims(output6/(output6+1e-3),1),-1)
  hiddenlayerresult1 = output4 * tf.reduce_sum(tf.expand_dims(Y2_hidden4_weights,0) * tf.expand_dims(outputresult/(output5+1e-3),1),-1)
  hiddenlayerresult2 = output3 * tf.reduce_sum(tf.expand_dims(Y2_hidden3_weights,0) * tf.expand_dims(hiddenlayerresult1/(output4+1e-3),1),-1)
  hiddenlayerresult3 = output2 * tf.reduce_sum(tf.expand_dims(Y2_hidden2_weights,0) * tf.expand_dims(hiddenlayerresult2/(output3+1e-3),1),-1)
  hiddenlayerresult4 = output1 * tf.reduce_sum(tf.expand_dims(Y2_hidden1_weights,0) * tf.expand_dims(hiddenlayerresult3/(output2+1e-3),1),-1)
  
  lrp_result = X * tf.reduce_sum(tf.expand_dims(shared_layer_weights,0) * tf.expand_dims(hiddenlayerresult4/(output1+1e-3),1),-1)
  lrp = lrp_result.eval({X:x, Y1:t})

  lrpheatmapinput = []
  for i in range(20): #number of examples passed in as a batch
    temp_count = 0
    temp = []
    for embedding in range(2000): 
      if embedding != 0 and embedding != 1999:
        if embedding%100!=0: 
          temp_count += lrp[i][embedding]
        else:
          if temp_count != 0:
            temp.append(temp_count)
            temp_count = 0
          temp_count += lrp[i][embedding]
      elif embedding == 1999:
        temp_count += lrp[i][embedding]
        if temp_count != 0:
            temp.append(temp_count)
      else:
        temp_count += lrp[i][embedding]
    lrpheatmapinput.append(temp)



  for i in range(len(lrpheatmapinput)):
    #print("Correct:", correct[i])
    print(heatmap.html_heatmap(bot_sentences[i], lrpheatmapinput[i]))
    print("predicted label: ", final_predictions[i])
    print("correct label: ", t[i])



# The multi-line comment below is all processing code and can be ignored, it doesn't really have anything to do with the function of the network

'''
# Processing code
bot_features, bot_sents = embeddingprocessing_bot.proc("bot_tweets.csv")
bot_labels = []
for i in range(len(bot_features)):
  bot_labels.append(1) # Labels should be 1 for all bots
user_features, user_sents = embeddingprocessing_bot.proc("real_tweets.csv")
user_labels = []
for i in range(len(user_features)):
  user_labels.append(0) # Labels should be 0 for all humans

bot_features = np.array(bot_features)
user_features = np.array(user_features)


fake_news_features, fake_sents = embeddingprocessing.proc('/home/dwj/Documents/school/534/pheme-rnr-dataset/*/rumours/*/so*/*')
fake_news_labels = []
for i in range(len(fake_news_features)):
  fake_news_labels.append(1) # Labels should be 1 for all fake news
real_news_features, real_sents = embeddingprocessing.proc('/home/dwj/Documents/school/534/pheme-rnr-dataset/*/non-rumours/*/so*/*')
real_news_labels = []
for i in range(len(real_news_features)):
  real_news_labels.append(0) # Labels should be 0 for all real news

fake_news_features = np.array(fake_news_features)
real_news_features = np.array(real_news_features)


bot_labels = bot_labels + user_labels #Uncomment for training bot task
bot_sentences = bot_sents + user_sents
bot_features = np.concatenate((bot_features, user_features)) #Uncomment for training bot task
print(len(bot_sentences))
print(len(bot_sentences[0]))
print(bot_sentences[0])
print(bot_features.shape)

news_labels = fake_news_labels + real_news_labels
news_sentences = fake_sents + real_sents
news_features = np.concatenate((fake_news_features, real_news_features))

print("loading finished")
print("bot_features shape: ", len(bot_features[0]))
# Make sure the data is shuffled so batches won't be fed into the network that contain only one class
x_bot, y_bot, bot_sentences = shuffle(bot_features, bot_labels, bot_sentences)
print("x_bot length: ", len(x_bot))
print("x_bot length: ", len(x_bot[0]))

x_bot_reshaped = []
for i in range(len(x_bot)):
  temp = []
  for x in range(len(x_bot[i])):
    for z in range(len(x_bot[i][x])):
      temp.append(x_bot[i][x][z])
  x_bot_reshaped.append(temp)

x_bot_reshaped = x_bot_reshaped[:12000]
y_bot = y_bot[:12000]

print("news_features shape: ", len(news_features))
print(len(news_sentences))
print(len(news_sentences[0]))
print(len(news_features))
x_news, y_news, news_sentences = shuffle(news_features, news_labels, news_sentences)
print("x_news length: ", len(x_news))
print("x_news length: ", len(x_news[0]))
x_news_reshaped = []
for i in range(len(x_news)):
  temp = []
  for x in range(len(x_news[i])):
    for z in range(len(x_news[i][x])):
      temp.append(x_news[i][x][z])
  x_news_reshaped.append(temp)

bot_sentences = bot_sentences [10000:]
news_sentences = news_sentences[4802:]

print("shuffled")

 #Uncomment for bot detection task
x_bot_train = x_bot_reshaped[:10000] #Use the first 2000 to train on
y_bot_train = y_bot[:10000]
x_bot_test = x_bot_reshaped[10000:]
y_bot_test = y_bot[10000:]

print(len(x_bot_train))
print(len(x_bot_train[0]))

x_news_train = x_news_reshaped[:4802] #Use the first 2000 to train on
y_news_train = y_news[:4802]
x_news_test = x_news_reshaped[4802:]
y_news_test = y_news[4802:]

print(len(x_news_train[0]))

with open('x_bot_train2.pkl', 'wb') as f:
    pickle.dump(x_bot_train, f)
with open('y_bot_train2.pkl', 'wb') as f:
    pickle.dump(y_bot_train, f)
with open('x_bot_test2.pkl', 'wb') as f:
    pickle.dump(x_bot_test, f)
with open('y_bot_test2.pkl', 'wb') as f:
    pickle.dump(y_bot_test, f)
with open('bot_sentences.pkl', 'wb') as f:
    pickle.dump(bot_sentences, f)


with open('x_news_train2.pkl', 'wb') as f:
    pickle.dump(x_news_train, f)
with open('y_news_train2.pkl', 'wb') as f:
    pickle.dump(y_news_train, f)
with open('x_news_test2.pkl', 'wb') as f:
    pickle.dump(x_news_test, f)
with open('y_news_test2.pkl', 'wb') as f:
    pickle.dump(y_news_test, f)
with open('news_sentences.pkl', 'wb') as f:
    pickle.dump(news_sentences, f)

'''
pkl = open('x_bot_train2.pkl', 'rb')
x_bot_train = pickle.load(pkl)
pkl.close()
pkl = open('y_bot_train2.pkl', 'rb')
y_bot_train = pickle.load(pkl) 
pkl.close()
pkl = open('x_bot_test2.pkl', 'rb')
x_bot_test = pickle.load(pkl)
pkl.close()
pkl = open('y_bot_test2.pkl', 'rb')
y_bot_test = pickle.load(pkl)
pkl.close()

pkl = open('x_news_train2.pkl', 'rb')
x_news_train = pickle.load(pkl)
pkl.close()
pkl = open('y_news_train2.pkl', 'rb')
y_news_train = pickle.load(pkl)
pkl.close()
pkl = open('x_news_test2.pkl', 'rb')
x_news_test = pickle.load(pkl)
pkl.close()
pkl = open('y_news_test2.pkl', 'rb')
y_news_test = pickle.load(pkl)
pkl.close()
pkl = open('news_sentences.pkl', 'rb')
news_sentences = pickle.load(pkl)
pkl.close()
pkl = open('bot_sentences.pkl', 'rb')
bot_sentences = pickle.load(pkl)
pkl.close()


# Create tensorflow placeholder variables. These variables do not get an actual value until we run the tf.Session() and feed in values for each variable
X = tf.placeholder("float", name="X")
Y1 = tf.placeholder("float", name="Y1")
Y2 = tf.placeholder("float", name="Y2")


initializer = tf.contrib.layers.xavier_initializer(seed=5) #This is used to do a Xavier initialization of the network weights instead of initializing weights from a random distribution

shared_layer_bias = tf.Variable(initializer([1000])) #Create a tensorflow variable and give it a dimension of 1000 and initialize it with a value from the Xavier initialization when we run our tf session
Y1_hidden1_bias = tf.Variable(initializer([500]))
Y1_hidden2_bias = tf.Variable(initializer([200]))
Y1_hidden3_bias = tf.Variable(initializer([100]))
Y1_hidden4_bias = tf.Variable(initializer([50]))
Y1_logits_bias = tf.Variable(initializer([1]))

shared_layer_weights = tf.Variable(initializer([2000,1000]), name="shared_layer_weights", dtype="float") # This is where we set the dimensions of the first layer at 2000 input dimensions (the number of input features we have) and 1000 output dimensions
Y1_hidden1_weights = tf.Variable(initializer([1000,500]), name="Y1_hidden1_layer_weights", dtype="float") # Here we have 1000 input dimensions to this middle layer (this needs to match the output dimensions of the previous layer) and we have 500 output dimensions
Y1_hidden2_weights = tf.Variable(initializer([500,200]), name="Y1_hidden2_layer_weights", dtype="float") 
Y1_hidden3_weights = tf.Variable(initializer([200,100]), name="Y1_hidden3_layer_weights", dtype="float")
Y1_hidden4_weights = tf.Variable(initializer([100,50]), name="Y1_hidden4_layer_weights", dtype="float")
Y1_output_weights = tf.Variable(initializer([50,1]), name="Y1_output_layer_weights", dtype="float") # The last layer has an output dimension of 1 because we are doing a binary prediction

Y2_hidden1_bias = tf.Variable(initializer([500]))
Y2_hidden2_bias = tf.Variable(initializer([200]))
Y2_hidden3_bias = tf.Variable(initializer([100]))
Y2_hidden4_bias = tf.Variable(initializer([50]))
Y2_logits_bias = tf.Variable(initializer([1]))

Y2_hidden1_weights = tf.Variable(initializer([1000,500]), name="Y2_hidden1_layer_weights", dtype="float") # This is the same as above, but we are making the task 2 neck of the network
Y2_hidden2_weights = tf.Variable(initializer([500,200]), name="Y2_hidden2_layer_weights", dtype="float") 
Y2_hidden3_weights = tf.Variable(initializer([200,100]), name="Y2_hidden3_layer_weights", dtype="float")
Y2_hidden4_weights = tf.Variable(initializer([100,50]), name="Y2_hidden4_layer_weights", dtype="float")
Y2_output_weights = tf.Variable(initializer([50,1]), name="Y2_output_layer_weights", dtype="float")

##### Shared layer
shared_layer = tf.add(tf.matmul(X, shared_layer_weights), shared_layer_bias) # do matrix multiplication in tf.matmul and feed the result to tf.add to add a bias value
shared_layer = tf.nn.relu(shared_layer) #relu is rectified linear activation function (see https://en.wikipedia.org/wiki/Rectifier_(neural_networks) )
#####

##### Task 1 network architecture. This code is the architecture for exclusively the task 1 part of the network
# This code (as well as the similar task 2 code) handles the linear algebra that we use in neural network calculations
Y1_hidden1_layer = tf.add(tf.matmul(shared_layer, Y1_hidden1_weights), Y1_hidden1_bias) #tf.matmul is matrix multiplication between previous layer and weights of current layer, then feed result to tf.add which adds to bias values
Y1_hidden1_layer = tf.nn.relu(Y1_hidden1_layer) 

Y1_hidden2_layer = tf.add(tf.matmul(Y1_hidden1_layer, Y1_hidden2_weights), Y1_hidden2_bias)
Y1_hidden2_layer = tf.nn.relu(Y1_hidden2_layer)

Y1_hidden3_layer = tf.add(tf.matmul(Y1_hidden2_layer, Y1_hidden3_weights), Y1_hidden3_bias)
Y1_hidden3_layer = tf.nn.relu(Y1_hidden3_layer)

Y1_hidden4_layer = tf.add(tf.matmul(Y1_hidden3_layer, Y1_hidden4_weights), Y1_hidden4_bias)
Y1_hidden4_layer = tf.nn.relu(Y1_hidden4_layer)

Y1_logits = tf.add(tf.matmul(Y1_hidden4_layer, Y1_output_weights), Y1_logits_bias)
Y1_output = tf.nn.sigmoid(Y1_logits) # Use a sigmoid activation function because our task is binary prediction and sigmoid predicts 0 or 1

Y1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y1_logits, labels=Y1)) # Calculate the loss using cross entropy. Note that the input to this is the "logits" ie. the unscaled output

Y1_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(Y1_loss) # There are a few different types of optimizers, like gradient descent etc., in this case use Adam. the learning_rate is a hyperparameter we have to tune, smaller values make it train slower, higher values make it train faster but can miss the optimum. Note that the learning rate is a hyperparameter we tune empirically
#####

##### Task 2 network architecture. This code is the architecture for exclusively the task 2 part of the network
Y2_hidden1_layer = tf.add(tf.matmul(shared_layer, Y2_hidden1_weights), Y2_hidden1_bias)
Y2_hidden1_layer = tf.nn.relu(Y2_hidden1_layer)

Y2_hidden2_layer = tf.add(tf.matmul(Y2_hidden1_layer, Y2_hidden2_weights), Y2_hidden2_bias)
Y2_hidden2_layer = tf.nn.relu(Y2_hidden2_layer)

Y2_hidden3_layer = tf.add(tf.matmul(Y2_hidden2_layer, Y2_hidden3_weights), Y2_hidden3_bias)
Y2_hidden3_layer = tf.nn.relu(Y2_hidden3_layer)

Y2_hidden4_layer = tf.add(tf.matmul(Y2_hidden3_layer, Y2_hidden4_weights), Y2_hidden4_bias)
Y2_hidden4_layer = tf.nn.relu(Y2_hidden4_layer)

Y2_logits = tf.add(tf.matmul(Y2_hidden4_layer, Y2_output_weights), Y2_logits_bias)
Y2_output = tf.nn.sigmoid(Y2_logits) # Use a sigmoid activation function because our task is binary prediction and sigmoid predicts 0 or 1

Y2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y2_logits, labels=Y2)) # Calculate the loss using cross entropy. Note that the input to this is the "logits" ie. the unscaled output

Y2_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(Y2_loss) # There are a few different types of optimizers, like gradient descent etc., in this case use Adam. the learning_rate is a hyperparameter we have to tune, smaller values make it train slower, higher values make it train faster but can miss the optimum
#####


batch_size = 100 # This is the number of examples that will be fed in at one time to the network
#config = tf.ConfigProto() #Uncomment this for GPU training
config = tf.ConfigProto( #Uncomment this for CPU training
        device_count = {'GPU': 0}
    )

#config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
  session.run(tf.initialize_all_variables()) # Start the actual tensorflow session, this creates the tensorflow computation graph
  for epochs in range(160): # epochs are the number of times we progress through the entire dataset. This is usually in the hundreds or thousands of times
    epoch_loss = 0
    i = 0
    if np.random.rand() > 10 :
        while i < 4802: #Need to change when changing datasets, this is training examples
            start = i
            end = i + batch_size
            batch_x = np.array(x_news_train[start:end])
            batch_y = np.array(y_news_train[start:end])
            _, Y1_Loss = session.run([Y1_op, Y1_loss], {X: batch_x, Y1: batch_y})
            epoch_loss += Y1_Loss
            i += batch_size
    else:
        while i < 10000: #Need to change when changing datasets, this is training examples
            start = i
            end = i + batch_size
            batch_x = np.array(x_bot_train[start:end])
            batch_y = np.array(y_bot_train[start:end])
            _, Y2_Loss = session.run([Y2_op, Y2_loss], {X: batch_x, Y2: batch_y})
            epoch_loss += Y2_Loss
            i += batch_size
    print('Epoch:',epochs+1, 'completed. Loss: ', epoch_loss)

  predictions = Y1_output.eval({X: x_news_test, Y1: y_news_test}) # This tells tensorflow to calculate the computation graph up to the point where we evaluate the value of the output variable given that we feed x_test for X and y_test for Y
  #test_y = Y1_output.eval({X: x_news_test, Y1: y_news_test}) # This was for debug purposes, can ignore this line
  #print(test_y[:5]) #this was for debug purposes, can ignore this line
  final_predictions_news = []
  ones = []
  zeroes = []
  for i in predictions:
    if i[0] > 0.34: # This is essentially a threshold for the sigmoid output. The output for sigmoid is between 0 and 1, we need to decide what the threshold is between where we want to predict 0 and where we want to predict 1
      final_predictions_news.append(1)
      ones.append(1)
    else:
      final_predictions_news.append(0)
      zeroes.append(0)
  print('News F-score: ', metrics.f1_score(y_news_test, final_predictions_news))
  print('News Precision: ', metrics.precision_score(y_news_test, final_predictions_news))
  print('News Recall: ', metrics.recall_score(y_news_test, final_predictions_news))
  print("News ones", len(ones))
  print("News zeroes", len(zeroes))

  predictions = Y2_output.eval({X: x_bot_test, Y2: y_bot_test}) # This tells tensorflow to calculate the computation graph up to the point where we evaluate the value of the output variable given that we feed x_test for X and y_test for Y
  #test_y = Y2_output.eval({X: x_bot_test, Y2: y_bot_test}) # This was for debug purposes, can ignore this line
  #print(test_y[:5]) #this was for debug purposes, can ignore this line
  final_predictions = []
  ones = []
  zeroes = []
  for i in predictions:
    if i[0] > 0.5: # This is essentially a threshold for the sigmoid output. The output for sigmoid is between 0 and 1, we need to decide what the threshold is between where we want to predict 0 and where we want to predict 1
      final_predictions.append(1)
      ones.append(1)
    else:
      final_predictions.append(0)
      zeroes.append(0)
  print('Bot F-score: ', metrics.f1_score(y_bot_test, final_predictions))
  print('Bot Precision: ', metrics.precision_score(y_bot_test, final_predictions))
  print('Bot Recall: ', metrics.recall_score(y_bot_test, final_predictions))
  print("Bot ones", len(ones))
  print("Bot zeroes", len(zeroes))

  #lrp1(news_sentences, final_predictions_news) #These function calls are for obtaining LRP values, can ignore
  #lrp2(bot_sentences, final_predictions)
