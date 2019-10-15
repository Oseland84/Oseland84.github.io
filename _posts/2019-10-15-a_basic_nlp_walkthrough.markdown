---
layout: post
title:      "A Basic NLP Walkthrough"
date:       2019-10-15 16:17:32 +0000
permalink:  a_basic_nlp_walkthrough
---


To be clear, this was my first project with NLP, but also with Deep Learning, but as I learned about these concepts, they seemed to go hand in hand. The first thing I needed to do was find a good dataset to work with. I ended up going with a News Category Dataset ("https://rishabhmisra.github.io/publications/") I found on Kaggle, which is becoming a bit of a go-to for me. I love that I can plug in what type of work I want to do in their search bar, and be given recomendations on what datasets to use from there, but moving on. The dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The idea being to train a model that could be used to identify tags for untracked news articles or to identify the type of language used in different news articles. To put it simply, I wanted to build a model that could take unlabeled news articles, and identify which catagory of article it belonged to. Each news headline has a corresponding category. Categories and corresponding article counts are as follows:

1. -POLITICS: 32739,
1. -WELLNESS: 17827,
1. -ENTERTAINMENT: 16058,
1. -TRAVEL: 9887,
1. -STYLE & BEAUTY: 9649,
1. -PARENTING: 8677,
1. -HEALTHY LIVING: 6694,
1. -QUEER VOICES: 6314,
1. -FOOD & DRINK: 6226,
1. -BUSINESS: 5937,
1. -COMEDY: 5175,
1. -SPORTS: 4884,
1. -BLACK VOICES: 4528,
1. -HOME & LIVING: 4195,
1. -PARENTS: 3955,
1. -THE WORLDPOST: 3664,
1. -WEDDINGS: 3651,
1. -WOMEN: 3490,
1. -IMPACT: 3459,
1. -DIVORCE: 3426,
1. -CRIME: 3405,
1. -MEDIA: 2815,
1. -WEIRD NEWS: 2670,
1. -GREEN: 2622,
1. -WORLDPOST: 2579,
1. -RELIGION: 2556,
1. -STYLE: 2254,
1. -SCIENCE: 2178,
1. -WORLD NEWS: 2177,
1. -TASTE: 2096,
1. -TECH: 2082,
1. -MONEY: 1707,
1. -ARTS: 1509,
1. -FIFTY: 1401,
1. -GOOD NEWS: 1398,
1. -ARTS & CULTURE: 1339,
1. -ENVIRONMENT: 1323,
1. -COLLEGE: 1144,
1. -LATINO VOICES: 1129,
1. -CULTURE & ARTS: 1030,
1. -EDUCATION: 1004






After importing the data set, everything looks to be in pretty good order, the only real change I want to make before I begin preparing for modeling is to combine 'THE WORLDPOST' with 'WORLDPOST' since they are basically the same thing.
```
df.category = df.category.map(lambda x: 'WORLDPOST' if x== 'THE WORLDPOST' else x)
```

### Preparing the Data¶
Since we're working with text data, we'll still need to do some basic preprocessing and tokenize our data. You'll notice from the sample of the data above that two different columns contain text data--headline and short_description. The more text data our Word2Vec model has, the better it will perform. Therefore, we'll want to combine the two columns before tokenizing each comment and training our Word2Vec model.
```
# takes a while, a lot of data...
target = df.category
df['combined_text'] = df.headline + ' ' +  df.short_description
data = df['combined_text'].map(word_tokenize).values
```
#### Loading A Pretrained GloVe Model
I will be loading the pretrained weights from GloVe (short for Global Vectors for Word Representation) from the Stanford NLP Group. These are commonly accepted as some of the best pre-trained word vectors available, and they're open source. Because of machine limitations, I will only be using the smallest(still containing 100-dimensional word vectors for 6 billion words!).

#### Getting the Total Vocabulary
Although our pretrained GloVe data contains vectors for 6 billion words and phrases, we don't need all of them. Instead, we only need the vectors for the words that appear in our dataset. If a word or phrase doesn't appear in our dataset, then there's no reason to waste memory storing the vector for that word or phrase.

This means that we need to start by computing the total vocabulary of our dataset. We can do this by adding every word in the dataset into a python set object. This is easy, since we've already tokenized each comment stored within data.
 `total_vocabulary = set(word for headline in data for word in headline)`
 
 `glove = {}
with open('glove.6B.50d.txt', 'rb') as f:
    for line in f:
        parts = line.split()
        word = parts[0].decode('utf-8')
        if word in total_vocabulary:
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector`
						
						
						
The next step is to combine all the vectors for a given headline into a Mean Embedding by finding the average of all the vectors in that headline.

```
class W2vVectorizer(object):
    
    def __init__(self, w2v):
        # takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])
    
    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # It can't be used in a sklearn Pipeline. 
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])
```
Creat a pipeline for my machine learning models

```
rf =  Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
              ("Random Forest", RandomForestClassifier(n_estimators=100, verbose=True))])

svc = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ('Support Vector Machine', SVC())])

lr = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
              ('Logistic Regression', LogisticRegression())])
```

```
models = [('Random Forest', rf),
          ("Support Vector Machine", svc),
          ("Logistic Regression", lr)]
```
```
scores = [(name, cross_val_score(model, data, target, cv=2).mean()) for name, model, in models]
```
```
scores
```
#### Random Forest', 0.3040862473652662
#### Support Vector Machine', 0.31789813859395244
#### Logistic Regression', 0.3186251591420255

These scores may seem pretty low, but remember that there are 40 (after some feature engineering) possible categories that headlines could be classified into. This means the naive accuracy rate (random guessing) would achieve an accuracy of just over 0.025!(2.5%) Our models have plenty of room for improvement, but they do work!

I think it is also worth mentioning that there was not much difference in scores when I ran 20% of data set through the model VS the entire dataset, except that it took much longer (almost 3 hours). Now it's time to get into a little more advanced modeling.

#### Deep Learning (With Word Embeddings)
after importing all the libraries needed I then convert our labels to a one-hot encoded format.
```
y = pd.get_dummies(target).values
```
Now, we'll preprocess our text data. To do this, we start from the step where we combined the headlines and short description. We'll then use Keras's preprocessing tools to tokenize each example, convert them to sequences, and then pad the sequences so they're all the same length.

Note how during the tokenization step, we set a parameter to tell the tokenizer to limit our overall vocabulary size to the 20000 most important words.

```
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(df.combined_text))
list_tokenized_headlines = tokenizer.texts_to_sequences(df.combined_text)
X_t = sequence.pad_sequences(list_tokenized_headlines, maxlen=100)
```
Now, we'll construct our neural network. Notice how the Embedding Layer comes second, after the input layer. In the Embedding Layer, we specify the size we want our word vectors to be, as well as the size of the embedding space itself. The embedding size we specified is 128, and the size of the embedding space is best as the size of the total vocabulary that we're using. Since we limited the vocab to 20000, that's the size we choose for the embedding layer.

Once our data has passed through an embedding layer, we feed this data into an LSTM layer, followed by a Dense layer, followed by output layer. We also add some Dropout layers after each of these layers, to help fight overfitting.

Our output layer is a Dense layer with 40 neurons, which corresponds to the 40 possible classes in our labels. We set the activation function for this output layer to 'softmax', so that our network will output a vector of predictions, where each element's value corresponds to the percentage chance that the example is the class that corresponds to that element, and where the sum of all elements in the output vector is 1.

```
embedding_size = 128
input_ = Input(shape=(100,))
x = Embedding(20000, embedding_size)(input_)
x = LSTM(25, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.5)(x)
# There are 40 different possible classes, so we use 41 neurons in our output layer
x = Dense(40, activation='softmax')(x)

model = Model(inputs=input_, outputs=x)
```

Once we have designed our model, we still have to compile it, and provide important parameters such as the loss function to use ('categorical_crossentropy', since this is a mutliclass classification problem), and the optimizer to use.

```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Finally, we can fit the model by passing in the data, our labels, and setting some other hyperparameters such as the batch size, the number of epochs to train for, and what percentage of the training data to use for validation data.

```
model.fit(X_t, y, epochs=10, batch_size=32, validation_split=0.1)
```
I ran 10 epochs as you can seebut I will just put the first and last to save space, that should still be enough to show give you an idea of the modeling strength:

Epoch 1/10
36153/36153 [==============================] - 107s 3ms/step - loss: 2.1910 - accuracy: 0.4193 - val_loss: 
2.1853 - val_accuracy: 0.4403

...

Epoch 10/10
36153/36153 [==============================] - 105s 3ms/step - loss: 1.2506 - accuracy: 0.6219 - val_loss: 2.9638 - val_accuracy: 0.4584


I''ll sve the conclusions for the end, but you can probably already see how much stronger this model was performing over the traditional machine learning models. It is also important to understand that the models up to this point have been taking all the text data, and essentially "jumbling" it into a "Bag of Words" it does not take in a sentance as a whole (which is a absolute must for us) but is still able to do this good of a job at classification. I find that amazing, but that 's not really what we are focused on, but it is a good "seg-way" to the improved models I used next.

#### Sequence Models, better known as Recurrent Neural Networks

The hallmark of Recurrent Neural Networks is that they are used to evaluate Sequences of data, rather than just individual data points. All text data is sequence data by default--a sentance only makes sense when it's words are in the proper order. Recurrent Neural Networks (RNN) can take in text as full sequences of words, from a single sentence up to an entire document or book! Because of this, they do not suffer the same loss of information that comes from a traditional Bag-of-Words vectorization approach.

A lot of the code becomes mostly repetive from this point, so I  will skip some of it from this point on. without getting too tech heavy, I used two advanced types of neurons that typically outperform basic RNNs, Long Short Term Memory Cells (LSTMs) and Gated Recurrent Units (GRUs). The first model I trained was an LSTM, let's take a look at how it performed:
#### Training Our LSTM Model

```
lstm_model.fit(X_t, y, epochs=5, batch_size=32, validation_split=0.1)
```
* Note, these models are a bit more complex, so apart from using only 20 of the dataset to work on, I also only ran 5 epochs for this, and the next model.

Epoch 1/5
36153/36153 [==============================] - 124s 3ms/step - loss: 3.0242 - accuracy: 0.2319 - val_loss: 2.5963 - val_accuracy: 0.3218

...

Epoch 5/5
36153/36153 [==============================] - 122s 3ms/step - loss: 1.6373 - accuracy: 0.5461 - val_loss: 2.2266 - val_accuracy: 0.4798

 
 
again, I will save my conclusions for the end, but notice how quickly this model is getting it's accuracy score up, as well as much less disparity between accuracy, and validation accuracy(test accuracy).

#### Training Our GRU Model
Now that we have a benchmark for how an LSTM model performs, let's build the exact same model, but with GRU() cells instead of LSTM() cells!

```
gru_model.fit(X_t, y, epochs=5, batch_size=32, validation_split=0.1)
```
Epoch 1/5
36153/36153 [==============================] - 123s 3ms/step - loss: 3.0236 - accuracy: 0.2337 - val_loss: 2.5191 - val_accuracy: 0.3606

...

Epoch 5/5
36153/36153 [==============================] - 122s 3ms/step - loss: 1.6458 - accuracy: 0.5453 - val_loss: 2.1333 - val_accuracy: 0.4843

Both the LSTM and GRU models performed about the same on this particular dataset. That is to say that they both did exceptionally well. After only 5 epochs, with only a small fraction of the entire dataset (machine constraints) they where both able to reach accuracy scores of nearly 55%!


#### Bidirectional Sequence Models

A Bidirectional RNN is just like a regular RNN, but with a twist--half of the neurons start by at the beginnig of the data and work towards the end one step at a time, while the other half start at the end of the data and work towards the beginning at the same pace. Neat!

```
# a Bidirectional LSTM Network
model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1, callbacks=callbacks)
```
Epoch 1/2
36153/36153 [==============================] - 176s 5ms/step - loss: 0.1008 - accuracy: 0.9763 - val_loss: 0.0875 - val_accuracy: 0.9770

...

Epoch 2/2
36153/36153 [==============================] - 182s 5ms/step - loss: 0.0905 - accuracy: 0.9771 - val_loss: 0.0814 - val_accuracy: 0.9774
Validation accuracy of over 97.7% when trained on only 20% of the data! I think it is pretty safe to say that this model works well.

#### Summary


I did achieve some level of success with traditional machine learning models. Using pre_trained weights to help speed things up I was able to achieve an accuracy score of a little over 30% for Random Forrest, Logistic Regression, and Support Vector Machine(SVM) respectively. I know if I spent more time playing around with the various weights and parameters, I could have gotten a bit more performance from them, but these were still very time consuming, and I knew that deep learning, and Recurrent Neural Networks were going to be strongest contenders.

My traditional deep learning classification model was a much stronger (and more realistic) right out of the gate. Without having to do much tuning, and still using 20% of the entire dataset (which still took a while) my model was able to achieve after only 10 epochs a training accuracy of over 60% with a Validation accuracy of nearly 50%. At this point, I was already starting to see some sighns of overtraining, as my training score was still slowly getting better, but my Validation score was strting to fall off a bit. I am confident that even keeping with most of my parameters, using more of the dataset, and making some small adjustments to way I preprocessed my data, I could still get a bit more performance out of that type of model. However, this type of fine tuning is still very time (computational and otherwise) consuming, and I knew I had likely saved the best models for last.

When it came time for running the data through a Recurrent Neural Network, I knew I was finally making some real strides. Not that the previous models had not performed admirably, they simply were not the best tool for the job. Both my Long Short Term Memory Cells (LSTU) and Gated Recurrent Units (GRU) models performed nearly as well as one another, which was makedly better than my traditional deep neural network. Because of the way these models work, they are more computationally expensive than my previous model, so again, using only 20% of the dataset, and after only 5 epochs each (half as many) both models where already about to overtake the previous benchmark for training accuracy having scores of well over 50%, but with much stronger validation accuracy. Evan after only 5 iterations though the network, they already Validation accuracy of nearly 50%. While I am confident these models would have only continued to grow stronger with more data and more iterations (to a point), I still had my "ACE" up my sleeve, and was ready to see how it performed.

My Bidirectional Sequence Model did amazingly well! With a Validation accuracy of over 97.7% when trained on only 20% of the data! I think it is pretty safe to say that this model all those who came before it. While every model I used on this dataset did progressively better than previous (more simple) models, this particular model is in a class all it's own. I admit that I worked through these various model types in the order that I assumed would get progressively stronger results, but I am plesantly surprised at how strongly this Bidirectional Sequence Model worked. unfortunetly, my old macbook is getting on in years, so I had to pic and choose carefully what to ask of it. doing only 2 epochs of 20% of the data set still took a while to train.... that being said, the model was still getting stronger with each iteration. I am confident that had I ran the entire dataset with 20 or more epochs(on a better computer), I could achieve an accuracy score of well over 98%, and that before playing with all the tunable parameters.

##### Future Work


Continue collecting and adding to the data (always).

Try some different preprocessing techniques on the data before modeling, there are large variety of way to preprocess and "normalize" the data before modeling, I only touched on a few.

Continue playing around with the many (many) tunable parameters of the models I am working with, not only to see if it improves the models performance, but alos to gain more insight about whats going on "under the hood" so to speak, and get more intuition about whats happening with the underlying data.

