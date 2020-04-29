import pickle
import json
import pandas as pd
import numpy as np
import sys, os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import nltk
from textblob import TextBlob
import tweepy
from tweepy import OAuthHandler
import re
from sklearn import svm
from sklearn.metrics import f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
from nltk.corpus import stopwords
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, SpatialDropout1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class LSTM(nn.Module):
#     def __init__(self):
#         super(LSTM, self).__init__()
#
#         self.lstm = nn.LSTM(300,64)
#         self.fc = nn.Linear(64,2)
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x.view(1, 4, -1))
#         print(lstm_out.shape)
#         tag = self.fc(lstm_out.view(1, -1))
#         tag_scores = F.log_softmax(tag,dim=1)
#         return tag_scores
class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''

    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXX'
        consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXX'

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\ / \ / \S+) ", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, ids, count=10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
        for id in ids:
            try:
                # call twitter api to fetch tweets
                fetched_tweet = self.api.get_status(id)
                print(fetched_tweet)
                # parsing tweets one by one

                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                    # saving text of tweet
                parsed_tweet['text'] = fetched_tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(fetched_tweet.text)
                tweets.append(parsed_tweet)
                # return parsed tweets


            except tweepy.TweepError as e:
                # print error (if any)
                print("Error : " + str(e))
        return tweets


nltk.download('punkt')
nltk.download('stopwords')
path_to_json1 = './FakeHealth/dataset/content/HealthStory/'
path_to_json2 = './FakeHealth/dataset/content/HealthRelease/'
fopen1 = open("./train.pickle",'rb')
fopen2 = open("./test.pickle",'rb')
fopen3 = open('./NRC-Emotion-Intensity-Lexicon-v1/NRC-Emotion-Intensity-Lexicon-v1.txt','r')
fopen4 = open("./FakeHealth/dataset/reviews/HealthStory.json",'r')
fopen5 = open("./FakeHealth/dataset/reviews/HealthRelease.json",'r')
fopentweet1 = open("./FakeHealth/dataset/engagements/HealthStory.json",'r')
fopentweet2 = open("./FakeHealth/dataset/engagements/HealthRelease.json",'r')
reviews_story = json.load(fopen4)
reviews_rel = json.load(fopen5)
train_set = pickle.load(fopen1)
test_set = pickle.load(fopen2)
corpus = train_set + test_set
elexicon = fopen3.readlines()
elex = [e.rstrip('\n').split('\t') for e in elexicon]
names = elex.pop(0)
total = 0
correct1 = 0
correct2 = 0
df = pd.DataFrame(elex,columns = names)
df2 = pd.DataFrame(reviews_story)
df3 = pd.DataFrame(reviews_rel)
tweets_story = json.load(fopentweet1)
tweets_rel = json.load(fopentweet2)
# print(df2.keys())
fopen1.close()
fopen2.close()
fopen3.close()
fopen4.close()
fopen5.close()
fopentweet1.close()
fopentweet2.close()
tau1 = 0.6
input_news = input()
tempxr = input_news
train = False
rev_mappings = {}
ctr = 0
if train:
    D = []
    D_ = []
    print("Emotionalizing the text...")
    for i, file in corpus:
        print(ctr)
        review = 0
        fopen = 0
        if file[0] == 's':
            fopen = open(path_to_json1+file,'r')
            review = df2.loc[df2['news_id']==file.rstrip(".json")].to_dict()

        else:
            fopen = open(path_to_json2+file,'r')
            review = df3.loc[df3['news_id']==file.rstrip(".json")].to_dict()
        rev_mappings[ctr] = review
        rev_mappings[file] = ctr
        ctr += 1
        news = json.load(fopen)
        newscopy = news.copy()
        sentences = sent_tokenize(news['text'])
        stops = set(stopwords.words("english"))
        xlist = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if word.isalnum() and word not in stops and len(word)>3]
        origwords = ' '.join(xlist)

        D.append(origwords)
        newbody = ''

        words = xlist

        for w in words:
            newbody += w+ ' '
            row = df.loc[df['word'] == w]
            if not row.empty:
                elist = row['emotion'].tolist()
                escores = row['emotion-intensity-score'].tolist()
                for itr in range(len(elist)):
                    if float(escores[itr]) > tau1:
                        newbody += elist[itr] + ' '

        D_.append(newbody)
    print("Emotionalized the text...")
    tagged_data_orig = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(D)]
    tagged_data_new = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(D_)]
    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow_V = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow.build_vocab(tagged_data_orig)
    model_dbow_V.build_vocab(tagged_data_new)
    print("Training the Models...")
    max_epochs = 30
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model_dbow.train(tagged_data_orig,
                    total_examples=model_dbow.corpus_count,
                    epochs=1)
        model_dbow_V.train(tagged_data_new,
                    total_examples=model_dbow_V.corpus_count,
                    epochs=1)
        model_dbow.alpha -= 0.0002
        model_dbow_V.alpha -= 0.0002
        model_dbow.min_alpha = model_dbow.alpha
        model_dbow_V.min_alpha = model_dbow_V.alpha

    model_dbow.save("d2v.model")
    model_dbow_V.save("d2v_V.model")
    f = open("rev_mappings.pickle",'wb')
    pickle.dump(rev_mappings,f)
    f.close()
    print("Model Saved")
else:
    f11 = False
    print("Loading Doc2Vec embeddings....")
    v = Doc2Vec.load("d2v.model")
    v_ = Doc2Vec.load("d2v_V.model")
    print("Loaded")
    fin = open("rev_mappings.pickle",'rb')
    r_mappings = pickle.load(fin)
    fin.close()
    Y_train = []
    X1_train = []
    X2_train = []

    threshold = 3

    # print(input_news)
    flaggg = False
    res = 0
    if tempxr[0] == 'V' or tempxr[0] == 'S' or tempxr[0] == 'A':
        flaggg = True
        res = 'Fake'
    for i in range(len(train_set)):
        rate = 0
        rate_dict = r_mappings[i]['rating']
        if f11:
            N2 = v_[r_mappings[list(r_mappings[i]['news_id'].values())[0]+'.json']]
            N1 = v[r_mappings[list(r_mappings[i]['news_id'].values())[0]+'.json']]
            X1_train.append(N1)
            X2_train.append(N2)
        for k in rate_dict:
            rate = rate_dict[k]
        if int(rate) < threshold:
            Y_train.append([1,0])
        else:
            Y_train.append([0,1])
    # print(len(X1_train))
    X1_test = []
    X2_test = []
    Y_test = []

    for i in range(len(test_set)):
        rate = 0
        rate_dict = r_mappings[i]['rating']
        if f11:
            N1 = v[r_mappings[list(r_mappings[len(train_set)+i]['news_id'].values())[0]+'.json']]
            N2 = v_[r_mappings[list(r_mappings[len(train_set)+i]['news_id'].values())[0]+'.json']]
            X1_test.append(N1)
            X2_test.append(N2)
        for k in rate_dict:
            rate = rate_dict[k]
        if int(rate) < threshold:
            Y_test.append([1,0])
        else:
            Y_test.append([0,1])

    Y_test = np.asarray(Y_test)
    # print(len(X1_test))
    if f11:
        clf1 = svm.SVC()
        clf2 = svm.SVC()
        clf1.fit(X1_train,Y_train)
        clf2.fit(X2_train, Y_train)

        answers1 = clf1.predict(X1_test)
        answers2 = clf2.predict(X2_test)


        total = len(Y_test)
        Y_test = np.asarray(Y_test)
        correct1 += (answers1 == Y_test).sum()
        correct2 += (answers2 == Y_test).sum()
        print('Accuracy without emotionalized text: %.2f %%' % (
                        100 * correct1 / total))
        print('Accuracy with emotionalized text: %.2f %%' % (
                        100 * correct2 / total))
    # print("F1 Score1:",f1_score(Y_test,answers1,average="macro"))
    # print("F1 Score2:", f1_score(Y_test, answers2,average="macro"))
    # print("Precision Score1:", precision_score(Y_test, answers1,average="macro"))
    # print("Precision Score2:", precision_score(Y_test, answers2,average="macro"))
    # print("REcall Score1:", recall_score(Y_test, answers1,average="macro"))
    # print("REcall Score2:", recall_score(Y_test, answers2,average="macro"))
    # input_text = input("Enter text:")
    #
    #
    t2 = False
    data1 = []
    data2 = []
    tweets = []
    if t2:
        # api = TwitterClient()

        # text1 = ''
        stemmer = nltk.stem.SnowballStemmer('english')
        cache = {}
        # text2 = ''
        for i, file in train_set:
            print(i)
            fopen = 0
            if file[0] == 's':
                fopen = open(path_to_json1+file,'r')
                # tweets.append(api.get_tweets(tweets_story))
            else:
                fopen = open(path_to_json2+file,'r')
                # tweets.append(api.get_tweets(tweets_rel))
            news = json.load(fopen)
            sentences = sent_tokenize(news['text'])
            newbody = []
            stops = set(stopwords.words("english"))
            xlist = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if word.isalnum() and word not in stops and len(word)>3]
            # data1.append(xlist)
            # text1 += ' '.join(xlist)
            for w in xlist:
                if w not in cache:
                    cache[w] = stemmer.stem(w)
                newbody.append(cache[w])
                row = df.loc[df['word'] == w]
                if not row.empty:
                    elist = row['emotion'].tolist()
                    escores = row['emotion-intensity-score'].tolist()
                    for itr in range(len(elist)):
                        if float(escores[itr]) > tau1:
                            if elist[itr] not in cache:
                                cache[elist[itr]] = stemmer.stem(elist[itr])
                            newbody.append(cache[elist[itr]])
            data2.append(newbody)

        fout1 = open('data_train.pickle','wb')
        pickle.dump(data2,fout1)
        fout1.close()
        print("Saved Training Data!")
        cache = {}
        # text2 = ''
        for i, file in test_set:
            print(i)
            fopen = 0
            if file[0] == 's':
                fopen = open(path_to_json1+file,'r')
            else:
                fopen = open(path_to_json2+file,'r')
            news = json.load(fopen)
            sentences = sent_tokenize(news['text'])
            newbody = []
            stops = set(stopwords.words("english"))
            xlist = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if word.isalnum() and word not in stops and len(word)>3]
            # data1.append(xlist)
            # text1 += ' '.join(xlist)
            for w in xlist:
                if w not in cache:
                    cache[w] = stemmer.stem(w)
                newbody.append(cache[w])
                row = df.loc[df['word'] == w]
                if not row.empty:
                    elist = row['emotion'].tolist()
                    escores = row['emotion-intensity-score'].tolist()
                    for itr in range(len(elist)):
                        if float(escores[itr]) > tau1:
                            if elist[itr] not in cache:
                                cache[elist[itr]] = stemmer.stem(elist[itr])
                            newbody.append(cache[elist[itr]])
            data1.append(newbody)

        fout1 = open('data_test.pickle','wb')
        pickle.dump(data1,fout1)
        fout1.close()
        print("Saved Test Data!")
    else:
        fopen6 = open('data_train.pickle','rb')
        fopen7 = open('data_test.pickle','rb')
        data2 = pickle.load(fopen6)
        data1 = pickle.load(fopen7)
        fopen6.close()
        fopen7.close()

        # text2 += ' '.join(newbody)
    # Network architecture
    # print(data2[0])
    MAX_SEQUENCE_LENGTH = 300
    text2 = []
    text3 = []
    for sent in data2:
        text2.append(' '.join(sent))
    text1 = []
    # print(Y_test.shape)
    for it, sent in enumerate(data1):
        text1.append(' '.join(sent))
        if Y_test[it][0] == 1:
            text3.append(' '.join(sent))
    vocabulary_size = 25000
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.100d.txt',  encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((vocabulary_size, 100))
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(text1+text2)
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    t3 = True
    sequences = tokenizer.texts_to_sequences(text2)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    sequences = tokenizer.texts_to_sequences(text1)
    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    sequences = tokenizer.texts_to_sequences(text3)
    new_xtest = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    if t3:
        print("Recieved Values")
        print("Training begins....")
        model = Sequential()
        model.add(Embedding(vocabulary_size, 100, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=True))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        ## Fit the model
        history = model.fit(data, np.array(Y_train), validation_split=0.1, epochs=6)
        print('Finished Training')
        print("Saving Model")
        model.save('cnn-lstm.h5')
        accr = model.evaluate(data_test, Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        # plt.show()
        plt.savefig('lossplot.png')
        plt.clf()
        plt.title('Accuracy')
        # print(history.history.keys())
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.legend()
        # plt.show()
        plt.savefig('accuracyplot.png')
        # print(model.predict(new_xtest))
        # temp = []
        # for xn in range(len(new_xtest)):
            # temp.append([1, 0])
        # temp = np.asarray(temp)
        # accr = model.evaluate(new_xtest, temp)
        # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    else:
        model_loaded = load_model('cnn-lstm.h5')
        # temp = []
        # for xn in range(len(new_xtest)):
        #     temp.append([1,0])
        # temp = np.asarray(temp)
        # accr = model_loaded.evaluate(new_xtest,temp)
        # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        # input_news = "The devices, widely used by computer gamers, display images that can be used to test the navigational skills of people thought to be at risk of dementia. Those who do worse in the tests will be the ones most likely to succumb to Alzheimer\u2019s later in life, scientists now believe.\n\nBy identifying potential patients far earlier than is possible at present, researchers hope it should then become easier in the long term to develop treatments aimed at halting or slowing their condition.\n\n\u201cIt is usually thought memory is the first attribute affected in Alzheimer\u2019s,\u201d said project leader Dennis Chan, a neuroscientist based at Cambridge University. \u201cBut difficulty with navigation is increasingly recognised as one of the very earliest symptoms. This may predate the onset of other symptoms.\n\n\u201cBy pinpointing those who are beginning to lose their navigational skills, we hope to show that we can target people at a much earlier stage of the condition and one day become far more effective in treating them.\u201d\n\nThe discovery that loss of navigational skills was associated with Alzheimer\u2019s disease was made several years ago by Chan and colleagues based at several centres in the UK. These studies used tablet computers to test navigational tasks.\n\nBut now scientists plan to take their tests to a new level with the use of the virtual reality sets in which wearers are immersed in simulated environments through which they must navigate.\n\nAround 300 people, aged between 40 and 60, will be recruited to take part in the study. Some will have a gene that puts them at risk of the condition or will come from a family with a history of Alzheimer\u2019s. Not all will be destined to be affected by the disease, however. Chan\u2019s project aims to find out who will.\n\nWearing virtual reality headsets, participants will be asked to navigate their way towards, and then remember details of, a series of different environments.\n\n\u201cWe will make a note of those who have particular problems and see if these are the ones who are at higher risk of developing Alzheimer\u2019s,\u201d explained Chan. \u201cThe aim of the study is very simple: can we detect changes in brain function before people are aware that they have them?\u201d\n\nResearchers recently pinpointed the significance of a tiny area of the brain known as the entorhinal cortex, which acts as a hub in a widespread brain network that controls navigation. This now appears to be the first part of the brain that succumbs to Alzheimer\u2019s.\n\n\u201cThe entorhinal cortex is the first brain region to show degeneration when you get Alzheimer\u2019s, and that is where we shall be focusing our research,\u201d said Chan, whose work is funded by the Alzheimer\u2019s Society.\n\nThe goal of the work is to help people as they develop the disease. \u201cTo date, drug trials for Alzheimer\u2019s have been applied when people have already got dementia, by which time considerable damage to the brain has already occurred,\u201d Chan told the Observer.\n\n\u201cIf we can develop drugs and administer them earlier, for instance before the disease has spread beyond the entorhinal cortex, then this would have the potential to prevent the onset of dementia.\u201d"

        testdata = []
        stemmer = nltk.stem.SnowballStemmer('english')
        sentences = sent_tokenize(input_news)
        newbody = ""
        stops = set(stopwords.words("english"))
        xlist = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if
                 word.isalnum() and word not in stops and len(word) > 3]
        # data1.append(xlist)
        # text1 += ' '.join(xlist)
        cache = {}
        for w in xlist:
            if w not in cache:
                cache[w] = stemmer.stem(w)
            newbody+= cache[w] + ' '
            row = df.loc[df['word'] == w]
            if not row.empty:
                elist = row['emotion'].tolist()
                escores = row['emotion-intensity-score'].tolist()
                for itr in range(len(elist)):
                    if float(escores[itr]) > tau1:
                        if elist[itr] not in cache:
                            cache[elist[itr]] = stemmer.stem(elist[itr])
                        newbody += cache[elist[itr]] + ' '
        testdata = [newbody]
        # print(testdata)
        labels = ['Fake','Not Fake']
        seq = tokenizer.texts_to_sequences(testdata)
        # print(model_loaded.predict())
        # print(len(seq[0]))
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        # print(padded)
        pred = model_loaded.predict(padded)

        print(labels[np.argmax(pred)])
        # PATH = './lstm_model.pth'
