# -*- coding: utf-8 -*-
"""
Created on Sun May 1 19:12:34 2016

@author: viral
"""
import re
import tweepy
import nltk

api_key = "IVFA1Mu0uekSuU30IVpZClj1e"
api_secret = "52EFa6sU0YtBE5JB2l9CM7uf223v9yJWYvbdl2MJPD67nZ6Hh8"
access_token = "62769323-9VD4Q3N7vluQgN8566PedW7FoYDcxt5lU77BBljGz"
token_secret = "hJsmOSXuBBArXgyt4bdaAAWFrwncD6zGg7hVfOoRuA86D"

stopwords = nltk.corpus.stopwords.words("english")
customList = ['rt','https', 'http']
stopwords = stopwords + customList


def preprocessText(tweet):
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+',' ',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #final cleanup
    validLetters = "abcdefghijklmnopqrstuvwxyz "
    tweet = ''.join([char for char in tweet if char in validLetters])
    words = tweet.split()
    
    tweetsWithoutStopWords = [w for w in words if w not in stopwords]
    tweet = ' '.join(tweetsWithoutStopWords)
    return tweet


auth = tweepy.auth.OAuthHandler(api_key,api_secret)
auth.set_access_token(access_token, token_secret)
count = 0
tweets = []
api = tweepy.API(auth)
for tweet in tweepy.Cursor(api.search,
                           q="#dfw delays",
                           since="2016-04-26",
                           until="2016-04-29",
                           lang="en").items():
    tweets.append(tweet)
    count +=1
    if count == 500:
        break
    
for item in tweets:
    date = str(item.created_at)
    date = date.split()
    actdate = date[0].split('-')
    currentDate = ['04','28']
    if currentDate[0] in actdate and currentDate[1] in actdate:
        time = date[1].split(':')
        text = preprocessText(item.text)
        out_file1 = open("D:\\MY ALL DESKTOP DATA\\Spring 2016\\Data Science 5378\\Final Project\\TwitterData\\28thApril\\TweetsFor"+ time[0]+"Hour.txt",'a')
        out_file1.write(text +'\n')
        out_file1.close()
        #print(item.created_at)
        