# -*- coding: utf-8 -*-
"""
Created on Sun May  1 20:04:55 2016

@author: viral
"""

import os
import glob
from textblob import TextBlob    
import csv

os.chdir("D:\\MY ALL DESKTOP DATA\\Spring 2016\\Data Science 5378\\Final Project\\TwitterData\\28thApril\\")
files = glob.glob("*.txt") 
#print(files)

f = csv.writer(open("D:\\MY ALL DESKTOP DATA\\Spring 2016\\Data Science 5378\\Final Project\\TwitterScore\\28thAprilNegativeTweetScoreHourlyBasis.csv","w"),lineterminator='\n')
f.writerow(['Hours','Score','Negative Tweets'])


for item in files:
    negativeCount = 0
    time = item[9:11]
    in_file = open(item, "r")  
    tweetsTogether = in_file.read()
    in_file.close()
    textBlobOfWords = TextBlob(tweetsTogether)
    print('For hour'+ str(time) + ' the score is')
    #print(textBlobOfWords)
    for sentence in textBlobOfWords.sentences:
        print(sentence.sentiment.polarity)
    in_file1 = open(item,"r")
    tweets_list = in_file1.read().splitlines()
    for eachTweetInHour in tweets_list:
        text = TextBlob(eachTweetInHour)
        if sentence.sentiment.polarity < 0.0:
            negativeCount += 1
    f.writerow([time,sentence.sentiment.polarity,negativeCount])
    
    