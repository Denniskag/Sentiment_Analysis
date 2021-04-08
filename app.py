import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import os

#url = 'https://inshorts.com/en/read/business'
#news_data = []
#news_category = url.split('/')[-1]


#data = requests.get(url)
#soup = bs(data.content)


urls = ['https://inshorts.com/en/read/business', 'https://inshorts.com/en/read/politics', 'https://inshorts.com/en/read/startup', 'https://inshorts.com/en/read/entertainment']
def build_dataset(urls):
  news_data = []
  for url in urls:
    news_category = url.split('/')[-1]
    data = requests.get(url)
    soup = bs(data.content)
     
    news_articles = [{'news_headline': headline.find('span', attrs = {"itemprop":"headline"}).string,
                      'news_article': article.find('div', attrs = {"itemprop":"articleBody"}).string,
                      'news_category': news_category}
                     
                     for headline,article in zip(soup.find_all('div', class_=["news-card-title news-right-box"]),
                                                soup.find_all('div', class_=["news-card-content news-right-box"]))]
    news_articles = news_articles[0:20]
    news_data.extend(news_articles)
 
  df = pd.DataFrame(news_data)
  df = df[['news_headline', 'news_article','news_category']]
  return df

df = build_dataset(urls)
df.head()

df.to_csv('news.csv', index=False)

import pandas as pd
df = pd.read_csv('news.csv')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
 
stop_words = stopwords.words('english')

import re
def remove_sp(text):
  pattern  = r'[^A-Za-z0-9]'
  text = re.sub(pattern, ' ', text)
  return text
 
def html_tag(text):
  soup = bs(text, "html.parser")
  new_text = soup.get_text()
  return new_text
 
import contractions
def con(text):
  expand = contractions.fix(text)
  return expand
 
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
 
def remove_stopwords(text):
  tokens = tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  filtered_tokens = [token for token in tokens if token not in stop_words]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_tokens



def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

df.news_headline = df.news_headline.apply(lambda x:x.lower())
df.news_article = df.news_article.apply(lambda x:x.lower())

df.news_headline = df.news_headline.apply(html_tag)
df.news_article = df.news_article.apply(html_tag)

df.news_headline = df.news_headline.apply(con)
df.news_article = df.news_article.apply(con)

df.news_headline = df.news_headline.apply(remove_sp)
df.news_article = df.news_article.apply(remove_sp)

df.news_headline = df.news_headline.apply(remove_numbers)
df.news_article = df.news_article.apply(remove_numbers)

df.news_headline = df.news_headline.apply(word_tokenize)
df.news_article = df.news_article.apply(word_tokenize)

df.news_headline = df.news_headline.apply(remove_stopwords)
df.news_article = df.news_article.apply(remove_stopwords)

String = ""
 
for i in range (0,len(df.index)) :
  for j in range (0,len(df['news_article'][i])) : 
    String = String + df['news_article'][i][j]+" "
  df['news_article'][i]=String
  String = ""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()

df['compound'] = df['news_article'].apply(lambda news_article: vs.polarity_scores(news_article)['compound'])

Store = []
for i in range (0,len(df.index)) :
  if df['compound'][i]>0 : Store.append('Positive')
  elif df['compound'][i]<0 : Store.append('Negative')
  elif df['compound'][i]==0 : Store.append('Neutral')  
  
import streamlit as st 
st.title('Sentiment Analysis')

df["Sentiment"] = Store

df
select = st.text_input('Enter text')
s=vs.polarity_scores(select)
result = ''
flag=0
#st.title(result)
if s['compound'] > 0 : 
  result='The news contains a positive sentiment'
  flag=flag+1
elif s['compound'] < 0 :
  result='The news contains a negetive sentiment'
  flag=flag+1
elif s['compound'] == 0 :
  result='The news contains a neutral sentiment'
  flag=flag+1
#st.title
#if flag>1:
st.title(result)
