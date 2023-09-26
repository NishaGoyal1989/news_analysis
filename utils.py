import spacy
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import pytextrank
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import warnings
warnings.filterwarnings("ignore")
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('textrank')
def get_keywords(sentence):   
    keywords=[]
    doc= nlp(sentence)
    for phrase in doc._.phrases:
        keywords.append(phrase.text)   
    return keywords

#function to get the summary of news article
def summarizer(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    output= summarizer(text)
    output="".join(map(str,output))
    return output

#function to web scrap the article 
def get_article(option,url):    
    r= requests.get(url)
    #Setting the correct text encoding of the HTML page
    r.encoding = 'utf-8'
    #Extracting the HTML from the request object
    html = r.text
    # Creating a BeautifulSoup object from the HTML
    soup = BeautifulSoup(html)
    # Getting the text out of the soup for different news sites
    if option =='Economic Times':
        text = soup.find('div',class_= 'artText').get_text()
    if option=='CNBC' :    
        text= soup.find('div',class_="ArticleBody-articleBody").get_text()
    if option=='Hindustan Business':     
        table= soup.find('div',class_="bl-news-section-split")        
        para = table.find_all('p')
        text = ' '.join ( p.text for p in para)
    return text

#function to create plots 
def plots(df):
    fig, ax = plt.subplots(1,2,figsize=(70,30))
    sns.set(font_scale=7.0)
    sns.histplot(df["Polarity"], ax= ax[0]).set(title="Sentence Polarity")
    sns.histplot(df["Subjectivity"],ax=ax[1]).set(title="Sentence Subjectivity")
    st.pyplot(fig)

