import pandas as pd 
import spacy
from textblob import TextBlob
import utils
import warnings
import streamlit as st 
warnings.filterwarnings("ignore")
nlp = spacy.load('en_core_web_md')

option=st.selectbox(
   'Choose the website',
     ('Economic Times', 'CNBC', 'Hindustan Business'))
url= st.text_input("Enter the url")

if url:
#url ="https://economictimes.indiatimes.com/markets/stocks/news/oil-prices-rise-on-supply-deficit-concerns/articleshow/103770779.cms?from=mdr"
    text = utils.get_article(option,url)
    # clean the text
    clean_text= text.replace("/n", " ")
    clean_text= clean_text.replace("/", " ")       
    clean_text= ''.join([c for c in clean_text if c != "'"])

    #split the article into sentences 
    sentence=[]
    tokens = nlp(clean_text)
    for sent in tokens.sents:
        sentence.append((sent.text.strip()))
    
    
    # get the sentiment for each sentence
    textblob_sentiment=[]
    for s in sentence:    
        txt= TextBlob(s)
        a= txt.sentiment.polarity
        b= txt.sentiment.subjectivity
        textblob_sentiment.append([s,a,b])    
    
    # created df of sentence and sentiments 
    df_textblob = pd.DataFrame(textblob_sentiment, columns =['Sentence', 'Polarity', 'Subjectivity'])
    # added keywords of each sentence 
    df_textblob['Keywords']= df_textblob['Sentence'].apply(lambda x : utils.get_keywords(x))
    #display df on streamlit
    st.dataframe(df_textblob)
    #display plots on streamlit
    utils.plots(df_textblob)
    summ_text= utils.summarizer(text)
    #to display summary on streamlit
    st.write("Summary of the news article")
    st.write(summ_text)
    txt= TextBlob(summ_text)
    #sentiment analysis of summary 
    st.write("Sentiment Analysis of the summarized text")
    st.write("Polarity : "+ str(txt.sentiment.polarity))

#print(txt.sentiment.polarity)

