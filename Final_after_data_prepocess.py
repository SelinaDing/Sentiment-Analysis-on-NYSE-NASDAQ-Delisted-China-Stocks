# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#First, we make the necessary imports; project_helper contains various utility 
# and graph functions.

import nltk
import numpy as np
import pandas as pd
import pickle
import pprint
import project_helper


from tqdm import tqdm

# Then we download the stopwords corpus for removing stopwords and wordnet corpus 
# for lemmatizing.

nltk.download('stopwords')
nltk.download('wordnet')

#To lookup 10-k documents, we use each company’s unique CIK (Central Index Key).
#please run the convert function first
cik_lookup =  {'AAC':'1099290','CBPO':'1369868'}
# WE prepare the data we need
data = pd.read_excel('https://github.com/SelinaDing/Sentiment-Analysis-on-NYSE-NASDAQ-Delisted-China-Stocks/blob/main/cik.xlsx?raw=true',index_col=None)
#print(data.head(3))
new_data = data.filter(['Ticker','CIK'],axis=1)
CIK_dict1 = new_data.set_index('Ticker')['CIK'].to_dict()
#print(CIK_dict1)
#Now we pull a list of filed 10-ks from the SEC and display Amazon data as an example.

sec_api = project_helper.SecAPI()
from bs4 import BeautifulSoup
def get_sec_data(cik, doc_type, start=0, count=60):
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)]
    return entries
example_ticker = 'AAC'
sec_data = {}
#change cik_lookup into cik_dict1
for ticker, cik in CIK_dict1.items():
    sec_data[ticker] = get_sec_data(cik, '10-K')
pprint.pprint(sec_data[example_ticker][:5])

#We received a list of urls pointing to files containing metadata related to each filling. Metadata isn’t relevant to us so we pull the filling by replacing the url with the filling url. Let’s view download progress by using tqdm and look at an example document.
raw_fillings_by_ticker = {}
for ticker, data in sec_data.items():
    raw_fillings_by_ticker[ticker] = {}
    for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
        if (file_type == '10-K'):
            file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')            
            
            raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)
print('Example Document:\n\n{}...'.format(next(iter(raw_fillings_by_ticker[example_ticker].values()))[:1000]))

#remove the NAs in sec_data
for key in list(sec_data.keys()):
    if not sec_data.get(key):
        sec_data.pop(key)
#Break downloaded files into their associated documents, which are sectioned off in the fillings with the tags <DOCUMENT> for the start of each document and </DOCUMENT> for the end of each document.

import re
def get_documents(text):
    extracted_docs = []
    
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')   
    doc_start_is = [x.end() for x in      doc_start_pattern.finditer(text)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(text)]
    
    for doc_start_i, doc_end_i in zip(doc_start_is, doc_end_is):
            extracted_docs.append(text[doc_start_i:doc_end_i])
    
    return extracted_docs
filling_documents_by_ticker = {}
for ticker, raw_fillings in raw_fillings_by_ticker.items():
    filling_documents_by_ticker[ticker] = {}
    for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker), unit='filling'):
        filling_documents_by_ticker[ticker][file_date] = get_documents(filling)
print('\n\n'.join([
    'Document {} Filed on {}:\n{}...'.format(doc_i, file_date, doc[:200])
    for file_date, docs in filling_documents_by_ticker[example_ticker].items()
    for doc_i, doc in enumerate(docs)][:3]))

#Define the get_document_type function to return the type of document given.
def get_document_type(doc):
    
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    
    doc_type = type_pattern.findall(doc)[0][len('<TYPE>'):] 
    
    return doc_type.lower()

#Filter out the non 10-k documents from the fillings using the get_document_type function.
ten_ks_by_ticker = {}
for ticker, filling_documents in filling_documents_by_ticker.items():
    ten_ks_by_ticker[ticker] = []
    for file_date, documents in filling_documents.items():
        for document in documents:
            if get_document_type(document) == '10-k':
                ten_ks_by_ticker[ticker].append({
                    'cik': cik_lookup[ticker],
                    'file': document,
                    'file_date': file_date})
project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['cik', 'file', 'file_date'])

# Preprocess Data
# remove the html and make all text lowercase to clean up the document text

def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    return text
def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    
    return text

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Cleaning {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_clean'] = clean_text(ten_k['file'])
project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_clean'])


# Now we lemmatize all the data.
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
def lemmatize_words(words):

    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]
    
    return lemmatized_words
word_pattern = re.compile('\w+')
for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Lemmatize {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = lemmatize_words(word_pattern.findall(ten_k['file_clean']))
project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_lemma'])
 
# remove stopwords
from nltk.corpus import stopwords
lemma_english_stopwords = lemmatize_words(stopwords.words('english'))
for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = [word for word in ten_k['file_lemma'] if word not in lemma_english_stopwords]
        
print('Stop Words Removed')
