from flask import Flask, render_template, request, url_for, redirect, flash, g, session
from selectorlib import Extractor
import requests
import json
from time import sleep
import csv
from dateutil import parser as dateparser
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import pandas as pd
import numpy as np
import re
import string
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import nltk
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
from imp import reload

e = Extractor.from_yaml_file('selectors.yml')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("./model/")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])


def scrape(url):
    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s" % url)
    r = requests.get(url, headers=headers)
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print(
                "Page %s was blocked by Amazon. Please try using better proxies\n" % url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d" % (
                url, r.status_code))
        return None
    # Pass the HTML of the page and create
    return e.extract(r.text)

def data_scrapping(prodcutid):
    with open('./static/data/data.csv', 'w', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=[
                                "title", "content", "date", "verified", "author", "rating"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i in range(1, 50):
            url = 'https://www.amazon.in/product-reviews/'+prodcutid+'/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={}'.format(
                i)
            data = scrape(url)
            if data['reviews']:
                for r in data['reviews']:
                    if 'verified' in r:
                        if r['verified'] == None:
                            continue
                        if 'Verified Purchase' in r['verified']:
                            r['verified'] = 'Yes'
                        else:
                            r['verified'] = 'Yes'
                    r['rating'] = r['rating'].split(' out of')[0]
                    date_posted = r['date'].split('on ')[-1]
                    r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
                    writer.writerow(r)
            else:
                break

        with open('./static/data/data.csv', 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            original_list = list(reader)
            cleaned_list = list(filter(None, original_list))

        with open('./static/data/final_data.csv', 'w', newline='', encoding="utf-8") as output_file:
            wr = csv.writer(output_file, dialect='excel')
            for data in cleaned_list:
                wr.writerow(data)

def classify_review():
    df = pd.read_csv('./static/data/final_data.csv')
    df['review'] = df['title'] + df['content']
    df['review'] = df['review'].dropna()
    reviews = df['content'].values.tolist()

    with open('./static/data/classified_data.csv', 'w', newline ='', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["review", "sentiment"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        p = 0
        n = 0
        for i in reviews:
            try:
                test_sentence = i
                predict_input = tokenizer.encode(test_sentence,truncation=True,padding=True,return_tensors="tf")
                
                tf_output = loaded_model.predict(predict_input)[0]
                tf_prediction = tf.nn.softmax(tf_output, axis=1)
                
                labels = ['Negative','Positive']
                label = tf.argmax(tf_prediction, axis=1)
                label = label.numpy()
                sentiment = labels[label[0]]
                
                if (labels[label[0]] == 'Positive'):
                    p += 1
                else:
                    n += 1
                
                data = [ [test_sentence, sentiment] ]
                file = open('./static/data/classified_data.csv', 'a+', newline ='')
                with file:    
                    write = csv.writer(file)
                    write.writerows(data)
            except:
                pass

    return p, n

# removes all the special characters (Symbols, emojis) present in the review. #
def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w)>3))])
    
    return text2.lower()

# remove stop words for eg. I, was, an, etc. #
def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

# Lematizing words. For eg, Develop, developed, development .. all will get converted to 'Develop' #
def lematization(texts, allowed_postags=['NOUN' , 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(sent)
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output

# Coherence Value - Finding the number of topics appropriate for the particular dataset in which the feactures/topics can be fitted properly. 
# A set of statements or facts is said to be coherent, if they support each other. Thus, a coherent fact set can be interpreted in a context that covers all or most of the facts.
# C_v measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics  in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence = 'u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        
    return model_list, coherence_values

def lda_graph(df, sentiment_name):
    df.dropna(axis=0, how='any', inplace=True)
    df['review'] = df['review'].apply(clean_text)
    df['num_words_text'] = df['review'].apply(lambda x:len(str(x).split()))
    df['review'] = df['review'].apply(remove_stopwords)
    text_list = df['review'].tolist()
    tokenized_reviews = lematization(text_list)
    dictionary = corpora.Dictionary(tokenized_reviews)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]

    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=tokenized_reviews, start=2, limit=50, step=1)
    max_value = max(coherence_values)
    max_index = coherence_values.index(max_value) + 2

    print(str(max_index) + " " + str(max_value))

    best_model = model_list[max_index]

    ldamodel= best_model

    vis = pyLDAvis.gensim_models.prepare(best_model, doc_term_matrix, dictionary)
    filename = "./templates/"+sentiment_name+"graph.html"
    pyLDAvis.save_html(vis, filename)


app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    context = {}
    if request.method == 'POST':
        productid = request.form['productid']
        data_scrapping(productid)

        p, n = classify_review()

        # p = 100
        # n = 50

        df = pd.read_csv('./static/data/classified_data.csv', encoding='cp1252')
        df = df[1:]
        positive = df[df['sentiment'] == 'Positive']
        negative = df[df['sentiment'] == 'Negative']

        lda_graph(positive, 'positive')
        lda_graph(negative, 'negative')

        context = {
            'p': p,
            'n': n
        }
    return render_template('index.html', **context)

@app.route("/positivegraph")
def positivegraph():
    return render_template('positive.html')

@app.route("/negativegraph")
def negativegraph():
    return render_template('negative.html')


if __name__ == "__main__":
    app.run(debug=True)