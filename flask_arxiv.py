# create a flask app that server that asks for a query and returns the a list of 
# results. Show the results in a table containing id, title and description.
# The results should be filtered by the query.

import csv
import requests
import os
import sys
import json
from flask import Flask, request, jsonify, render_template
from jinja2 import Template
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## calculate the cosine similarity between a vector and a numpy array of vectors
cosine_similarity = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# get top-k sorted indices in descending order
top_k = lambda a, k: np.argsort(-a)[:k]

app = Flask(__name__)

with open("vector_np.data","rb") as f:
    vector_np = pickle.load(f)
    print("Docs embedding loaded")

with open("vectorizer.model","rb") as f:
    vectorizer = pickle.load(f)
    print("vectorizer loaded")
    feature_names = vectorizer.get_feature_names()

docs_df = pd.read_csv("arxiv_docs_after_2019.csv")
docs_df.head()
print("csv metadata read")

# function to render results in a table containing id, title and description
def render_results(query, tokens, results):
    return render_template('search.html', query = query, tokens = tokens, results=results)


# function to get the tfidf vector for the text
def get_tfidf_vector(text):
    sklearn_representation = vectorizer.transform([text])
    v = sklearn_representation.toarray()[0]
    return v.reshape(-1,1)


def find_similar(query_vector):
    similarities = vector_np @ query_vector
    return similarities.flatten()


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    print("query: ",query)
    K = 10
    if not query or len(query) == 0:
        rand_index = np.random.randint(0,len(docs_df))
        query = docs_df.loc[rand_index].abstract
        
    query_vector = get_tfidf_vector(query)

    top_k_tokens = top_k(query_vector.flatten(),70)
    tokens = []
    for index in top_k_tokens:
        weight = query_vector[index]
        if weight > 0.001:
            tokens.append( {"name":feature_names[index],"weight": weight} )
        
    print("top tokens: ",tokens)
    
    similarities = find_similar(query_vector)
    if not len(similarities):
        matched_ids = range(K)
    else:
        matched_ids = top_k(similarities,K)
   
    entries = []
    for ind in matched_ids:
        # format the result into a list of dictionaries
        meta = docs_df.loc[ind]
        if len(similarities):
            score = round(similarities[ind],2)
        else:
            score = 0
        entries.append({'score':score, 'id': meta.id, 'title': meta.title, 'description': meta.abstract})
    return render_results(query, tokens, entries)


# function to show page with form to enter query
@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')
  
def main():
    app.run(host='0.0.0.0', port=5000,debug=True)

if __name__ == '__main__':
    main()
