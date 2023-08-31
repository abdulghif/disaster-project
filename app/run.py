import json
import plotly
import pandas as pd
from src import   custom_multiclf_score
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster.db')
df = pd.read_sql_table('messages_clean_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():

    # Extract data for the visuals

    # Visual 1: Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Visual 2: Most Frequent Categories
    category_sums = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_sums.index)

    # Visual 3: Proportion of 'Related' Messages
    related_counts = df['related'].value_counts()
    related_names = ['Related', 'Not Related']

    # Visual 4: Most Frequent Words in Messages
    words = []
    for message in df['message']:
        tokens = tokenize(message)
        words.extend(tokens)
    word_count = Counter(words)
    common_words = word_count.most_common(10)
    common_word_names = [word[0] for word in common_words]
    common_word_values = [word[1] for word in common_words]

    # Create visuals
    graphs = [
        # Visual 1
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {'title': 'Distribution of Message Genres', 'xaxis': {'title': 'Genre'}, 'yaxis': {'title': 'Count'}}
        },
        # Visual 2
        {
            'data': [Bar(x=category_names, y=category_sums)],
            'layout': {'title': 'Most Frequent Categories', 'xaxis': {'title': 'Category'}, 'yaxis': {'title': 'Count'}}
        },
        # Visual 3
        {
            'data': [Pie(labels=related_names, values=related_counts)],
            'layout': {'title': 'Proportion of Messages that are Related to Disasters'}
        },
        # Visual 4
        {
            'data': [Bar(x=common_word_names, y=common_word_values)],
            'layout': {'title': 'Most Frequent Words in Messages', 'xaxis': {'title': 'Word'}, 'yaxis': {'title': 'Count'}}
        }
    ]

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with Plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()