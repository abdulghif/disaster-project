import sys
import pandas as pd
from src import get_score, score
from sqlalchemy import create_engine

import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

import pickle

def load_data(database_filepath):
    """
    Load and preprocess data from a SQLite database.

    Parameters:
        database_filepath (str): Filepath of the SQLite database.

    Returns:
        X (pandas Series): Messages.
        Y (pandas DataFrame): Categories.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    print(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages_clean_table',con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X,Y

def tokenize(text):
    """
    Tokenize and preprocess input text.

    Parameters:
        text (str): Input text.

    Returns:
        words (list): List of preprocessed words.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return words


def build_model():
    """
    Build a text classification model pipeline.

    Returns:
        pipeline (sklearn Pipeline): Model pipeline.
    """
    vectorizer = TfidfVectorizer( tokenizer=tokenize ,use_idf=True, smooth_idf=True, sublinear_tf=False)

    rf = RandomForestClassifier(random_state=123)

    pipeline = Pipeline([
        ('vectorizer',vectorizer),
        ('model',rf)
    ])

    parameters = {
        'model__max_depth':[2,4,None],
        'model__max_features':['sqrt','log2'],
    }

    cv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=score,
        cv=5, 
        refit=True, 
        return_train_score=True,
        verbose=10
    )
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate model performance on test data.

    Parameters:
        model (sklearn Pipeline): Trained classification model.
        X_test (pandas Series): Test messages.
        Y_test (pandas DataFrame): True categories.
    """
    y_pred = model.predict(X_test)
    test_results = []
    for i,column in enumerate(Y_test.columns):
        result = get_score(Y_test.loc[:,column].values,y_pred[:,i])
        test_results.append(result)
    test_results_df = pd.DataFrame(test_results)
    print('Average F1-Score:',test_results_df['F1-score'].mean())
    print('Average Accuracy:',test_results_df['Accuracy'].mean())
    print('Average Recall:',test_results_df['Recall'].mean())
    print('Average Precision:',test_results_df['Precision'].mean())

def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Parameters:
        model (sklearn Pipeline): Trained classification model.
        model_filepath (str): Filepath to save the model.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()