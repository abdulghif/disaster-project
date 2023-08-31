import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files.
    
    Args:
        messages_filepath (str): Path to the CSV file containing messages.
        categories_filepath (str): Path to the CSV file containing categories.

    Returns:
        pd.DataFrame: Loaded messages and categories as separate DataFrames.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories

def clean_data(messages, categories):
    """
    Clean and transform the raw data.

    Args:
        messages (pd.DataFrame): DataFrame containing messages.
        categories (pd.DataFrame): DataFrame containing categories.

    Returns:
        pd.DataFrame: Cleaned and transformed DataFrame.
    """
    df = messages.merge(categories, on='id')
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df[df['related']!=2]
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Save cleaned data to a SQLite database.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        database_filename (str): Name of the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages_clean_table', engine, index=False)

def main():
    """
    Main function to execute data processing pipeline.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument.\n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
