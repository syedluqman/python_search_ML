import pandas as pd
import os
import glob
import time
import datetime
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define database structure to store file metadata
file_metadata = pd.DataFrame(columns=['file_name', 'file_path', 'file_data'])

# Define function to scrape file metadata and store it in the database
def scrape_files(directory_path):
    # Use glob to find all csv files in the directory
    for file_path in glob.glob(os.path.join(directory_path, '*.csv')):
        # Read csv file as a dataframe
        df = pd.read_csv(file_path)
        # Concatenate all the rows and columns in the dataframe into a single string
        file_data = ' '.join(df.stack().astype(str).tolist())
        # Append file metadata to the database
        file_metadata.loc[len(file_metadata)] = [os.path.basename(file_path), file_path, file_data]

# Define function to search for files based on search terms
def search_files(search_term):
    # Calculate the fuzzy match score between the search term and each file's data
    file_metadata['fuzzy_match_score'] = file_metadata['file_data'].apply(lambda x: fuzz.token_set_ratio(x, search_term))
    # Use machine learning to calculate the cosine similarity between the search term and each file's data
    count_vect = CountVectorizer()
    file_data_matrix = count_vect.fit_transform(file_metadata['file_data'])
    cosine_sim = cosine_similarity(count_vect.transform([search_term]), file_data_matrix).flatten()
    # Calculate the weighted average score of the fuzzy match score and cosine similarity for each file
    file_metadata['score'] = 0.7 * file_metadata['fuzzy_match_score'] + 0.3 * cosine_sim * 100
    # Sort files by score in descending order
    file_metadata.sort_values(by=['score'], ascending=False, inplace=True)
    # Return a list of file paths and scores that match the search term
    search_results = []
    for index, row in file_metadata.iterrows():
        if row['score'] > 50:
            search_results.append((row['file_path'], row['score']))
    return search_results

# Define function to update the database every Tuesday morning at 6am
def update_database():
    while True:
        current_time = datetime.datetime.now().strftime('%H:%M')
        if current_time == '06:00':
            # Scrape file metadata and store it in the database
            scrape_files('/path/to/directory')
            # Wait until the next Tuesday to update the database again
            next_tuesday = datetime.datetime.now() + datetime.timedelta(days=(1 - datetime.datetime.now().weekday()) % 7, hours=6)
            sleep_time = (next_tuesday - datetime.datetime.now()).total_seconds()
            time.sleep(sleep_time)

# Define function to build the user interface
def build_user_interface():
    while True:
        # Prompt user to enter a search term
        search_term = input('Enter a search term: ')
        # Search for files based on the search term
        search_results = search_files(search_term)
        # Display search results to the user
        print('Search Results:')
        for result in search_results:
            print(result[0],
