import ast
import os

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# NLP libraries — these are what clean and understand the text
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# -------------------------------------------------------
# Step 1 — Load the movie data
# -------------------------------------------------------
# we have roughly 45,000 movies in this CSV file
# each row is one movie and has info like title, genres, overview (plot), tagline, ratings etc.

df = pd.read_csv('movies_metadata.csv')


# -------------------------------------------------------
# Step 2 — Remove duplicate rows
# -------------------------------------------------------
# there were 13 duplicate entries hiding in the data
# dropping them so the same movie doesn't show up twice in recommendations

df = df.drop_duplicates().reset_index(drop=True)


# -------------------------------------------------------
# Step 3 — Keep only the columns we actually need
# -------------------------------------------------------
# the original dataset has 24 columns — most of them we don't need for recommendations
# we only care about: title, overview (the plot), genres, tagline, rating, popularity

df = df[['title', 'overview', 'genres', 'tagline', 'vote_average', 'popularity']]
# shape after this: (45453, 6) — 45k movies, 6 columns


# -------------------------------------------------------
# Step 4 — Handle missing values
# -------------------------------------------------------
# some movies have no title at all — those are useless, drop them
df = df.dropna(subset=['title'])

# overview and tagline are text fields so we can't fill them with an average number
# just leave them as empty strings so the rest of the code doesn't break
df['overview'] = df['overview'].fillna("")


# -------------------------------------------------------
# Step 5 — Clean up the genres column
# -------------------------------------------------------
# right now genres look like this in the file:
# [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]
#
# that's a mess — we just want the names as plain text: "Animation Comedy Family"
# ast.literal_eval reads the string as actual Python data, then we pull out just the 'name' values

df['genres'] = df['genres'].apply(lambda x: ' '.join([i['name'] for i in ast.literal_eval(x)]))

df['tagline'] = df['tagline'].fillna("")


# -------------------------------------------------------
# Step 6 — Combine overview + genres + tagline into one big text field called 'tags'
# -------------------------------------------------------
# when comparing two movies, we want to look at everything about them — not just the plot
# by combining all the text into one column, the TF-IDF vectorizer can see the full picture
#
# example for Toy Story:
# "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear
#  onto the scene. Afraid of losing his place in Andy's heart... Animation Comedy Family"

df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']


# -------------------------------------------------------
# Step 7 — Download the NLTK tools we need (runs once, then cached)
# -------------------------------------------------------
# stopwords = common filler words like "the", "is", "a", "an" — removing these cuts the noise
# wordnet = the dictionary that lemmatization uses to find root forms of words

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# -------------------------------------------------------
# Step 8 — Define a function that cleans a piece of text
# -------------------------------------------------------
# this function does 3 things to each movie's tags:
#   1. lowercase everything — so "Action" and "action" are treated as the same word
#   2. remove punctuation and numbers — they don't add anything useful here
#   3. lemmatize and remove stop words — "running" becomes "run", "the" gets removed
#
# the result is a clean string of meaningful root words ready for vectorizing

def preprocess_text(txt):
    txt = str(txt).lower()
    txt = re.sub(r'[^a-zA-Z\s]', "", txt)
    words = txt.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# run the cleaning function on every movie's tags column
df['tags'] = df['tags'].apply(preprocess_text)

# after cleaning, Toy Story's tags look like:
# "led woody andy toy live happily room andy birthday bring buzz lightyear onto scene afraid
#  lose place andy heart woody plot buzz circumstance separate ... animation comedy family"


# -------------------------------------------------------
# Step 9 — Build a lookup table: movie title → row number
# -------------------------------------------------------
# this lets us quickly find a movie by name when someone asks for recommendations
# drop_duplicates() handles cases where two different movies share the exact same title

df = df.reset_index(drop=True)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# example of what this looks like:
# Toy Story     → row 0
# Jumanji       → row 1
# The Matrix    → row ...
# and so on for all 45,000 movies


# -------------------------------------------------------
# Step 10 — Convert all movie tags into TF-IDF vectors
# -------------------------------------------------------
# TF-IDF converts text into numbers while being smart about it
# common words like "the" get a low score, rare meaningful words get a higher score
#
# settings we're using:
# max_features=50000  → only keep the top 50,000 most useful words/phrases
# ngram_range=(1, 2)  → look at single words AND two-word combos like "action comedy"
# stop_words='english' → extra pass to remove filler words

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# result: a grid of ~45,000 rows (movies) × 50,000 columns (features)
# each row is one movie's fingerprint as a list of numbers


# -------------------------------------------------------
# Step 11 — The recommendation function
# -------------------------------------------------------
# given a movie title, this finds the n most similar movies
#
# here's how it works step by step:
#   1. look up which row this movie sits on in our data
#   2. calculate cosine similarity between that movie's vector and every other movie's vector
#      (cosine similarity = how similar the "direction" of two vectors is, score 0 to 1)
#   3. sort all movies by their similarity score, highest first
#   4. return the top n titles — skipping index 0 because that's the movie itself

def recommend(title, n=10):
    if title not in indices:
        return ['movie not found in our database — double check the spelling']
    idx = indices[title]
    similarity_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_index = similarity_score.argsort()[::-1][1:n + 1]
    return df['title'].iloc[similar_index]


# quick test — what does Toy Story recommend?
# recommend('Toy Story')
# → Toy Story 2, Toy Story 3, Small Fry, Superstar Goofy, and a few others


# -------------------------------------------------------
# Step 12 — Save the model pieces as pickle files
# -------------------------------------------------------
# when we build the FastAPI web app later, we don't want to re-run all this heavy
# processing every single time someone asks for a recommendation
# instead we save everything here once, and the app just loads these files and goes
#
# what each file is:
#   tfidf_matrix.pkl  → the big number grid — one row per movie, 50k columns
#   indices.pkl       → the title → row number lookup table
#   df.pkl            → the cleaned movie dataframe with all the info
#   tfidf.pkl         → the trained TF-IDF vectorizer (needed to process new search terms later)

os.makedirs('saved_models', exist_ok=True)

pickle.dump(tfidf_matrix, open('saved_models/tfidf_matrix.pkl', 'wb'))
pickle.dump(indices, open('saved_models/indices.pkl', 'wb'))
df.to_pickle('saved_models/df.pkl')
pickle.dump(tfidf, open('saved_models/tfidf.pkl', 'wb'))

print("Done! All model files saved in the saved_models/ folder.")
