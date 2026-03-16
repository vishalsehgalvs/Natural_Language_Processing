# ============================================================
# Step 1 — Load and Clean the Emotions Dataset
# ============================================================
# Dataset: Kaggle emotions dataset (train.txt)
# Each line looks like:  "i feel humiliated;sadness"
# The text and the emotion label are separated by a semicolon.
# 6 possible emotions: sadness, anger, love, surprise, fear, joy
# ============================================================

import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ------------------------------------------------------------
# LOAD THE DATA
# ------------------------------------------------------------
# sep=';'       → split each line at the semicolon
# header=None   → the file has no column-name row at the top
# names=...     → give the two columns readable names

df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotions'])

# Quick check — are there any missing values?
# df.isnull().sum() → text: 0, emotions: 0  (all good, no missing data)


# ------------------------------------------------------------
# CONVERT EMOTION WORDS TO NUMBERS
# ------------------------------------------------------------
# Machine learning models only understand numbers, not words.
# So we replace 'sadness' → 0, 'anger' → 1, 'love' → 2, etc.

unique_emotions = df['emotions'].unique()
# unique_emotions = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

emotion_numbers = {}
for index, emotion in enumerate(unique_emotions):
    emotion_numbers[emotion] = index

df['emotions'] = df['emotions'].map(emotion_numbers)

# Note: sklearn's LabelEncoder does the same thing in fewer lines.
# We did it manually here so it's easier to understand what's happening.


# ------------------------------------------------------------
# STEP 1 — Make all text lowercase
# ------------------------------------------------------------
# "Feel" and "feel" should be the same word to the model.
# Lowercasing makes sure of that.

df['text'] = df['text'].apply(lambda x: x.lower())


# ------------------------------------------------------------
# STEP 2 — Remove punctuation marks
# ------------------------------------------------------------
# Things like ! , . ? ' don't add meaning in basic ML models.
# string.punctuation holds all common punctuation characters.
# translate() strips them all out in one go.

def remove_punctuations(text):
    return text.translate(str.maketrans("", "", string.punctuation))

df['text'] = df['text'].apply(remove_punctuations)


# ------------------------------------------------------------
# STEP 3 — Remove numbers
# ------------------------------------------------------------
# Numbers are usually not helpful for emotion classification.
# We loop through every character and only keep non-digits.

def remove_numbers(text):
    return "".join(char for char in text if not char.isdigit())

df['text'] = df['text'].apply(remove_numbers)


# ------------------------------------------------------------
# STEP 4 — Remove emojis and non-standard characters
# ------------------------------------------------------------
# Standard English characters all have ASCII values below 128.
# Emojis and special symbols go above that range.
# i.isascii() keeps only standard characters and drops the rest.

def remove_emojis(text):
    return "".join(char for char in text if char.isascii())

df['text'] = df['text'].apply(remove_emojis)


# ------------------------------------------------------------
# STEP 5 — Remove stopwords (filler words)
# ------------------------------------------------------------
# Stopwords are very common English words that don't carry meaning
# on their own — things like 'is', 'the', 'was', 'and', 'a'.
# NLTK has a built-in list of 198 English stopwords.
# Using a set() makes the lookup very fast.

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    # Split the sentence into individual words (tokens)
    words = word_tokenize(text)
    # Keep only words that are NOT in the stopwords list
    meaningful_words = [word for word in words if word not in stop_words]
    # Stitch the words back into a sentence
    return " ".join(meaningful_words)

df['text'] = df['text'].apply(remove_stopwords)


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------
# The dataframe is now clean and ready to be used in models.
# Before cleaning: "I didn't feel humiliated"  → emotion: sadness
# After cleaning:  "feel humiliated"            → emotion: 0

print("Cleaned data (first 5 rows):")
print(df.head())

