# ============================================================
# Bag of Words (BoW) — Turning Sentences Into Word Counts
# ============================================================
# The idea: represent each sentence as a list of numbers.
# Each number = how many times a particular word appears.
# Word order doesn't matter — only the word counts matter.
# That's why it's called a "Bag" of words (like words in a bag).
# ============================================================

from sklearn.feature_extraction.text import CountVectorizer


# Four simple sentences we'll use as our example
documents = [
    "i love pizza",
    "pizza is the best",
    "i love pasta",
    "pasta is great"
]


# ------------------------------------------------------------
# BASIC BAG OF WORDS (single words only)
# ------------------------------------------------------------
# CountVectorizer does two things:
#   1. Builds a vocabulary — a list of all unique words
#   2. Converts each sentence into a count vector
# Note: single-character words like 'i' are removed by default.

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("=== Basic Bag of Words ===")
print("Vocabulary (unique words found):", vectorizer.get_feature_names_out())
# Output: ['best' 'great' 'is' 'love' 'pasta' 'pizza' 'the']
# Note: 'i' is dropped because it's a single character.

print("Word count table (each row = one sentence):")
print(X.toarray())
# Each row is a sentence, each column is a word from the vocabulary.
# The number = how many times that word appears in that sentence.
#
# Sentence                | best  great  is  love  pasta  pizza  the
# "i love pizza"          |  0     0     0    1     0      1      0
# "pizza is the best"     |  1     0     1    0     0      1      1
# "i love pasta"          |  0     0     0    1     1      0      0
# "pasta is great"        |  0     1     1    0     1      0      0


# ------------------------------------------------------------
# BIGRAMS — looking at word PAIRS instead of single words
# ------------------------------------------------------------
# Problem with basic BoW: "not good" gets split into two words.
# The model sees 'not' and 'good' separately and misses the meaning.
# Bigrams fix this by including two-word phrases as features.
#
# ngram_range=(1, 2) means: include both single words AND pairs.

vectorizer_bigram = CountVectorizer(ngram_range=(1, 2))
X_bigram = vectorizer_bigram.fit_transform(documents)

print("\n=== Bigrams (single words + word pairs) ===")
print("Vocabulary:", vectorizer_bigram.get_feature_names_out())
# Now includes pairs like: 'is great', 'is the', 'love pasta', 'love pizza', etc.
# The model can now tell 'love pizza' and 'love pasta' apart as distinct features.


# ------------------------------------------------------------
# TRIGRAMS — looking at groups of THREE words
# ------------------------------------------------------------
# ngram_range=(3, 3) means: only include 3-word groups.
# Useful when 3-word phrases carry the meaning (e.g. 'not very good').

vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))
X_trigram = vectorizer_trigram.fit_transform(documents)

print("\n=== Trigrams (three-word groups only) ===")
print("Vocabulary:", vectorizer_trigram.get_feature_names_out())
# Output: ['is the best', 'pasta is great', 'pizza is the']
# Only 3 features — short sentences don't have many 3-word combinations.


# ------------------------------------------------------------
# PRACTICAL TIP
# ------------------------------------------------------------
# In real projects, ngram_range=(1, 2) is the most common choice.
# You get both single words AND pairs — good balance of
# understanding vs table size.