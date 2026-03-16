# ============================================================
# TF-IDF — Smarter Word Scoring
# ============================================================
# Problem with Bag of Words: common words like 'is', 'the', 'a'
# appear very often and get high counts — even though they don't
# actually tell you much about what the text is about.
#
# TF-IDF fixes this by scoring words based on two things:
#
#   TF  (Term Frequency)          — how often the word appears
#                                   in THIS document
#   IDF (Inverse Document Freq.)  — how RARE the word is across
#                                   ALL documents
#
# Result: a word that's common everywhere (like 'is') gets a LOW
# score. A word that's rare and specific (like 'pizza') gets a
# HIGH score. This way, important/distinctive words stand out.
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer


# Same four sentences as the Bag of Words example
documents = [
    "i love pizza",
    "pizza is the best",
    "i love pasta",
    "pasta is great"
]


# ------------------------------------------------------------
# RUN TF-IDF
# ------------------------------------------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary (unique words):")
print(vectorizer.get_feature_names_out())
# Output: ['best' 'great' 'is' 'love' 'pasta' 'pizza' 'the']
# Same vocabulary as BoW — but the scores will be different.

print("\nTF-IDF score table (each row = one sentence):")
print(X.toarray())

# What the scores look like (rounded for readability):
#
# Sentence                | best   great   is    love   pasta  pizza   the
# "i love pizza"          | 0      0       0     0.707  0      0.707   0
# "pizza is the best"     | 0.555  0       0.438 0      0      0.438   0.555
# "i love pasta"          | 0      0       0     0.707  0.707  0       0
# "pasta is great"        | 0      0.668   0.526 0      0.526  0       0
#
# Notice:
#  - 'love' scores 0.707 in sentences 1 and 3     → it only appears in those two
#  - 'is'   scores only 0.438 / 0.526             → it's in two sentences, less distinctive
#  - 'best' scores 0.555                          → only appears in one sentence, so it
#                                                   strongly defines that sentence
#
# The higher the score, the more that word DEFINES that particular
# sentence compared to all the others.


# ------------------------------------------------------------
# WHEN TO USE TF-IDF OVER BAG OF WORDS
# ------------------------------------------------------------
# Use TF-IDF when:
#   - Documents vary in length (a tweet vs a full article)
#   - Common filler words keep confusing the model
#   - You're doing search, topic detection, or document comparison
#
# Bag of Words is fine when:
#   - Data is short and simple
#   - You just need a quick starting point