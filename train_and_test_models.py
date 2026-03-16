# ============================================================
# Step 4 — Train Models and Compare Their Accuracy
# ============================================================
# We now take the cleaned data from clean_text_data.py and
# train three different models to predict emotions from text.
#
# Models we'll try:
#   1. Naive Bayes       + Bag of Words  → 76.8% accuracy
#   2. Naive Bayes       + TF-IDF        → 66.1% accuracy
#   3. Logistic Regression + TF-IDF     → 86.2% accuracy  ← best
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from clean_text_data import df   # load the cleaned dataset from Step 1


# ------------------------------------------------------------
# SPLIT DATA INTO TRAINING AND TESTING
# ------------------------------------------------------------
# We hold back 20% of the data for testing.
# The model trains on the other 80% and is then tested on the
# 20% it has never seen before — so the accuracy score is honest.
#
# Total rows: 16,000
#   Training set: 12,800 rows  (80%)
#   Testing set:   3,200 rows  (20%)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['emotions'],
    test_size=0.20,
    random_state=42   # fixed seed so results are reproducible
)


# ============================================================
# MODEL 1 — Naive Bayes with Bag of Words
# ============================================================
# Bag of Words turns each sentence into word counts.
# Naive Bayes is good at working with word counts — it's the
# classic algorithm for spam detection and text classification.
#
# fit_transform on training data: build vocab + convert to numbers
# transform on test data: convert using the SAME vocab (no refitting)

bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow  = bow_vectorizer.transform(X_test)

nb_bow_model = MultinomialNB()
nb_bow_model.fit(X_train_bow, y_train)
predictions_bow = nb_bow_model.predict(X_test_bow)

print("Model 1 — Naive Bayes + Bag of Words")
print("Accuracy:", accuracy_score(y_test, predictions_bow))
# Result: ~76.8%


# ============================================================
# MODEL 2 — Naive Bayes with TF-IDF
# ============================================================
# TF-IDF gives less weight to common words and more to rare,
# distinctive ones. You'd expect this to do better than BoW.
# But actually it does slightly WORSE with Naive Bayes here.
# Why? Naive Bayes works best with whole-number counts,
# and TF-IDF produces decimal scores that don't suit it as well.

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

nb_tfidf_model = MultinomialNB()
nb_tfidf_model.fit(X_train_tfidf, y_train)
predictions_nb_tfidf = nb_tfidf_model.predict(X_test_tfidf)

print("\nModel 2 — Naive Bayes + TF-IDF")
print("Accuracy:", accuracy_score(y_test, predictions_nb_tfidf))
# Result: ~66.1%  (worse than BoW with Naive Bayes — see note above)


# ============================================================
# MODEL 3 — Logistic Regression with TF-IDF  ← Best result
# ============================================================
# Logistic Regression handles the decimal TF-IDF scores much
# better. It finds a decision boundary between emotion classes
# and is generally more powerful for text classification.
#
# max_iter=1000 gives the model enough steps to fully converge
# (the default of 100 is often not enough for text data).

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
predictions_lr = lr_model.predict(X_test_tfidf)

print("\nModel 3 — Logistic Regression + TF-IDF")
print("Accuracy:", accuracy_score(y_test, predictions_lr))
# Result: ~86.2%  ← best of the three


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
# | Model                        | Vectorizer   | Accuracy |
# |------------------------------|--------------|----------|
# | Naive Bayes                  | Bag of Words |  76.8%   |
# | Naive Bayes                  | TF-IDF       |  66.1%   |
# | Logistic Regression          | TF-IDF       |  86.2%   | ← winner
#
# Takeaway: the choice of MODEL matters as much as the choice
# of how you turn words into numbers.











