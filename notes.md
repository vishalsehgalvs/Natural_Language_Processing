# NLP Project — Code Notes

These are my personal notes on every file in this repo, written so I can come back months later and actually understand what I was doing and why.

Read them top to bottom — each file builds on the previous one.

---

## The Big Picture

This project teaches a computer to read a sentence and figure out the emotion behind it — sadness, anger, love, surprise, fear, or joy.

The steps are:

1. `clean_text_data.py` — load the dataset and clean the text
2. `bag_of_words.py` — demo: turn words into counts (simple)
3. `tfidf_scoring.py` — demo: turn words into smarter scores
4. `train_and_test_models.py` — train models and see who wins

---

## clean_text_data.py

### What this file does

Loads a Kaggle dataset of 16,000 sentences, each labelled with an emotion.
Cleans the text so a machine learning model can actually use it.

### The Dataset (train.txt)

Each line looks like:

```
i didnt feel humiliated;sadness
i feel romantic too;love
i feel as confused about life as a teenager;fear
```

Two things per line, separated by a semicolon — the sentence and the emotion.
6 possible emotions: sadness, anger, love, surprise, fear, joy.

### Loading it

```python
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotions'])
```

- `sep=';'` — split each line at the semicolon
- `header=None` — the file has no column names on the first row
- `names=[...]` — give the two columns readable names ourselves

No missing values (both columns come back zero nulls).

### Converting emotion words to numbers

ML models only understand numbers. So sadness becomes 0, anger becomes 1, love becomes 2, etc.

```python
for index, emotion in enumerate(unique_emotions):
    emotion_numbers[emotion] = index
df['emotions'] = df['emotions'].map(emotion_numbers)
```

We build a dictionary like {'sadness': 0, 'anger': 1, ...} and replace the words.
(sklearn's LabelEncoder would do the same thing in 2 lines — we did it manually so it's obvious what's happening.)

### Cleaning steps

**1. Lowercase everything**
"Feel" and "feel" should be the same word.

```python
df['text'] = df['text'].apply(lambda x: x.lower())
```

**2. Remove punctuation**
`string.punctuation` contains all common punctuation chars. `translate()` strips them all in one go.

```python
df['text'] = df['text'].apply(remove_punctuations)
```

**3. Remove numbers**
Loop every character, keep only non-digits. So "iphone 15" becomes "iphone ".

```python
return "".join(char for char in text if not char.isdigit())
```

**4. Remove emojis**
Standard English chars all have ASCII values below 128. Emojis go above 127.
`char.isascii()` keeps standard chars, drops everything else.

```python
return "".join(char for char in text if char.isascii())
```

**5. Remove stopwords (filler words)**
Stopwords are words that appear everywhere and do not add meaning — is, the, was, and, a.
NLTK has a built-in list of 198 English stopwords. Using a set() makes lookups very fast.

```python
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
return " ".join(word for word in words if word not in stop_words)
```

`word_tokenize` splits the sentence into words properly (handles contractions better than just splitting on spaces).

### Before and after

```
Before:  "I didn't feel humiliated"  ->  sadness
After:   "feel humiliated"           ->  0
```

The noise is gone. Only the meaningful words remain.

---

## bag_of_words.py

### What this file does

A standalone demo showing how Bag of Words works.
No real dataset here — just four simple sentences to keep it easy to follow.

### The core idea

Represent each sentence as a list of word counts.
`CountVectorizer` from sklearn does all the work.

```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
```

- `fit` — scans all sentences and builds a vocabulary (list of unique words)
- `transform` — converts each sentence into a count vector
- Single-character words like 'i' are removed by default

### What the output looks like

Vocabulary: [best, great, is, love, pasta, pizza, the]

| Sentence            | best | great | is  | love | pasta | pizza | the |
| ------------------- | ---- | ----- | --- | ---- | ----- | ----- | --- |
| "i love pizza"      | 0    | 0     | 0   | 1    | 0     | 1     | 0   |
| "pizza is the best" | 1    | 0     | 1   | 0    | 0     | 1     | 1   |
| "i love pasta"      | 0    | 0     | 0   | 1    | 1     | 0     | 0   |
| "pasta is great"    | 0    | 1     | 1   | 0    | 1     | 0     | 0   |

### Bigrams (word pairs)

Problem: "not good" gets split into "not" and "good" separately.
The model might learn "good" = positive and get confused.

Bigrams fix this — they include two-word phrases as features too.

```python
CountVectorizer(ngram_range=(1, 2))
```

Now "love pizza" and "love pasta" are separate features. The model knows the difference.

### Trigrams (groups of three words)

```python
CountVectorizer(ngram_range=(3, 3))
```

Only 3 features come out of our short sentences — short text does not have many 3-word combos.

### Practical tip

Use `ngram_range=(1, 2)` in real projects. You get single words AND pairs — good balance between power and table size.

---

## tfidf_scoring.py

### What this file does

Same four sentences as bag_of_words.py, but using TF-IDF scoring instead of plain counts.

### The problem with plain counts

Common words like "is", "the", "a" appear in many sentences and get high counts — even though they do not tell you much about what the sentence is really about.

### How TF-IDF fixes it

Every word gets a score based on two things:

- **TF (Term Frequency)** — how often the word appears in THIS sentence
- **IDF (Inverse Document Frequency)** — how rare the word is ACROSS ALL sentences

If a word is common everywhere (like "is") its IDF is low, so its overall score is low.
If a word is rare and specific (like "pizza") its IDF is high, so its score stays high.

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
```

### Reading the scores

| Sentence            | best  | great | is    | love  | pasta | pizza | the   |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| "i love pizza"      | 0     | 0     | 0     | 0.707 | 0     | 0.707 | 0     |
| "pizza is the best" | 0.555 | 0     | 0.438 | 0     | 0     | 0.438 | 0.555 |
| "i love pasta"      | 0     | 0     | 0     | 0.707 | 0.707 | 0     | 0     |
| "pasta is great"    | 0     | 0.668 | 0.526 | 0     | 0.526 | 0     | 0     |

- "love" scores 0.707 in sentences 1 and 3 — it only appears in those two, so it is distinctive
- "is" scores only 0.438/0.526 — it appears in multiple sentences, so it is less useful
- "best" scores 0.555 — it only appears once, so it strongly defines that sentence

### When to use TF-IDF vs plain counts

Use TF-IDF when:

- Your documents vary in length
- Common filler words keep confusing the model
- You are doing search or document comparison

Bag of Words is fine when:

- Data is short and simple
- You just need a quick starting point

---

## train_and_test_models.py

### What this file does

Takes the cleaned data from clean_text_data.py and trains three different ML models.
Compares their accuracy to see which one does best at predicting emotions.

### Splitting the data

We hold back 20% of the data for testing so we can measure accuracy honestly.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['emotions'], test_size=0.20, random_state=42
)
```

- Training set: 12,800 sentences (80%)
- Testing set: 3,200 sentences (20%)
- `random_state=42` — fixed seed so the split is the same every run

### The fit_transform vs transform rule

**On training data**: use `fit_transform` — this builds the vocabulary AND converts to numbers.
**On test data**: use `transform` only — converts using the SAME vocabulary, never refit.

If you fit on test data too, the model gets a sneak peek at words from the test set — the results are no longer fair.

```python
X_train_bow = bow_vectorizer.fit_transform(X_train)  # learn vocab + convert
X_test_bow  = bow_vectorizer.transform(X_test)        # convert only, same vocab
```

### Model 1 — Naive Bayes + Bag of Words

```python
nb_bow_model = MultinomialNB()
nb_bow_model.fit(X_train_bow, y_train)
```

Naive Bayes is the classic spam filter algorithm. It works well with whole-number word counts.

Result: **76.8% accuracy**

### Model 2 — Naive Bayes + TF-IDF

You would expect TF-IDF to do better. But with Naive Bayes it actually does WORSE.
Why? Naive Bayes was designed for whole-number counts. TF-IDF produces decimal scores that do not suit it as well.

Result: **66.1% accuracy**

### Model 3 — Logistic Regression + TF-IDF

Logistic Regression handles TF-IDF decimal scores much better.
It draws a decision boundary between emotion classes and is generally more powerful for text.

`max_iter=1000` gives the model enough steps to fully converge (the default of 100 is often not enough for text data).

Result: **86.2% accuracy — winner**

### Results side by side

| Model               | How words were converted | Accuracy  |
| ------------------- | ------------------------ | --------- |
| Naive Bayes         | Bag of Words             | 76.8%     |
| Naive Bayes         | TF-IDF                   | 66.1%     |
| Logistic Regression | TF-IDF                   | **86.2%** |

Takeaway: the choice of model matters just as much as how you turned words into numbers.

---

## Quick Reminders

- Always clean text before any model — garbage in, garbage out
- Use a `set()` for stopwords, not a list — lookups are much faster
- `word_tokenize` handles contractions better than splitting on spaces
- Never `fit_transform` on test data — only `transform`. That would be cheating.
- TF-IDF almost always beats plain Bag of Words for real tasks — it is one extra line of code
- `ngram_range=(1, 2)` is a safe default — adds bigrams without making the table huge
- `random_state=42` in train_test_split keeps results reproducible across runs

---

## movie_recommendation_system/movie_recommender.py

### What this file does

A separate mini-project. Instead of classifying emotions, this one recommends similar movies.
You give it a movie name and it comes back with 10 movies that feel the most like it.

It uses the same tools from above — TF-IDF and cosine similarity — but applies them differently.

### The dataset

`movies_metadata.csv` — about 45,000 movies.
Key columns we use: `title`, `overview` (the plot), `genres`, `tagline`.

### The approach

1. Combine each movie's plot + genres + tagline into one big blob of text called `tags`
2. Clean that text the same way we do in this project (lowercase, remove filler words, lemmatize)
3. Convert every movie's tags into a TF-IDF vector
4. When someone asks for recommendations, calculate cosine similarity between their movie and every other movie
5. Return the 10 highest scoring matches

### What cosine similarity does here

Think of each movie as a point in space. Movies that talk about similar things (same genres, similar plot words) will point in the same direction. Cosine similarity measures that angle — closer to 0° means more similar.

Full notes on the theory are in `cosine_similarity_notes.md` inside the same folder.

### The saved_models folder

Running TF-IDF on 45,000 movies takes a while.
So after building everything, we save it all as pickle files in `saved_models/`:

| File             | What's inside                                     |
| ---------------- | ------------------------------------------------- |
| tfidf_matrix.pkl | The number grid — 45,000 movies × 50,000 features |
| indices.pkl      | Title → row number lookup                         |
| df.pkl           | The cleaned movie dataframe                       |
| tfidf.pkl        | The trained TF-IDF vectorizer                     |

When we build the FastAPI web app, it'll just load these files on startup instead of re-computing everything.

### Example output

```python
recommend('Toy Story')
# → Toy Story 2, Toy Story 3, Small Fry, Superstar Goofy, ...
```
