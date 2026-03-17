# 📓 My Notes — TF-IDF & Cosine Similarity

> These are my personal study notes. Written in plain English so future-me doesn't have to Google everything again.

---

## 🧰 TF-IDF Vectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Same idea as Bag of Words — converts text into numbers so the machine can actually understand it.

But here's the key difference:

> **BoW just counts words. TF-IDF says "okay but HOW important is this word?"**

- A word that appears everywhere (like "the", "is") gets a **low score** → not useful
- A word that appears rarely but in specific docs (like "cricket") gets a **high score** → very useful

That's the whole magic. It weights words by importance, not just frequency.

---

## ✅ Why TF-IDF is useful

- Powers search engines like **Google Search** — that's basically information retrieval
- Doesn't give too much weight to stop words
- Works well for document similarity tasks (like this movie recommendation system!)

---

## ❌ But it's not perfect...

| Problem                     | What it means in plain English                                           |
| --------------------------- | ------------------------------------------------------------------------ |
| **Sparsity**                | Most numbers in your vector are 0. A lot of wasted space.                |
| **OOV (Out of Vocabulary)** | If a word wasn't in training data, it just gets ignored. Silently.       |
| **High Dimensions**         | Thousands of features = slow + hard to work with                         |
| **Needs Lemmatization**     | "running" and "run" are treated as different words unless you preprocess |

---

## 🌱 Lemmatization — Quick Note

Converts words back to their **root/base form** before vectorizing.

```
ending   →  end
running  →  run
studies  →  study
better   →  good  (yes, really)
```

Why it matters: so "run" and "running" end up as the **same feature** in your vector. Without this, TF-IDF sees them as two completely different words.

---

---

## 📐 Cosine Similarity

Okay this is the main thing for the recommendation system. Let me break it down slowly.

---

### 🤔 What is it, actually?

It measures **how similar two pieces of text are** by comparing the vectors they produce.

But instead of asking "how far apart are these vectors?" (that would be Euclidean distance), it asks:

> **"What angle is between these two vectors?"**

This is smart because two documents can have different lengths but still talk about the same thing. The angle stays consistent, the magnitude doesn't matter.

---

### 🔢 The Formula

$$
\cos(\theta) = \frac{A \cdot B}{|A| \cdot |B|}
$$

In English:

- **A · B** = dot product of the two vectors (multiply matching positions, then add them all)
- **|A|** and **|B|** = magnitudes (lengths) of each vector
- The result is always between **-1 and 1** (for text, between **0 and 1**)

---

### 📊 Example walk-through

Take these two sentences:

```
D1 = "Virat Kohli"
D2 = "Virat"
```

**Vocabulary:** `[Virat, Kohli]`

| Sentence | Virat | Kohli |
| -------- | ----- | ----- |
| D1       | 1     | 1     |
| D2       | 1     | 0     |

Now apply the formula:

```
A · B  = (1×1) + (1×0) = 1

|A| = √(1² + 1²) = √2 ≈ 1.414
|B| = √(1² + 0²) = √1 = 1

cos(θ) = 1 / (1.414 × 1) ≈ 0.707
```

So similarity = **0.71** → moderately similar. Makes sense — they share one word out of two.

---

### 🖼️ Visual — Vector Space Diagram

```
         ↑ Kohli axis
         |
    D1 ● | (1,1)
        \|
         +-------→ Virat axis
         |   ● D2 (1,0)
```

- D1 points diagonally (has both Virat AND Kohli)
- D2 points straight right (only has Virat)
- The **angle θ between them ≈ 45°**
- cos(45°) ≈ **0.71**

---

### 📉 Angle vs Similarity — The Key Table

| Angle   | Cosine Value | What it means           |
| ------- | ------------ | ----------------------- |
| **0°**  | **1.0**      | ✅ Identical documents  |
| **45°** | **0.71**     | 👍 Pretty similar       |
| **90°** | **0.0**      | ❌ Completely different |

> **Rule of thumb:** Smaller angle = more similar. That's it.

---

### 🖼️ Intuition Diagram

```
  1.0 |●
      |  \
      |    \
 0.71 |      ●  ← our D1 vs D2 case
      |        \
      |          \
  0.0 |____________●___
      0°    45°    90°
           angle (θ)
```

As the angle grows from 0° → 90°, similarity drops from 1 → 0.

---

## 🔗 How This Connects to Movie Recommendations

In `movie_recommender.py`:

1. Movie descriptions, genres, and taglines are all combined into one big text blob per movie
2. That text is converted into TF-IDF vectors (one vector per movie)
3. Cosine similarity is calculated between the searched movie and every other movie
4. The movies with the **highest cosine scores** = most similar = get recommended

So when you search for a movie, it's basically asking:

> "Which other movie vectors point in roughly the same direction as this one?"

---

## 📁 What's in This Folder

```
movie_recommendation_system/
├── movie_recommender.py       ← the main script — runs everything
├── movies_metadata.csv        ← the raw data (~45,000 movies)
├── cosine_similarity_notes.md ← you're reading this
└── saved_models/              ← pre-computed files the future web app will load
    ├── tfidf_matrix.pkl       ← the big number grid (45k movies × 50k features)
    ├── indices.pkl            ← title → row number lookup
    ├── df.pkl                 ← the cleaned movie dataframe
    └── tfidf.pkl              ← the trained TF-IDF vectorizer
```

> **Why the saved_models folder?** Running all that TF-IDF computation takes a while.
> When we build the FastAPI web app, we don't want to redo it on every request.
> We run the script once, save the outputs, and the app just loads them instantly.

---

## 🧠 Quick Mental Summary

```
TF-IDF   → turns text into smart vectors (important words get higher scores)
Cosine   → measures the angle between those vectors
Score    → 0 (nothing in common) to 1 (identical)
```

That's the whole pipeline. Simple when you break it down.

---

_Written while building the movie recommendation system — March 2026_
_Updated after cleaning up the code and organising the folder structure._
