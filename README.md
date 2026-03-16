# Natural Language Processing (NLP)

> My personal notes on how computers learn to understand human language.

---

## Files in This Repo

| File | What it does |
| --- | --- |
| clean_text_data.py | Loads the Kaggle emotions dataset and cleans the text - lowercase, remove punctuation, numbers, emojis, and filler words |
| bag_of_words.py | Shows how Bag of Words turns sentences into word counts, including bigrams and trigrams |
| tfidf_scoring.py | Shows how TF-IDF scores words smarter than plain word counts |
| train_and_test_models.py | Trains three ML models on the cleaned data and compares their accuracy |
| train.txt | The Kaggle emotions dataset - 16,000 sentences each labelled with one of 6 emotions: sadness, anger, love, surprise, fear, joy |
| notes.md | Plain notes on what each file does and why, written so they make sense weeks later |

---

## What is NLP?

Computers normally only understand 0s and 1s — they have no idea what a word means. **Natural Language Processing (NLP)** is the field of AI that teaches computers to read, understand, and work with human language — things like sentences, questions, reviews, and conversations.

Think of it as building a translator between human language and computer logic.

---

## Where Do We See NLP in Real Life?

You interact with NLP every single day without realising it:

- Typing _"Weather in Delhi"_ into Google
- Saying _"Alexa, play this song"_
- Gmail filtering your spam emails
- Reading product reviews summarised automatically
- Chatbots answering your questions on websites

| Where                            | What NLP does there                                                  |
| -------------------------------- | -------------------------------------------------------------------- |
| Virtual Assistants (Alexa, Siri) | Understands what you say and figures out what you want               |
| Email Spam Filters               | Reads the email and decides if it looks suspicious                   |
| Product Review Analysis          | Reads thousands of reviews and tells you if people liked the product |
| Chatbots                         | Reads your message and writes a reply that makes sense               |

The goal in all of these is the same — **figure out what you actually mean** and give you a useful response.

---

## Two Practical Examples

### Spam Email Detection

When you get an email saying _"You won a lottery!"_, NLP reads that text and recognises the suspicious phrasing. It then stamps the email as **spam** so it doesn't reach your inbox.

### Figuring Out If a Review is Positive or Negative (Sentiment Analysis)

NLP can read a product review and decide whether the person was happy, unhappy, or somewhere in between:

| How the person felt    | Label    |
| ---------------------- | -------- |
| Happy / good feedback  | Positive |
| Unhappy / bad feedback | Negative |
| Neither good nor bad   | Neutral  |

For example, after a product launch a company could feed thousands of reviews into an NLP model and instantly know if customers liked it or not — without reading every review manually.

---

## Three Ways NLP Has Been Built Over the Years

### 1. The "Write Every Rule By Hand" Approach

The earliest NLP systems worked by having someone sit down and write out rules manually.

**How it worked:**

- A programmer would write rules like: _if the sentence contains "not good", mark it as a negative review_
- The computer would check every sentence against those written rules

**Why this approach fell apart:**

- Language is messy and full of exceptions — you can't write a rule for every possible sentence
- Sarcasm, slang, regional phrases — none of this can be handled with simple written rules
- The list of rules would need to be millions of lines long to cover real-world language
- You'd have to update the rules every time language changed

---

### 2. The "Learn from Examples" Approach (Machine Learning)

Instead of writing rules by hand, this approach feeds the computer thousands of labelled examples and lets it figure out the patterns on its own.

**How it worked:**

- Give the computer 10,000 product reviews, each labelled as Positive, Negative, or Neutral
- It learns what kind of words and phrases tend to appear in each type
- When it sees a new review, it uses what it learned to make a prediction

**Algorithms used in this approach:**

- Naive Bayes
- Logistic Regression
- Support Vector Machines (SVM)

**One catch — computers still can't read words directly:**
Before training, all text has to be turned into numbers first. Two common ways to do this are:

- **Bag of Words (BoW)** — count how many times each word appears in a document
- **TF-IDF** — similar to BoW but gives less weight to very common words like "the" or "is"

**Where this approach breaks down — it misses the meaning:**

> _"Staying all night debugging is what I dreamed of."_

| Who's reading it               | What they think it means                        |
| ------------------------------ | ----------------------------------------------- |
| The ML model                   | Sounds like someone passionate about their work |
| What the person actually meant | Almost certainly sarcastic or complaining       |

This approach counts words but doesn't understand how they sit together in a sentence. So sarcasm, tone, and word order are basically invisible to it.

---

### 3. The "Deep Understanding" Approach (Deep Learning / Modern NLP)

This is the approach behind tools like ChatGPT, Google Translate, and Gmail's autocomplete. Instead of counting words or following rules, these systems use large brain-inspired networks that learn the deep meaning behind language.

**Models used in this approach:**

- RNN (reads text one word at a time, remembers the previous words)
- LSTM (a smarter RNN that has a longer memory)
- GRU (similar to LSTM, slightly simpler)
- **Transformers** (the current gold standard — reads the whole sentence at once and understands relationships between all words simultaneously)

**Popular transformer-based models:**

- **BERT** (Google — great at understanding text)
- **GPT** (OpenAI — great at generating text)

These models understand that _"he didn't like it"_ and _"he hated it"_ mean roughly the same thing, even though the words are completely different. They understand context, sarcasm, tone, and intent in a way the earlier approaches could not.

**How words are turned into numbers here:**

In this approach, words are turned into number lists in a smart way — words with similar meanings get similar numbers. This is called a **word embedding**.

- Word2Vec
- GloVe

These tools turn words into number lists where the closeness of the numbers reflects the closeness of meaning. The famous example:

> King − Man + Woman ≈ Queen

This means the model has genuinely learned that "King" and "Queen" are related the same way "Man" and "Woman" are.

Modern models go one step further — they break text into small pieces (called tokens), give each token a number, then run those numbers through many layers to figure out the full meaning in context.

---

## How You Actually Build an NLP System (The Pipeline)

### Step 1 — Get Your Text Data

You can't build anything without data. Common sources:

- Ready-made datasets from Kaggle or similar sites
- Scraping text from websites
- Calling an API that gives you data
- Collecting it manually
- Crowdsourcing (paying people to write or label examples)
- Generating it automatically using existing NLP tools

### Step 2 — Clean the Text

Raw text from the real world is messy. Before you can do anything useful with it, you need to clean it up. Here's what that usually looks like:

| What you do                    | Why                                                            | Example                   |
| ------------------------------ | -------------------------------------------------------------- | ------------------------- |
| Make everything lowercase      | So "Hello" and "hello" are treated as the same word            | `HELLO` → `hello`         |
| Remove punctuation and symbols | They usually don't add meaning in basic models                 | `Hello!` → `Hello`        |
| Remove numbers (sometimes)     | Depend on your task — remove if irrelevant                     | `iPhone 15` → `iphone`    |
| Remove website links and HTML  | They break the analysis and add no meaning                     | `https://...` → _(gone)_  |
| Remove emojis                  | They can confuse basic models                                  | 😊 → _(gone)_             |
| Remove filler words            | Words like "is", "was", "the", "and" add noise without meaning | "the cat sat" → "cat sat" |
| Fix spelling mistakes          | Typos create new phantom words the model won't recognise       | `recieve` → `receive`     |

> **Worth knowing:** Advanced models like BERT and GPT are smart enough that they actually _want_ to keep punctuation, filler words, and even emojis — because those details help them understand tone and context better. The cleaning steps above are mainly for the simpler approaches.

---

## Quick Comparison of All Three Approaches

| Approach                           | How it works                                              | Examples         |
| ---------------------------------- | --------------------------------------------------------- | ---------------- |
| Write rules by hand                | Someone manually lists every rule                         | Grammar rules    |
| Learn from examples (ML)           | Feed labelled data, the model finds patterns              | Naive Bayes, SVM |
| Deep understanding (Deep Learning) | Large networks learn meaning from massive amounts of text | BERT, GPT        |

---

## Turning Words Into Numbers

This is one of the most important ideas in NLP. Computers only work with numbers, so every word in every sentence has to be turned into a number (or a list of numbers) before any model can learn from it.

There are two broad styles of doing this:

| Style                | Methods                                | What the numbers look like                            |
| -------------------- | -------------------------------------- | ----------------------------------------------------- |
| Old / simple style   | One-Hot Encoding, Bag of Words, TF-IDF | Long lists with mostly zeros                          |
| Modern / smart style | Word2Vec, GloVe, FastText, BERT        | Compact lists where similar words get similar numbers |

---

## One-Hot Encoding (the Simplest Method)

One-Hot Encoding is the most basic way to turn words into numbers. Every word gets its own unique slot in a list, and that slot is set to 1 while everything else stays 0.

Before you can do this, you need two things:

**1. A Corpus** — your collection of text documents.

| Document | Text                       |
| -------- | -------------------------- |
| Doc1     | Akarsh watch sheryians     |
| Doc2     | Harsh also watch sheryians |
| Doc3     | sheryians teach Akarsh     |

**2. A Word List (Vocabulary)** — every unique word that appears across all your documents.

```
[Akarsh, watch, sheryians, Harsh, also, teach]
Total unique words = 6
```

### Giving Each Word Its Own Pattern

Since there are 6 unique words, each word gets a list of 6 numbers — a 1 in its own position, and 0 everywhere else:

| Word      | Its number pattern   |
| --------- | -------------------- |
| Akarsh    | `[1, 0, 0, 0, 0, 0]` |
| watch     | `[0, 1, 0, 0, 0, 0]` |
| sheryians | `[0, 0, 1, 0, 0, 0]` |
| Harsh     | `[0, 0, 0, 1, 0, 0]` |
| also      | `[0, 0, 0, 0, 1, 0]` |
| teach     | `[0, 0, 0, 0, 0, 1]` |

### What a Full Document Looks Like

You just stack the number patterns for each word in the sentence.

**Doc1:** _Akarsh watch sheryians_ → 3 words × 6 slots = a 3×6 grid

```
[1, 0, 0, 0, 0, 0]  ← Akarsh
[0, 1, 0, 0, 0, 0]  ← watch
[0, 0, 1, 0, 0, 0]  ← sheryians
```

**Doc2:** _Harsh also watch sheryians_ → 4 words × 6 slots = a 4×6 grid

```
[0, 0, 0, 1, 0, 0]  ← Harsh
[0, 0, 0, 0, 1, 0]  ← also
[0, 1, 0, 0, 0, 0]  ← watch
[0, 0, 1, 0, 0, 0]  ← sheryians
```

**Doc3:** _sheryians teach Akarsh_ → 3 words × 6 slots = a 3×6 grid

```
[0, 0, 1, 0, 0, 0]  ← sheryians
[0, 0, 0, 0, 0, 1]  ← teach
[1, 0, 0, 0, 0, 0]  ← Akarsh
```

### What's Good About It

- Dead simple to understand
- Easy to code
- Works fine when your word list is small

### Where It Falls Apart

**1. Too many zeros** — with a vocabulary of even a few thousand words, each word's number pattern would be thousands of numbers long with only a single 1. That's a huge waste of memory and makes the model slow.

```
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]
```

**2. Can't handle new words** — if a word never appeared in your original word list, the model has no way to represent it. It simply doesn't know what to do with it.

```
Word list = [Akarsh, watch, sheryians]
New word   = "learning"  →  the model has no slot for this
```

**3. Every document has a different size** — a 3-word sentence and a 10-word sentence produce grids of different heights, which causes problems for many models.

```
Doc1 → 3 × 6 grid
Doc2 → 4 × 6 grid
```

---

## Bag of Words (BoW)

One-Hot Encoding just marks whether a word exists — it doesn't tell you how often it appears in a document. Bag of Words fixes that. Instead of 1s and 0s, it counts how many times each word shows up.

The name makes sense once you think about it — imagine shaking all the words out of a sentence into a bag. The bag doesn't care what order they were in; it just knows which words are in there and how many of each.

### The Same Example Sentences

| Document | Text                      |
| -------- | ------------------------- |
| D1       | Akash watches Shreyans    |
| D2       | Harsh also watch Shreyans |
| D3       | Shreyans teach Shreyans   |

**Word list (Vocabulary):** `[Akash, watch, Shreyans, Harsh, also, teach]`

### Counting Words Per Document

Instead of marking 1 for "present", we write the actual count of how many times each word appears:

| Document | Akash | watch | Shreyans | Harsh | also | teach |
| -------- | ----- | ----- | -------- | ----- | ---- | ----- |
| D1       | 1     | 1     | 1        | 0     | 0    | 0     |
| D2       | 0     | 1     | 1        | 1     | 1    | 0     |
| D3       | 0     | 0     | 2        | 0     | 0    | 1     |

Notice D3 — "Shreyans" appears **twice** in that sentence, so its count is 2. One-Hot Encoding would have missed that entirely.

### The Resulting Number Lists (Vectors)

```
D1 = [1, 1, 1, 0, 0, 0]
D2 = [0, 1, 1, 1, 1, 0]
D3 = [0, 0, 2, 0, 0, 1]
```

These lists can now be fed directly into a machine learning model.

### How Similarity Works (the Geometry Idea)

Think of each word as its own direction in space. Each document becomes an arrow pointing in a direction based on its word counts.

- Documents that share a lot of the same words will point in roughly the **same direction** (small angle between them)
- Documents with completely different words will point in **very different directions** (large angle)

This is the idea behind **cosine similarity** — measuring how closely two documents point in the same direction. It's used in search engines, recommendation systems, and document grouping.

> If D1 and D2 both contain "watch" and "Shreyans", their arrows point somewhat in the same direction — so they'd be considered similar.

### What's Good About BoW

| Pros                       | Why it matters                                                                   |
| -------------------------- | -------------------------------------------------------------------------------- |
| Simple and fast to compute | Easy to build, runs quickly even on large datasets                               |
| Captures word frequency    | Knows "Shreyans" appeared twice, not just once — unlike One-Hot                  |
| Works well for basic tasks | Spam detection, topic classification, and document search all work fine with BoW |
| Easy to understand         | You can look at the table and immediately see what's going on                    |

### Where BoW Falls Short

| Cons                                  | Example                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------- |
| Word order is completely ignored      | "dog bites man" and "man bites dog" produce the exact same vector                         |
| No understanding of meaning           | "good" and "excellent" are treated as totally unrelated words                             |
| Common words drown out important ones | "the" appears 50 times, "cancer" appears 3 times — the model pays more attention to "the" |
| Still mostly zeros                    | With a large vocabulary, most cells in the table are 0 — same memory waste as One-Hot     |
| Can't handle new words                | A word not seen during training has no column in the table                                |

> This last problem — common words drowning out important ones — is exactly what **TF-IDF** was designed to solve.

---

## Bigrams and Trigrams — Teaching BoW to Remember Word Pairs

The biggest weakness of plain BoW is that it treats every word in isolation. The phrase "not good" gets split into two separate words — "not" and "good" — and the model loses the fact that they go together and change each other's meaning.

**N-grams** are a simple fix for this. Instead of only looking at single words, you also look at groups of consecutive words.

| N-gram type              | What it looks at      | Example from "the food was not good"           |
| ------------------------ | --------------------- | ---------------------------------------------- |
| **Unigram** (normal BoW) | One word at a time    | `the`, `food`, `was`, `not`, `good`            |
| **Bigram**               | Two words at a time   | `the food`, `food was`, `was not`, `not good`  |
| **Trigram**              | Three words at a time | `the food was`, `food was not`, `was not good` |

### Why This Helps

With plain BoW:

- "not good" → two separate features: `not` and `good`
- The model might learn `good` = positive, and get confused

With bigrams:

- "not good" → one feature: `not good`
- The model can now learn that `not good` = negative

### Example

Take this sentence: **"the food was not good"**

**Unigrams** (what normal BoW sees):

```
[the, food, was, not, good]
```

**Bigrams** (pairs of consecutive words):

```
[the food, food was, was not, not good]
```

**Trigrams** (groups of three):

```
[the food was, food was not, was not good]
```

You can combine all of these together. A model using **unigrams + bigrams** would have both single words and word pairs as features — giving it a better shot at understanding phrases.

### The Trade-off

|                             | Unigrams only | Unigrams + Bigrams | Unigrams + Bigrams + Trigrams |
| --------------------------- | ------------- | ------------------ | ----------------------------- |
| Understands word pairs?     | No            | Yes                | Yes                           |
| Understands 3-word phrases? | No            | No                 | Yes                           |
| Number of features          | Small         | Larger             | Very large                    |
| Memory usage                | Low           | Higher             | Very high                     |

The more n-grams you add, the better the model understands phrases — but the table gets much bigger and slower to work with. In practice, **unigrams + bigrams** is the most common sweet spot.

---

## Summary of All the Ways to Turn Words Into Numbers

| Method           | Style          | What the numbers look like                                                             |
| ---------------- | -------------- | -------------------------------------------------------------------------------------- |
| One-Hot Encoding | Old / simple   | Mostly zeros — one 1 per word                                                          |
| Bag of Words     | Old / simple   | Mostly zeros — word counts                                                             |
| TF-IDF           | Old / simple   | Mostly zeros — weighted word counts                                                    |
| Word2Vec         | Modern / smart | Compact — similar words, similar numbers                                               |
| GloVe            | Modern / smart | Compact — trained on global word patterns                                              |
| FastText         | Modern / smart | Compact — also handles parts of words                                                  |
| BERT             | Modern / smart | Context-aware — the same word gets different numbers depending on the sentence it's in |

> **Note:** Word2Vec, GloVe, FastText, and BERT are modern deep learning techniques and will be covered in detail in the **Deep Learning repository**.
