# final.py - Fake News Detection using NLP and ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Hide sklearn warnings

# 1. Load Dataset
df = pd.read_csv("fake_or_real_news.csv")
df.dropna(inplace=True)

# 2. Prepare Data
X = df['text']
y = df['label']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

hash_vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False)
hash_train = hash_vectorizer.transform(X_train)
hash_test = hash_vectorizer.transform(X_test)

# 5. Model Training and Evaluation

# Naive Bayes with TF-IDF
nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
pred_nb = nb.predict(tfidf_test)
print("Naive Bayes (TF-IDF) accuracy:  ", round(accuracy_score(y_test, pred_nb), 3))
print("Confusion matrix, without normalization")
print(confusion_matrix(y_test, pred_nb))

# Naive Bayes with CountVectorizer
nb2 = MultinomialNB()
nb2.fit(count_train, y_train)
pred_nb2 = nb2.predict(count_test)
print("Naive Bayes (CountVectorizer) accuracy:  ", round(accuracy_score(y_test, pred_nb2), 3))
print("Confusion matrix, without normalization")
print(confusion_matrix(y_test, pred_nb2))

# Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
pred_pac = pac.predict(tfidf_test)
print("Passive Aggressive (TF-IDF) accuracy:  ", round(accuracy_score(y_test, pred_pac), 3))
print("Confusion matrix, without normalization")
print(confusion_matrix(y_test, pred_pac))

# Naive Bayes with HashingVectorizer
nb3 = MultinomialNB()
nb3.fit(hash_train, y_train)
pred_nb3 = nb3.predict(hash_test)
print("Naive Bayes (HashingVectorizer) accuracy:  ", round(accuracy_score(y_test, pred_nb3), 3))
print("Confusion matrix, without normalization")
print(confusion_matrix(y_test, pred_nb3))
