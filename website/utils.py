import re
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
import os
corpus = []
MODEL_PATH = "Downloads/myenv/venv/word2vec_model"
def update_corpus(new_text):
    global corpus
    cleaned_text = clean_text(new_text)
    tokens = preprocess_text(cleaned_text)
    corpus.extend(tokens)
    model = Word2Vec(sentences=[corpus], vector_size=100, window=5, min_count=1, workers=4)
    model.train([corpus], total_examples=len(corpus), epochs=10)
    model.save(MODEL_PATH)

def add_record_to_corpus(new_record):
    update_corpus(new_record.content)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

def preprocess_text(record: str) -> list:
    tokens = word_tokenize(record.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def visualize_embeddings(embeddings, words):
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(words)-1)
    embedding_vector = np.array([embeddings[word] for word in words])
    two_d_embeddings = tsne.fit_transform(embedding_vector)

    plt.figure(figsize=(7, 7))
    for i, word in enumerate(words):
        x, y = two_d_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords="offset points", ha='right', va="bottom")
    plt.show()


