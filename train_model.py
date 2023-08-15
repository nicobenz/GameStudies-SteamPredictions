import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


def vectorize_sentence(sentence, embedding_model):
    words = sentence.split()
    word_embeddings = [embedding_model[word] for word in words if word in embedding_model]

    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
        return sentence_embedding
    else:
        # Handle the case where none of the words are in the embeddings vocabulary
        return None


# Load pre-trained GloVe model
model_path = "/Volumes/Data/steam/models/word2vec/GoogleNews-vectors-negative300.bin"
word2vec_model = KeyedVectors.load_word2vec_format(model_path)

# Load and preprocess data
with open("/Volumes/Data/steam/finished_corpus/corpus.json", "r") as file_in:
    data = json.load(file_in)

sentences = []
labels = []
for label, sentence_list in data.items():
    sentences.extend(sentence_list)
    labels.extend([label] * len(sentence_list))

# Convert labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(sentences, y_encoded, test_size=0.2, random_state=42)

# Convert sentences to vectorized representations using pre-trained embedding

X_train_vec = [vectorize_sentence(sentence, word2vec_model) for sentence in X_train]
X_test_vec = [vectorize_sentence(sentence, word2vec_model) for sentence in X_test]

# Train a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vec, y_train)

# Predict on the test data
y_pred = naive_bayes_classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
