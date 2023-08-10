from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

word2vec_model = Word2Vec(sentences, size=100, window=5, min_count=1)

X_train_vec = [word2vec_model.wv[word] for word in X_train]
X_test_vec = [word2vec_model.wv[word] for word in X_test]

