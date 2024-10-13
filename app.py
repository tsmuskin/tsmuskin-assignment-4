from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

def cosine_similarity_manual(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    Cosine similarity = dot_product(vec1, vec2) / (||vec1|| * ||vec2||)
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)


# Custom SVD class
class CustomSVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.U_reduced = None
        self.S_reduced = None
        self.VT_reduced = None

    def fit(self, X):
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        self.U_reduced = U[:, :self.n_components]
        self.S_reduced = S[:self.n_components]
        self.VT_reduced = VT[:self.n_components, :]

    def fit_transform(self, X):
        self.fit(X)
        return np.dot(self.U_reduced, np.diag(self.S_reduced))

    def transform(self, query_vector):
        return np.dot(query_vector, self.VT_reduced.T)

nltk.download('stopwords')
app = Flask(__name__)

# TODO: Fetch dataset, initialize vectorizer and LSA here
# Fetch 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data  # All documents from the dataset

# Initialize TF-IDF vectorizer and fit it to the documents
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=10000)
X = vectorizer.fit_transform(documents)

# Apply Truncated SVD to reduce dimensionality (LSA)
n_components = 110  # Number of dimensions to reduce to

# Reduced representation of the documents

svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X.toarray())


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 


     # Vectorize the query and apply SVD
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)

    # Compute cosine similarity between the query and all documents
    # similarities = cosine_similarity(query_reduced, X_reduced).flatten()
    similarities = []
    for doc_vec in X_reduced:
        similarity = cosine_similarity_manual(query_reduced.flatten(),doc_vec)
        similarities.append(similarity)

    similarities = np.array(similarities)


    # Get the indices of the top 5 most similar documents
    top_indices = similarities.argsort()[-5:][::-1]

    # Get the top 5 documents and their similarity scores
    top_documents = [documents[i] for i in top_indices]
    top_similarities = similarities[top_indices]

    return top_documents, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True, port=3000)
