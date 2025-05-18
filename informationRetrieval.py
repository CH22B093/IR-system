from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

class InformationRetrieval():
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None
        self.docIDs = None
        self.execution_time = 0


    def flatten_document(self, doc):
        """Flatten a document into a single string"""
        return ' '.join(word for sentence in doc for word in sentence)

    def buildIndex(self,docs,docIDs):
        """
        Builds the document index using sklearn's TfidfVectorizer.
        """
        start_time = time.time()
        self.docIDs = docIDs

        # Flattening each document into a string
        corpus = [self.flatten_document(doc) for doc in docs]

        # Using TfidfVectorizer to compute TF-IDF matrix
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self.execution_time = time.time() - start_time
        print(f"Indexing time for the IR system time: {self.execution_time:.3f} seconds")

    def rank(self,queries):
        """
        Rank the documents based on cosine similarity with each query.
        """
        start_time = time.time()
        doc_IDs_ordered = []

        for query in queries:
            query_str = self.flatten_document(query)
            query_vector = self.vectorizer.transform([query_str])

            # Computing cosine similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            ranked_indices = np.argsort(similarities)[::-1] 
            ranked_docIDs = [self.docIDs[i] for i in ranked_indices]
            doc_IDs_ordered.append(ranked_docIDs)

        self.execution_time += time.time() - start_time
        return doc_IDs_ordered
