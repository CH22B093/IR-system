from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

class InformationRetrieval():
    def __init__(self, method="vsm"):
        """
        Initialize the Information Retrieval system with the desired method.
        :param method: Choose either "vsm" or "bm25" for VSM or BM25 retrieval methods.
        """
        self.method = method
        self.execution_time = 0
        if self.method == "vsm":
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = None
            self.docIDs = None
        elif self.method == "bm25":
            self.bm25 = None
            self.tokenized_corpus = None
            self.docIDs = None
        else:
            raise ValueError("Invalid method! Use 'vsm' or 'bm25'.")

    def flatten_document(self, doc):
        """Flatten a document into a single string."""
        return ' '.join(word for sentence in doc for word in sentence)

    def buildIndex(self, docs, docIDs):
        """
        Build the index based on the specified method (VSM or BM25).
        :param docs: List of documents to index
        :param docIDs: List of document IDs
        """
        self.docIDs = docIDs
        self.execution_time = 0
        start_time = time.time()

        if self.method == "vsm":
            # Flatten each document into a string
            corpus = [self.flatten_document(doc) for doc in docs]
            self.doc_vectors = self.vectorizer.fit_transform(corpus)
        
        elif self.method == "bm25":
            # Tokenize the corpus (split documents into tokens)
            self.tokenized_corpus = [self.flatten_document(doc).split() for doc in docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.execution_time = time.time() - start_time

    def rank(self, queries):
        """
        Rank the documents based on the specified method (VSM or BM25).
        :param queries: List of query terms
        :return: List of ranked document IDs
        """
        start_time = time.time()
        doc_IDs_ordered = []

        if self.method == "vsm":
            for query in queries:
                query_str = self.flatten_document(query)
                query_vector = self.vectorizer.transform([query_str])

                # Computing cosine similarities
                similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
                ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order of relevance
                ranked_docIDs = [self.docIDs[i] for i in ranked_indices]
                doc_IDs_ordered.append(ranked_docIDs)

        elif self.method == "bm25":
            for query in queries:
                query_text = self.flatten_document(query)
                query_tokens = query_text.split()

                # Get BM25 scores for each document with respect to the query
                scores = self.bm25.get_scores(query_tokens)

                # Rank the documents based on BM25 scores
                ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order of relevance
                ranked_docIDs = [self.docIDs[i] for i in ranked_indices]
                doc_IDs_ordered.append(ranked_docIDs)

        self.execution_time += time.time() - start_time
        return doc_IDs_ordered
