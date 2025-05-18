import time
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer
from itertools import product
class InformationRetrieval():
    def __init__(self,w2v_model_path):
        self.bm25 = None
        self.corpus = None
        self.docIDs = None
        self.tokenized_corpus = None
        self.lsa_matrix = None
        self.vectorizer = TfidfVectorizer(min_df=1)
        self.svd = TruncatedSVD(n_components=250)
        self.w2v = KeyedVectors.load_word2vec_format(w2v_model_path,binary=True)
        self.execution_time = 0
        self.tokenizer = TreebankWordTokenizer()
        self.best_config = None
        self.best_map = 0
        
    def flatten_document(self,doc):
        return ' '.join(word for sentence in doc for word in sentence)

    def tokenize(self,text):
        if isinstance(text, list):
            text = ' '.join(text)
        return self.tokenizer.tokenize(text)

    def expand_query(self,query_tokens,top_n=5,min_similarity=0.8):
        expanded = list(query_tokens)
        for word in query_tokens:
            if word in self.w2v.key_to_index:
                similar = self.w2v.most_similar(word, topn=top_n)
                expanded.extend([w for w, sim in similar if sim >= min_similarity])
        return expanded

    def buildIndex(self,docs,docIDs,k1=1.5,b=0.6,n_components=250):
        start_time = time.time()
        self.docIDs = docIDs
        
        flattened_docs = [self.flatten_document(doc) for doc in docs]
        self.tokenized_corpus = [self.tokenize(doc) for doc in flattened_docs]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus,k1=k1,b=b)
        
        self.svd = TruncatedSVD(n_components=n_components)
        joined_docs = [' '.join(doc) for doc in self.tokenized_corpus]
        tfidf_mat = self.vectorizer.fit_transform(joined_docs)
        self.lsa_matrix = self.svd.fit_transform(tfidf_mat)
        
        self.execution_time = time.time() - start_time
        return self.execution_time

    def rank(self,queries,top_n=5,min_similarity=0.8,alpha=0.7):
        start_time = time.time()
        doc_IDs_ordered = []
        
        for query in queries:
            query_tokens = self.tokenize(self.flatten_document(query))
            expanded_query = self.expand_query(query_tokens,top_n=top_n,min_similarity=min_similarity)
            bm25_scores = self.bm25.get_scores(expanded_query)
            non_zero_indices = np.where(bm25_scores > 0)[0]
            if len(non_zero_indices) == 0:
                doc_IDs_ordered.append([])
                continue
                
            max_bm25 = np.max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = bm25_scores/max_bm25
            
            q_str = ' '.join(expanded_query)
            q_vec = self.vectorizer.transform([q_str])
            q_lsa = self.svd.transform(q_vec)
            
            if alpha == 1.0: 
                ranked_indices = np.argsort(bm25_scores)[::-1]
            elif alpha == 0.0:  
                lsa_scores = cosine_similarity(q_lsa, self.lsa_matrix).flatten()
                ranked_indices = np.argsort(lsa_scores)[::-1]
            else:  
                lsa_scores = cosine_similarity(q_lsa, self.lsa_matrix).flatten()
                combined_scores = alpha*bm25_scores+(1-alpha)*lsa_scores
                ranked_indices = np.argsort(combined_scores)[::-1]
            
            ranked_docIDs = [self.docIDs[i] for i in ranked_indices]
            doc_IDs_ordered.append(ranked_docIDs)
        
        self.execution_time += time.time() - start_time
        return [doc_IDs_ordered, self.execution_time]
    
    