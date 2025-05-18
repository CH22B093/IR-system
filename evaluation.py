import numpy as np
import math

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered,query_id,true_doc_IDs,k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        if k <= 0 or not query_doc_IDs_ordered:
            return 0.0
        
        # Considering only the top k documents
        top_k_docs = query_doc_IDs_ordered[:k]
        
        # Counting number of relevant documents in top k
        relevant_count = sum(1 for doc_id in top_k_docs if doc_id in true_doc_IDs)
        
        # precision calculation
        precision = relevant_count/min(k,len(query_doc_IDs_ordered))
        
        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        if not doc_IDs_ordered or not query_ids:
            return 0.0
        query_to_relevant_docs = {}
        for qrel in qrels:
            query_id = int(qrel["query_num"])
            doc_id = int(qrel["id"])
            if query_id not in query_to_relevant_docs:
                query_to_relevant_docs[query_id] = []
            query_to_relevant_docs[query_id].append(doc_id)
        
        total_precision = 0.0
        num_queries = 0
        
        for i, query_id in enumerate(query_ids):
            if query_id in query_to_relevant_docs:
                true_doc_IDs = query_to_relevant_docs[query_id]
                precision = self.queryPrecision(doc_IDs_ordered[i],query_id,true_doc_IDs,k)
                total_precision += precision
                num_queries += 1
        
        if num_queries == 0:
            return 0.0
        
        meanPrecision = total_precision/num_queries
        return meanPrecision

    
    def queryRecall(self,query_doc_IDs_ordered,query_id,true_doc_IDs,k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        if not true_doc_IDs or k <= 0 or not query_doc_IDs_ordered:
            return 0.0
        
        top_k_docs = query_doc_IDs_ordered[:k]
        
        # Counting relevant documents in top k
        relevant_count = sum(1 for doc_id in top_k_docs if doc_id in true_doc_IDs)
        
        recall = relevant_count/len(true_doc_IDs)
        
        return recall


    def meanRecall(self,doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        if not doc_IDs_ordered or not query_ids:
            return 0.0
        
        query_to_relevant_docs = {}
        for qrel in qrels:
            query_id = int(qrel["query_num"])
            doc_id = int(qrel["id"])
            if query_id not in query_to_relevant_docs:
                query_to_relevant_docs[query_id] = []
            query_to_relevant_docs[query_id].append(doc_id)
        
        total_recall = 0.0
        num_queries = 0
        
        for i, query_id in enumerate(query_ids):
            if query_id in query_to_relevant_docs:
                true_doc_IDs = query_to_relevant_docs[query_id]
                recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
                total_recall += recall
                num_queries += 1
        
        if num_queries == 0:
            return 0.0
        
        meanRecall = total_recall/num_queries
        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        precision = self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
        recall = self.queryRecall(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
        
        if precision + recall == 0:
            return 0.0
        
        fscore = 2*precision*recall/(precision + recall)
        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value
        
        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        if not doc_IDs_ordered or not query_ids:
            return 0.0
        
        query_to_relevant_docs = {}
        for qrel in qrels:
            query_id = int(qrel["query_num"])
            doc_id = int(qrel["id"])
            if query_id not in query_to_relevant_docs:
                query_to_relevant_docs[query_id] = []
            query_to_relevant_docs[query_id].append(doc_id)
        
        total_fscore = 0.0
        num_queries = 0
        
        for i,query_id in enumerate(query_ids):
            if query_id in query_to_relevant_docs:
                true_doc_IDs = query_to_relevant_docs[query_id]
                fscore = self.queryFscore(doc_IDs_ordered[i],query_id,true_doc_IDs,k)
                total_fscore += fscore
                num_queries += 1
        
        if num_queries == 0:
            return 0.0
        
        meanFscore = total_fscore/num_queries
        return meanFscore
    

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        if not query_doc_IDs_ordered or not true_doc_IDs or k <= 0:
            return 0.0
        

        top_k_docs = query_doc_IDs_ordered[:k]
        
        dcg = 0.0
        for i,doc_id in enumerate(top_k_docs):
            rel = true_doc_IDs.get(doc_id,0)
            dcg += rel/math.log2(i + 2)
        
        ideal_rels = sorted(true_doc_IDs.values(),reverse = True)[:k]
        idcg = sum(rel/math.log2(i + 2) for i,rel in enumerate(ideal_rels))
        if idcg == 0:
            return 0.0
        
        nDCG = dcg/idcg
        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        if not doc_IDs_ordered or not query_ids:
            return 0.0
        query_to_relevant_docs = {}
        for qrel in qrels:
            query_id = int(qrel["query_num"])
            doc_id = int(qrel["id"])
            pos = int(qrel["position"])
            rel = 5 - pos
            if query_id not in query_to_relevant_docs:
                query_to_relevant_docs[query_id] = {}
            query_to_relevant_docs[query_id][doc_id] = rel
        
        total_ndcg = 0.0
        num_queries = 0
        
        for i, query_id in enumerate(query_ids):
            if query_id in query_to_relevant_docs:
                true_doc_IDs = query_to_relevant_docs[query_id]
                ndcg = self.queryNDCG(doc_IDs_ordered[i],query_id,true_doc_IDs,k)
                total_ndcg += ndcg
                num_queries += 1
        
        if num_queries == 0:
            return 0.0
        
        meanNDCG = total_ndcg/num_queries
        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query.

        Parameters
        ----------
        query_doc_IDs_ordered : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        query_id : int
            The ID of the query in question
        true_doc_IDs : list
            The list of documents relevant to the query (ground truth)
        k : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        if not query_doc_IDs_ordered or not true_doc_IDs or k <= 0:
            return 0.0

        top_k_docs = query_doc_IDs_ordered[:k]
        num_relevant = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(top_k_docs):
            if doc_id in true_doc_IDs:
                num_relevant += 1
                precision_at_i = num_relevant/(i + 1)
                precision_sum += precision_at_i

        if num_relevant == 0:
            return 0.0

        avg_precision = precision_sum/num_relevant
        return avg_precision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of mean average precision (MAP) at k for all queries.

        Parameters
        ----------
        doc_IDs_ordered : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        query_ids : list
            A list of IDs of the queries
        qrels : list
            A list of dictionaries containing document-relevance judgements
        k : int
            The k value

        Returns
        -------
        float
            The mean average precision as a number between 0 and 1
        """

        if not doc_IDs_ordered or not query_ids:
            return 0.0

        query_to_relevant_docs = {}
        for qrel in qrels:
            query_id = int(qrel["query_num"])
            doc_id = int(qrel["id"])
            if query_id not in query_to_relevant_docs:
                query_to_relevant_docs[query_id] = []
            query_to_relevant_docs[query_id].append(doc_id)

        total_ap = 0.0
        num_queries = 0

        for i,query_id in enumerate(query_ids):
            if query_id in query_to_relevant_docs:
                true_doc_IDs = query_to_relevant_docs[query_id]
                ap = self.queryAveragePrecision(doc_IDs_ordered[i],query_id,true_doc_IDs,k)
                total_ap += ap
                num_queries += 1

        if num_queries == 0:
            return 0.0

        mean_ap = total_ap/num_queries
        return mean_ap
