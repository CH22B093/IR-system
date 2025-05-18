from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from information_Retrieval_3 import InformationRetrieval
from evaluation import Evaluation
from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
import time

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")

class SearchEngine:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        self.informationRetriever = InformationRetrieval(self.args.w2v_model_path)
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """
        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))

        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))

        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))

        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))

        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))

        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))

        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - Preprocesses the queries and documents, stores in output folder
        - Invokes the IR system
        - Evaluates precision, recall, fscore, nDCG and MAP
        for all queries in the Cranfield dataset
        - Produces graphs of the evaluation metrics in the output folder
        """
        queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))
        query_ids = [item["query number"] for item in queries_json]
        queries = [item["query"] for item in queries_json]
        
        processedQueries = self.preprocessQueries(queries)
        
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]
        
        # Process documents
        processedDocs = self.preprocessDocs(docs)
        
        # Read relevance judgments
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))
        
        if args.grid_search:
            print("\nPerforming grid search to find optimal parameters...")
            
            def calculate_map_for_grid(doc_IDs_ordered, relevance_judgments):
                return self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, relevance_judgments, 10)
            
            best_config, best_map = self.informationRetriever.grid_search(
                processedDocs, doc_ids, processedQueries, qrels, calculate_map_for_grid
            )
            
            print(f"\nBest configuration found: {best_config}")
            print(f"Best MAP@10: {best_map:.4f}")
            
            start_time = time.time()
            doc_IDs_ordered = self.informationRetriever.rank_with_best_config(processedQueries)[0]
            ranking_time = time.time() - start_time
            print(f"\nRanking Time with best configuration: {ranking_time:.3f} seconds\n")
        else:
            start_time = time.time()
            self.informationRetriever.buildIndex(processedDocs, doc_ids)
            
            
            doc_IDs_ordered, ranking_time = self.informationRetriever.rank(processedQueries)
            print(f"\nRanking Time: {ranking_time:.3f} seconds\n")
        
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)
            MAPs.append(MAP)
            nDCGs.append(nDCG)
            
            print(f"Precision, Recall and F-score @ {k} : {precision:.4f}, {recall:.4f}, {fscore:.4f}")
            print(f"MAP, nDCG @ {k} : {MAP:.4f}, {nDCG:.4f}")
        
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset- Improved Model-2")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig(args.out_folder + "eval_plot_2.png")
        plt.close()
        
        if args.grid_search and self.informationRetriever.best_config:
            with open(args.out_folder + "best_config.json", 'w') as f:
                json.dump(self.informationRetriever.best_config, f, indent=4)

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """
        print("Enter query below")
        query = input()
        processedQuery = self.preprocessQueries([query])[0]
        
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]
        
        processedDocs = self.preprocessDocs(docs)
        
        best_config_path = args.out_folder + "best_config.json"
        try:
            with open(best_config_path, 'r') as f:
                best_config = json.load(f)
                print(f"Using saved best configuration: {best_config}")
                
                start_time = time.time()
                self.informationRetriever.buildIndex(
                    processedDocs, doc_ids, 
                    best_config['k1'], best_config['b'], best_config['n_components']
                )
                
                doc_IDs_ordered = self.informationRetriever.rank(
                    [processedQuery], 
                    best_config['top_n'], 
                    best_config['min_similarity'], 
                    best_config['alpha']
                )[0][0]
                end_time = time.time()
        except (FileNotFoundError, json.JSONDecodeError):
            print("No saved best configuration found. Using default parameters.")
            
            start_time = time.time()
            self.informationRetriever.buildIndex(processedDocs, doc_ids)
            doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0][0]
            end_time = time.time()
        
        print(f"\n[INFO] IR System (indexing + ranking) took {end_time - start_time:.3f} seconds.\n")       
        print("\nTop five document IDs:")
        for i, doc_id in enumerate(doc_IDs_ordered[:5]):
            doc_info = next((item for item in docs_json if item["id"] == doc_id), None)
            if doc_info:
                print(f"{i+1}. ID: {doc_id} - Title: {doc_info['title']}")
            else:
                print(f"{i+1}. ID: {doc_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main.py')
    
    parser.add_argument('-dataset', default = "cranfield/",
                      help = "Path to the dataset folder")
    parser.add_argument('-out_folder', default = "output/",
                      help = "Path to output folder")
    parser.add_argument('-segmenter', default = "punkt",
                      help = "Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer',  default = "ptb",
                      help = "Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action = "store_true",
                      help = "Take custom query as input")
    parser.add_argument('-w2v_model_path', required=True,
                  help = "Path to pretrained Word2Vec model (e.g., GoogleNews-vectors-negative300.bin)")
    parser.add_argument('-grid_search', action = "store_true",
                  help = "Perform grid search to find optimal parameters")
    parser.add_argument('-use_dpr', action="store_true", 
                  help="Use DPR for reranking")
    parser.add_argument('-dpr_top_k', type=int, default=20,
                  help="Number of documents to rerank with DPR")

    args = parser.parse_args()
    
    searchEngine = SearchEngine(args)
    
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
