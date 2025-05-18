This folder contains the files required for Part 2 of the assignment, involving building a search engine application. Note that this code works for both Python 2 and Python 3.

The following files have been attached:
- Team_5_WarmUp.pdf has the answers to the warm up part
Files from Part 1 of the assignment:
- sentenceSegmentation.py
- tokenization.py
- stopwordRemoval.py
- inflectionReduction.py

Common Files:
- evaluation.py => Calculate evaluation metrics
- cranfield folder => has the cranfield dataset based on which the models were evaluated

- output => empty folder to store all the output files

VSM Model Files:
- informationRetrieval.py => has indexing (using TF-IDF) and ranking part for the VSM Model
- main.py => runs this base model and returns the evaluation metrics based on argument passed

Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 

Improved model 1:
- information_Retrieval_1.py => uses BM25 ranking
- main_1.py 

Usage: main_1.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] [-ir_method IR METHOD (vsm|bm25)]

Improved model 2:
- information_retrieval_2.py => Uses Hybrid ranking (BM25 + LSA) with Query Expansion.
- main_2.py 

Usage: python main_2.py [-custom] [-dataset DATASET_FOLDER] [-out_folder OUTPUT_FOLDER]
               [-segmenter SEGMENTER_TYPE] [-tokenizer TOKENIZER_TYPE]
               -w2v_model_path WORD2VEC_MODEL_PATH


Improved model 3:
- information_retrieval_3.py => Uses Hybrid ranking (BM25 + LSA) with a DPR reranking along with Query Expansion.
- main_3.py 

Usage: python main_3.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
        [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)]
        [-grid_search] [-use_dpr] [-dpr_top_k TOP_K] 
        -w2v_model_path WORD2VEC_MODEL_PATH

models folder should have GoogleNews-vectors-negative300.bin which is used for query expansion (Word2vec model)


NOTE: DOWNLOAD GoogleNews-vectors-negative300.bin and add it to models folder to run improved model 2 and 3