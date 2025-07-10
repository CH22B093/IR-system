# Add your import statements here
import re
import json
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
nltk.download('punkt')
class Tokenization():
    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        # Tokenization using regex to split words
        tokenizedText = [re.findall(r'\b\w+\b',sentence) for sentence in text]
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        # Initializing Penn Treebank Tokenizer
        treebank_tokenizer = TreebankWordTokenizer()

		# Perform word tokenization using the Penn Treebank Tokenizer
        tokenizedText = [treebank_tokenizer.tokenize(sentence) for sentence in text]
        return tokenizedText


