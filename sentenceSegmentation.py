# Add your import statements here
import re
import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""


		#Fill in code here
		# Defining the pattern in which the sentence is segmented (mentioned in my approach as well)
		pattern = r'(?<=[.!?])\s+' # This was not the pattern I mentioned in my approach, my pattern expected capital letters after space, but since in the doc all punctuation marks are followed by small letters, I did this change.

		# perform sentence segmentation using the proposed approach on the documents of the Cranfield dataset
		segmentedText = re.split(pattern,text)
		return segmentedText
	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		#Fill in code here
		# Performing sentence segmentation using the Punkt Tokenizer
		segmentedText = sent_tokenize(text)
		return segmentedText
