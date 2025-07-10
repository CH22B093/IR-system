# Add your import statements here
import json 
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load the set of stop words
stop_words = set(stopwords.words('english'))

class StopwordRemoval():
    def fromList(self, text):
        """
        Stopword Removal from Tokenized Sentences

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        stopwordsRemovedText = [[word for word in sentence if word.lower() not in stop_words] for sentence in text]
        return stopwordsRemovedText

