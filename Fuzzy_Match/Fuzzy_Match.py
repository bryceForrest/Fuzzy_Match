from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import numpy as np
import re
from collections import namedtuple
import re

class _Weight():
    def fit(self, X, y, reg=1):
        self.reg = reg
        ATA = X.T @ X
        B = (X.T @ y).todense()
        self.weight = np.linalg.solve(ATA + self.reg * np.identity(ATA.shape[0]), B)

    def transform(self, X, y=None):
        return np.asarray(X @ self.weight)

class Fuzzy_Match:
    def fit(self, corpus, stop_words = [], acr_dict = {}, 
            ngram=2, use_idf=False, max_features = None, vocabulary=None, norm='l2',
            valid_matches = None, reg=1):
        """Fit the Fuzzy_Match object to the corpus

        Args:
            corpus (pandas DataFrame):
                Documents to be searched from.
            stop_words (list):
                List of words to clean out.
            acr_dict (dictionary):
                Dictionary of known acronyms.
            ngram (int or tuple of ints):
                Length or range of lengths for n-grams.
            use_idf (bool):
                Use inverse document frequency in word embedding.
            max_features (int):
                Maximum number of embedded features.
            vocabulary (list or dictionary):
                Provide features rather than infer from document.
            norm ({‘l1’, ‘l2’} or None):
                Normalization for feature embedding.
            valid_matches (pandas DataFrame):
                Provides example document of valid matches.
            reg (int):
                Regularization term to be used if valid_matches are provided.

        Returns:
            None
        """
        self.corpus = np.array(corpus)
        self.stop_words = stop_words # STOP WORDS MAY NOT BE USEFUL SINCE THEY WILL HAVE LOW TF-IDF SCORES
        self.acr_dict = acr_dict
        try:
            iter(ngram)
            self.ngram = ngram
        except:
            self.ngram = (ngram, ngram)
        self.vectorizer = TfidfVectorizer(
            analyzer='char', preprocessor=Fuzzy_Match._clean, 
            ngram_range=self.ngram, max_features=max_features, use_idf=use_idf, norm=norm,
            vocabulary=vocabulary
        )
        
        self.vectorizer.fit(self.corpus)
        self.corpus_tf_idf = self.vectorizer.transform(self.corpus)
 
        self.valid_matches = valid_matches
        if self.valid_matches is not None:
            self.weight = _Weight()
            queries, mapped = valid_matches
            queries = self.vectorizer.transform(queries)
            mapped = self.vectorizer.transform(mapped)
            self.weight.fit(queries, mapped, reg)
            self.corpus_tf_idf = self.weight.transform(self.corpus_tf_idf)
        else:
            self.weight = None


    def search(self, query, top_n=5):
        """Search the documents

        Args:
            query (str or iterable of str):
                Search terms to match.
            top_n:
                Number of possible matches to return.

        Returns:
            Dicionary of {query : [list of top_n matches (sorted by similarity score, desc)]}
        """
        self.query = np.array([query]) if type(query) == str else np.array(query)
        self.query_tf_idf = self.vectorizer.transform(self.query)

        if self.weight is not None:
            self.query_tf_idf = self.weight.transform(self.query_tf_idf)

        if scipy.sparse.isparse(self.query_tf_idf):
            norm_q = scipy.sparse.linalg.norm(self.query_tf_idf, axis=1)
            norm_q[norm_q.nonzero()] += np.finfo(float).eps
            norm_c = scipy.sparse.linalg.norm(self.corpus_tf_idf, axis=1)  
            norm_c[norm_c.nonzero()] += np.finfo(float).eps
        else:
            norm_q = np.linalg.norm(self.query_tf_idf, axis=1, keepdims=True)
            norm_q[norm_q == 0] = np.finfo(float).eps
            norm_c = np.linalg.norm(self.corpus_tf_idf, axis=1, keepdims=True)  
            norm_c[norm_c == 0] = np.finfo(float).eps
        
        distances = (self.query_tf_idf @ self.corpus_tf_idf.T) / (norm_q.dot(norm_c.T))
        indices = distances.argsort()[:,::-1][:,:top_n]
        result = namedtuple('result', ['matches', 'scores'], defaults = [None, None])
        matches = {self.query[i] : result(self.corpus[idx], distances[i, idx]) for i, idx in enumerate(indices)}

        return matches


    @staticmethod
    def _num_replace(match):
        """*CURRENTLY UNUSED* Handles numeric values
        
        Args:
            match: numeric value to be handled

        Returns:
            fixed match
        """
        # if match == '7' or match == '8':
        #     return num2words(match)
        # match = match.group()
        # return ' ' + num2words(match) + ' '
        return ''


    @staticmethod
    def _acr_replace(string, acr_dict = {}):
        """Handles known acronyms
        
        Args:
            string: string to check
            acr_dict: known acronyms

        Returns:
            expanded acronym
        """
        for x in acr_dict.keys():
            string = string.replace(x, acr_dict[x])

        return string


    @staticmethod
    def _clean(string, stop_words=[], acr_dict={}):
        """Cleans strings, replaces stop words, calls _acr_replace
        
        Args:
            string: query to be cleaned
            stop_words: words to be removed
            acr_dict: known acronyms

        Returns:
            cleaned string        
        """
        string = string.lower()                             # normalize case
        string = Fuzzy_Match._acr_replace(string, acr_dict)

        string = " ".join(re.findall(r"(?u)\b\w\w+\b", string))
        string = re.sub(' +', ' ', string)                  # clean out extraneous spaces
        
        pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
        string = pattern.sub(' ', string)

        string = re.sub(' +', ' ', string)                  # clean out extraneous spaces
        
        return string

    def __ngram__(self,string):
        return(Fuzzy_Match._ngram(string,self.ngram))
    
    @staticmethod
    def _ngram(string,n=3):
        """Converts string to ngram
        
        Args:
            string: string to convert
            n: length of "chunks" string is broken into

        Returns:
            list of length n chunks of string
        """
        return [string[i:i+n] for i in range(len(string)-(n-1))]
