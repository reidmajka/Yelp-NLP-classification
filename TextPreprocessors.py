#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import packages required
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords, wordnet

#create a class to stem the texts
#FOR FUTURE: move variables that are hashed into __init__ function
class TextPreprocessorSTEM(BaseEstimator, TransformerMixin):
    def __init__(self):  #, language='english'):
        #self.stemmer = SnowballStemmer(language)
        #self.stop_words = set(stopwords.words(language))
        pass
    
    def fit(self, data, y = 0):
        # this is where you would fit things like corpus specific stopwords
        # fit probable bigrams with bigram model in here
        
        # save as parameters of Text preprocessor
        
        return self        
        
    def transform(self, data, y = 0):
        fully_normalized_corpus = data.apply(self.process_doc)
        return fully_normalized_corpus
       
    def process_doc(self, doc):
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        # Tokenize the text
        doc = re.sub(text_cleaning_re, ' ', str(doc).lower()).strip()
        doc_norm = [tok.lower() for tok in word_tokenize(doc) if ((tok.isalpha()) & (tok not in stop_words)) ]    
        # Apply stemming using Snowball Stemmer
        stemmed_words = [stemmer.stem(word) for word in doc_norm if word not in stop_words]
        return " ".join(stemmed_words)


# In[ ]:


#import packages required
from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk import WordNetLemmatizer, pos_tag
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords, wordnet

#baseline class that lemmatizes text
class TextPreprocessorLEM(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        #define attributes to store if text preprocessing requires fitting from data
        pass
    
    def fit(self, data, y = 0):
        # this is where you would fit things like corpus specific stopwords
        # fit probable bigrams with bigram model in here
        
        # save as parameters of Text preprocessor
        
        return self
    
    def transform(self, data, y = 0):
        fully_normalized_corpus = data.apply(self.process_doc)
        
        return fully_normalized_corpus
        
    
    def process_doc(self, doc):

        #initialize lemmatizer
        wnl = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        
        # helper function to change nltk's part of speech tagging to a wordnet format.
        def pos_tagger(nltk_tag):
            if nltk_tag.startswith('J'):
                return wordnet.ADJ
            elif nltk_tag.startswith('V'):
                return wordnet.VERB
            elif nltk_tag.startswith('N'):
                return wordnet.NOUN
            elif nltk_tag.startswith('R'):
                return wordnet.ADV
            else:         
                return None

        #text cleaning with re:
        text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        #"@\S+" : Matches one or more non-whitespace characters after "@".
        #"https?:\S+|http?:\S:" This part of the pattern matches URLs starting with "http" or "https".
        #[^A-Za-z0-9]+: This part of the pattern matches any non-alphanumeric character.    
                   
        #remove URL's and tagged users (using the @ symbol), lower-cases the text, and gets rid of all spaces
        doc = re.sub(text_cleaning_re, ' ', str(doc).lower()).strip()
        # remove stop words and punctuations, then lower case
        doc_norm = [tok.lower() for tok in word_tokenize(doc) if ((tok.isalpha()) & (tok not in stop_words)) ]
        
        #  POS detection on the result will be important in telling Wordnet's lemmatizer how to lemmatize

        # creates list of tuples with tokens and POS tags in wordnet format
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag(doc_norm))) 
        doc_norm = [wnl.lemmatize(token, pos) for token, pos in wordnet_tagged if pos is not None]

        return " ".join(doc_norm)

