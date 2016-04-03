# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from gensim.models import word2vec
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

#Text pre-processing for word2vec training

def text_to_wordlist(text, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    text = BeautifulSoup(text).get_text()
    #
    # 2. Remove non-letters
    text = re.sub("[\W]"," ", text)
    #
    # 3. Convert words to lower case and split them
    words = text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download("punkt")

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def text_to_sentences(text, tokenizer, remove_stopwords=False ):
    # Function to split a text into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(text_to_wordlist(raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


#word2vec training or loaing (to save time)
def load_model(sentences, train = False, num_features = 200, min_word_count = 10, num_workers = 4, context = 10, downsampling = 1e-3):
        # Set values for various parameters

    min_word_count = min_word_count   # Minimum word count
    num_workers = num_workers      # Number of threads to run in parallel
    context = context          # Context window size
    downsampling = downsampling   # Downsample setting for frequent words


    if train:
        # Import the built-in logging module and configure it so that Word2Vec
        # creates nice output messages

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



        # Initialize and train the model (this will take some time)

        print "Training model..."
        model = word2vec.Word2Vec(
            sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling
        )

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = "200features_10minwords_10context"
        model.save(model_name)
    else:
        model = Word2Vec.load("200features_10minwords_10context")

    return model

# Functions to go from text to vectors
def makeFeatureVec(list_of_words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in list_of_words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(text_list, model, num_features):
    # Given a set of texts (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    textFeatureVecs = np.zeros((len(text_list),num_features),dtype="float32")
    #
    # Loop through the reviews
    for list_of_words in text_list:
        # Print a status message every 1000th review
        """
        if counter%1000. == 0.:
            print "Review %d of %d" % (counter, len(text_list))
        """
        # Call the function (defined above) that makes average feature vectors
        textFeatureVecs[counter] = makeFeatureVec(list_of_words, model, num_features)

        # Increment the counter
        counter = counter + 1.
    return textFeatureVecs

#From text to bag of centroids
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

#How to compute cosines and deal with missing data while doing it
def compute_cosines(centroids_dic, X):
    bag = []
    for idx, row in X.iterrows():

        condition = (np.isnan(centroids_dic[row[0]]).any() or np.isnan(centroids_dic[row[1]]).any())
        if condition:
            bag.append(0)
        else:
            bag.append(cosine(centroids_dic[row[0]], centroids_dic[row[1]]))

    return np.array(bag)

####Jaccard####
def jaccard(a, b):
    """
    Adapted Jaccard for list of words
    """
    if len(a) + len(b) < 1:
        return 0
    else :
        c = set(a).intersection(set(b))
        return float(len(c)) / (len(set(a)) + len(set(b)) - len(set(c)))

def compute_jaccard(list_of_words_dic, X):
    """
    Applying Jaccard to an array of data
    """
    bag=[jaccard(list_of_words_dic[x[0]], list_of_words_dic[x[1]]) for x in X]
    return np.array(bag)

def create_nlp_features(
X, centroid_title, centroid_abstract, bag_centroids_abstract,
w_l_title, w_l_abstract):
    """
    A consolidating function computing all NLP features at once
    """
    X_ = X.copy()

    X_['Jaccard_title'] = compute_jaccard(w_l_title, X.values)
    X_['Jaccard_abstract'] = compute_jaccard(w_l_abstract, X.values)
    X_['CosineD_title_centroid'] = compute_cosines(centroid_title,X)
    X_['CosineD_abstract_centroid'] = compute_cosines(centroid_abstract,X)
    X_['CosineD_abstract_bag_centroids'] = compute_cosines(bag_centroids_abstract,X)


    return X_
