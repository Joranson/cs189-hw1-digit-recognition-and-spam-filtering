'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words

# --------- Add your own feature methods ----------
# 
keyword_pool = list(set(["pain", "private", "bank", "money", "drug", "spam", "prescription", "creative", "height",
    "featured", "differ", "other", "energy", "business", "message", "volumes", "revision", "path",
    "meter", "memo", "planning", "pleased", "record", "out", ";", "$", "#", "!", "(","[", "&", 
        "save","cheap","pills","medication","discount","stock","invest","promotion",
        "opportunity","trail","mortgage","market","customer","cost","fee","credit","sign","profit",
        "pay","buy","unlimited","warranty","limited", "click","quote","member","membership",
        "paid","refund","card","offer","asset","bonus","apply","master","website","loss"]))

idf_dic = {}

# Generates a feature vector
import math

def generate_feature_vector(text, freq):
    feature = []
    for word in keyword_pool:
        # count the word occurrences in the document
        word_count = freq[word]
        if word_count!=0:
            # calculate the term frequency
            tf = float(word_count) / len(re.findall(r'\w+', text))  ## need revise ----!!!
            # calculate the inverse document frequency
            idf = idf_dic[word]
            feature.append(tf*idf)
        else:
            feature.append(0)
    # print feature
    # raise Exception
    return feature

def inverse_document_frequency(word, filenames):
    # find the number of documents containing the word
    count = 0
    for filename in filenames:
        with open(filename) as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            # print words
            for w in words:
                if w.lower()==word:
                    count+=1
                    break
    # return the log of the inverse frequency
    return math.log(len(filenames)/float(count)) if count!=0 else 0


all_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')+ glob.glob(BASE_DIR + HAM_DIR + '*.txt')
for kw in keyword_pool:
    idf_dic[kw] = inverse_document_frequency(kw, all_filenames)

print idf_dic


# --------- Add your own features here ---------
# Make sure type is int or float

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    count = 0

    for filename in filenames:
        print count
        count+=1
        with open(filename) as f:
            text = f.read() # Read in text from file
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word.lower()] += 1
            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            # print feature_vector
            # break
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)

