
# coding: utf-8

# In[14]:

# get_ipython().magic(u'pylab inline')


# In[15]:

import scipy.io as sio
import python_helper as ph
import random as rdm
from sklearn import svm, metrics


# In[16]:

digits = sio.loadmat("data/digit-dataset/train.mat")
N = (len(digits['train_labels']))
population_list = [_ for _ in range(N)]

def train_and_predict(training_set, validation_set, specific_collecting, CC=1, dataset=digits, train_labels="train_labels", train_objects="train_images"):
    sample_labels = []
    sample_objects = []
    specific_collecting(training_set, dataset, sample_objects, sample_labels)
    svc = svm.SVC(kernel="linear", C=CC).fit(sample_objects, sample_labels)
    
    validation_objects = []
    specific_collecting(validation_set, dataset, validation_objects)
    predict_labels=svc.predict(validation_objects)
    return svc, predict_labels

# In[17]:

##### Problem 1
validation = rdm.sample(population_list, 10000)
training = list(set(population_list) - set(validation))
true_labels = [digits["train_labels"][_][0] for _ in validation]
nums = [100,200,500,1000,2000,5000,10000]
predicted_lables =[]


# In[20]:
def digit_specific_collecting(collection, dataset, objects, objects_labels=None, train_labels="train_labels", train_objects="train_images"):
    for j in collection:
        if objects_labels is not None:
            objects_labels.append(dataset[train_labels][j][0])
        single_object = []
        for m in range(28):
            for n in range(28):
                single_object.append(dataset[train_objects][m][n][j])
        objects.append(single_object)

def run_problem1():
    for n in nums:
        training_set = rdm.sample(training, n)
        predicted_lables.append(train_and_predict(training_set, validation, digit_specific_collecting)[1])
    error_rates = [ph.benchmark(predicted_lables[i],true_labels) for i in range(len(nums))]
    plt.plot(nums, error_rates)
    plt.xlabel("Number of Training Data")
    plt.ylabel('Error Rate')
    plt.title('Number of Training Data vs. Error Rate')

# run_problem1()


# In[19]:

##### Problem 2
# def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
# def run_problem2():
#     for i in range(len(nums)):
#         cm = metrics.confusion_matrix(true_labels, predicted_lables[i])
#         plt.figure()
#         plot_confusion_matrix(cm, title = "Confusion matrix with "+str(nums[i])+" training data")
        
# run_problem2()


# In[8]:

##### Problem 3
def digit_specific_true_labels(validation_fold, dataset=digits, train_labels="train_labels"):
    return [dataset[train_labels][_][0] for _ in validation_fold]

def run_k_folds(k_folds, C=1, specific_collecting=digit_specific_collecting, specific_true_labels=digit_specific_true_labels, dataset=digits, train_labels="train_labels", train_objects="train_images", k=10):
    error_rate_k_folds = 0
    for i in range(k):
        validation_fold = k_folds[i]
        training_folds = []
        for j in range(k):
            if j!=i:
                training_folds+=k_folds[j] 
        svc, predicted_labels_k_folds = train_and_predict(training_folds, validation_fold, specific_collecting, C, dataset, train_labels, train_objects)
        true_labels_k_folds = specific_true_labels(validation_fold)
        error_rate = ph.benchmark(predicted_labels_k_folds, true_labels_k_folds)
        print "Error Rate for C=",C," is ", error_rate
        error_rate_k_folds += error_rate
    return svc, error_rate_k_folds/float(k)

def find_c_value_problem3():
    k = 10
    n = 10000
    k_folds = []
    sample_list_k_folds = rdm.sample(population_list, n)
    for i in range(k):
        fold = rdm.sample(sample_list_k_folds, n/k)
        k_folds.append(fold) 
        sample_list_k_folds = list(set(sample_list_k_folds)-set(fold))
    error_rates = []
    c_guesses = []
    for power in range(-7,-5):
        for qq in [2,4,6,8]:
            c_attempt = 10**power*qq
            svc, rate = run_k_folds(k_folds, c_attempt)
            error_rates.append(rate)
            c_guesses.append((c_attempt, svc)) 
            print rate, c_attempt
    return error_rates, c_guesses

def run_problem3():
    error_rates, c_guesses = find_c_value_problem3()
    min_error = 1
    c = 1
    for i in range(len(error_rates)):
        if error_rates[i]<min_error:
            min_error = error_rates[i]
            c= c_guesses[i][0]
    print "------------------- Selected C Value for Problem 3 is: ", c
    digit_test = sio.loadmat("data/digit-dataset/test.mat")
    test_objects = []
    sample_labels = []
    sample_objects = []
    digit_specific_collecting(population_list, digits, sample_objects, sample_labels)
    svc = svm.SVC(kernel="linear", C=c).fit(sample_objects, sample_labels)
    for i in range(10000):
        test_image = []
        for m in range(28):
            for n in range(28):
                test_image.append(digit_test["test_images"][i][n][m])
        test_objects.append(test_image)
    labels_predicted = svc.predict(test_objects)
    import csv
    csvfile = open('./test-digit.csv', 'wb')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Id","Category"])
    for i in range(len(labels_predicted)):
        csvwriter.writerow([i+1, labels_predicted[i]])
    csvfile.close()

# run_problem3()

    


# In[13]:

##### Problem 4
spams = sio.loadmat("data/spam-dataset/spam_data.mat")
num_features = 68
def spam_specific_collecting(collection, dataset, objects, objects_labels=None, train_labels="training_labels", train_objects="training_data"):
    for j in collection:
        if objects_labels is not None:
            objects_labels.append(dataset[train_labels][0][j])
        single_object = []
        for m in range(num_features):
            single_object.append(dataset[train_objects][j][m])
        objects.append(single_object)

def spam_specific_true_labels(validation_fold, dataset=spams, train_labels="training_labels"):
    return [dataset[train_labels][0][_] for _ in validation_fold]

def find_c_value_problem4():
    k = 6
    n = 5172
    k_folds = []
    sample_list_k_folds = [_ for _ in range(n)]
    for i in range(k):
        fold = rdm.sample(sample_list_k_folds, n/k)
        k_folds.append(fold) 
        sample_list_k_folds = list(set(sample_list_k_folds)-set(fold))
    error_rates = []
    c_guesses = []
    for power in range(5,8):
        for qq in [3,6,9]:
            c_attempt = 10**power*qq
            svc, rate = run_k_folds(k_folds, c_attempt, spam_specific_collecting, spam_specific_true_labels, spams, "training_labels", "training_data", k)
            error_rates.append(rate)
            c_guesses.append((c_attempt,svc)) 
            print rate, c_attempt
    return error_rates, c_guesses

def run_problem4():
    error_rates, c_guesses = find_c_value_problem4()
    min_error = 1
    c = 1
    svc = None
    for i in range(len(error_rates)):
        if error_rates[i]<min_error:
            min_error = error_rates[i]
            c, svc = c_guesses[i]
    print "------------------- Selected C Value for Problem 4 is: ", c
    spam_test = spams["test_data"]
    test_objects = []

    sample_labels = []
    sample_objects = []
    spam_specific_collecting([_ for _ in range(5172)], spams, sample_objects, sample_labels)
    svc = svm.SVC(kernel="linear", C=10000).fit(sample_objects, sample_labels)
    for i in range(5857):
        test_feature = []
        for m in range(num_features):
            test_feature.append(spam_test[i][m])
        test_objects.append(test_feature)
    labels_predicted = svc.predict(test_objects)
    import csv
    csvfile = open('./test-spam.csv', 'wb')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Id","Category"])
    for i in range(len(labels_predicted)):
        csvwriter.writerow([i+1, labels_predicted[i]])
    csvfile.close()

run_problem4()

