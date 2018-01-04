#
# # coding: utf-8
#
# # In[2]:
#
#
# # Tasks (Visit prediction)
# # First, since the data is quite large, when prototyping solutions it may be too time-consuming to work with all of
# # the training examples. Also, since we don’t have access to the test labels, we’ll need to simulate validation/test
# # sets of our own.
# #
#

import numpy
import os
from collections import defaultdict, Counter
from random import random
import gc
import pprint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import log, sqrt
import pickle
import itertools


def parseData(fname):
  for l in open(fname):
    yield eval(l)

def rmse(label, prediction):
    return numpy.sqrt(numpy.asarray(((numpy.asarray(prediction) - numpy.asarray(label)) ** 2).mean()))


def feature(bookASIN, reviews):

  feat = [[1, datum[0], bookASIN] for datum in reviews ]
  return feat

def label(reviews):
    return [review[2] for review in reviews]

def process_into_trainable_data(books_DC, books_Marvel):
    X_all_DC = [feature(book, books_DC[book]) for book in books_DC]
    X_all_DC = list(itertools.chain.from_iterable(X_all_DC))

    y_all_DC = [label(books_DC[book]) for book in books_DC]
    y_all_DC = list(itertools.chain.from_iterable(y_all_DC))




    X_all_Marvel = [feature(book, books_Marvel[book]) for book in books_Marvel]
    X_all_Marvel = list(itertools.chain.from_iterable(X_all_Marvel))

    y_all_Marvel = [label(books_Marvel[book]) for book in books_Marvel]
    y_all_Marvel = list(itertools.chain.from_iterable(y_all_Marvel))

    return X_all_DC, y_all_DC, X_all_Marvel, y_all_Marvel

def mse(prediction, label):
    return numpy.asarray(((numpy.asarray(prediction) - numpy.asarray(label)) ** 2).mean())


from multiprocessing import Pool, freeze_support
import os

best_lam = None
best_MSE = None


def calculate_alpha_lfm(X, y, bias_by_user, bias_by_business, user_gamma, business_gamma):
    total = 0
    num_train = len(X)

    for i in range(0, len(X)):
        user = X[i][1]
        business = X[i][2]
        user_bias = bias_by_user[user]
        business_bias = bias_by_business[business]
        rating = y[i]
        total += rating - (user_bias + business_bias + numpy.dot(business_gamma[business], user_gamma[user]))
    return total/num_train


def build_user_business_review_dicts_lfm(X, y, num_feat=20):
    gc.collect() #too many dictionaries in memory at once during loop

    reviews_by_user = defaultdict(list)
    reviews_by_business = defaultdict(list)

    user_gamma = defaultdict(numpy.ndarray)
    business_gamma = defaultdict(numpy.ndarray)

    bias_by_user = defaultdict(int)
    bias_by_business = defaultdict(int)

    for i in range(0, len(X)):
        user = X[i][1]
        business = X[i][2]

        reviews_by_user[user].append((business, y[i]))
        reviews_by_business[business].append((user, y[i]))

        bias_by_user[user] = random()
        bias_by_business[business] = random()

        scaling_factor = numpy.sqrt(5.0 / num_feat)
        business_gamma[business] = numpy.random.rand(num_feat) * scaling_factor
        user_gamma[user] = numpy.random.rand(num_feat) * scaling_factor

    return reviews_by_user, reviews_by_business, bias_by_user, bias_by_business, user_gamma, business_gamma

def single_inference(user, business, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma):
    # print (numpy.dot(business_gamma[business], user_gamma[user]))

    return alpha + bias_by_user[user] + bias_by_business[business] + numpy.dot(business_gamma[business], user_gamma[user])

def get_averages(X, y):
    allRatings = []
    userRatings = defaultdict(list)
    businessRatings = defaultdict(list)
    for x, rating in zip(X, y):
        user, business = x[1], x[2]
        allRatings.append(rating)
        userRatings[user].append(rating)
        businessRatings[business].append(rating)

    globalAverage = sum(allRatings) / len(allRatings)
    userAverage = {}
    userMode = {}
    businessAverage = {}
    businessMode = {}
    for u in userRatings:
        ratings = userRatings[u]
        numRatings = len(ratings)
        userAverage[u] = sum(ratings) / numRatings
        data = Counter(ratings)

        mostCommon = data.most_common(1)[0]
        #     print(mostCommon)

        userMode[u] = (mostCommon[0], mostCommon[1] / numRatings)
    userAverage['unknown'] = globalAverage

    for b in businessRatings:
        ratings = businessRatings[b]
        numRatings = len(ratings)
        businessAverage[b] = sum(ratings) / numRatings
        data = Counter(ratings)

        mostCommon = data.most_common(1)[0]
        businessMode[b] = (mostCommon[0], mostCommon[1] / numRatings)
    businessAverage['unknown'] = globalAverage

        # print( businessMode[b])
    return userAverage, businessAverage
def run_inference_lfm_ensemble(X, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma, userAverage, businessAverage):
    businessWeight = 0.5
    userWeight = 1 - businessWeight
    lfmWeight = 0.3
    avgWeight = 1-lfmWeight
    predictions = []

    for x in X:
        # user = int(x[1][1:])
        # business = int(x[2][1:])
        user = x[1]
        business = x[2]


        if user not in user_gamma :
            if user == 'R2PFO9VMQH06RR':
            
                x = "break"
            user = 'unknown'
        if business not in business_gamma :
            business = 'unknown'
        predict_lfm = single_inference(user, business, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma)
        predict_avg = ((userWeight * userAverage[user]) + (businessWeight * businessAverage[business]))
        # predict_final = lfmWeight*predict_lfm + avgWeight*predict_avg
        predictions.append([predict_lfm, predict_avg])
    return predictions
def run_inference_lfm(X, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma):
    predictions = []

    for x in X:
        # user = int(x[1][1:])
        # business = int(x[2][1:])
        user = x[1]
        business = x[2]
        if user not in user_gamma.keys():
            user = 'unknown'
        if business not in business_gamma.keys():
            business = 'unknown'
        # alpha + bias_by_user[user] + bias_by_business[business] + numpy.dot(business_gamma[business], user_gamma[user])
        predict = single_inference(user, business, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma)
        predictions.append(predict)
    return predictions

def stochastic_gradient_descent(X, y, lam, eta, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma):
    biases_converged = True
    total_error = 0
    for i in range(0, len(X)):
        user = X[i][1]
        business = X[i][2]
        rating = y[i]
        prediction = single_inference(user, business, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma)
        error = rating - prediction
        total_error += error ** 2
        user_bias_update = eta * (error - lam * bias_by_user[user])
        bias_by_user[user] += user_bias_update

        business_bias_update = eta * (error - lam *bias_by_business[business])
        bias_by_business[business] +=  business_bias_update

        user_gamma_update = eta * (error * business_gamma[business] - lam*user_gamma[user])
        user_gamma[user] += user_gamma_update

        business_gamma_update = eta * (error * user_gamma[user] - lam*business_gamma[business])
        business_gamma[business] += business_gamma_update

        if user_bias_update + business_bias_update + sum(user_gamma_update) + sum(business_gamma_update) > .00001:
            biases_converged = False
    training_loss = total_error/len(X)
    print(sqrt(training_loss))
    return biases_converged

def update_biases_with_avg_values(bias_by_user, bias_by_business, user_gamma, business_gamma, num_features):#in case user or business is unknown
    mean_user_gamma = numpy.zeros(num_features)
    num_users_DC_DC = 0
    mean_user_bias = 0
    for user in user_gamma.keys():
        mean_user_gamma += user_gamma[user]
        mean_user_bias += bias_by_user[user]
        num_users_DC_DC += 1
    mean_user_gamma /= num_users_DC_DC
    mean_user_bias /= num_users_DC_DC
    user_gamma['unknown'] = mean_user_gamma
    bias_by_user['unknown'] = mean_user_bias

    mean_business_gamma = numpy.zeros(num_features)
    mean_business_bias = 0
    num_businesses = 0
    for business in business_gamma.keys():
        mean_business_gamma += business_gamma[business]
        mean_business_bias += bias_by_business[business]
        num_businesses += 1
    mean_business_gamma /= num_businesses
    mean_business_bias /= num_businesses
    business_gamma['unknown'] = mean_business_gamma
    bias_by_business['unknown'] = mean_business_bias


def train_latent_factor_model(X, y, lam, eta, num_steps=1000, num_features=20):
    reviews_by_user, reviews_by_business, bias_by_user, bias_by_business, user_gamma, business_gamma = \
        build_user_business_review_dicts_lfm(X, y, num_features)
    alpha, last_alpha = (None, None)
    converged = False
    iteration = 0

    while (not converged and iteration < num_steps):
        if iteration>1:
            decayed_weight = eta#/ (1+(1/log(iteration)))
        else:
            decayed_weight = eta

        last_alpha = alpha
        alpha = calculate_alpha_lfm(X, y, bias_by_user, bias_by_business, user_gamma, business_gamma)
        # if iteration%1000 == 0:
        #     pprint.pprint(alpha)
        #     pprint.pprint(last_alpha)


        biases_converged = stochastic_gradient_descent(X, y, lam, decayed_weight, alpha,  bias_by_user, bias_by_business, user_gamma, business_gamma) \
                           and not (last_alpha is None) \
                           and (abs(alpha)- abs(last_alpha) < 0.00001)
        if biases_converged :
            converged = True
        iteration += 1
    return alpha, bias_by_user, bias_by_business, user_gamma, business_gamma


def write_MSE_lfm(lfm_params, num_steps = 10000):

    lam = lfm_params[0]
    eta = lfm_params[1]
    num_features = lfm_params[2]

    gc.collect() #clear out old dicts
    alpha, bias_by_user, bias_by_business, user_gamma, business_gamma = train_latent_factor_model(X_train_rating, y_train_rating, lam, eta, num_features=num_features)

    update_biases_with_avg_values(bias_by_user, bias_by_business, user_gamma, business_gamma, num_features)


    predictions = run_inference_lfm(X_valid_rating, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma)
    train_predictions = run_inference_lfm(X_train_rating, alpha, bias_by_user, bias_by_business, user_gamma,
                                          business_gamma)

    MSE = mse(predictions, y_valid_rating)

    train_MSE = mse(train_predictions, y_train_rating)
    cwd = os.path.abspath(os.path.curdir)
    predictions_Rating_file = os.path.join(cwd, "assignment1", "lfm_RMSE__appended.txt".format(lam))
    MSE_out = open(predictions_Rating_file, 'a')
    MSE_out.write("lambda: " + str(lam) + "\neta: " + str(eta) + "\nk: " + str(num_features) +"\ntrain error: "+ str(train_MSE) +" \nRMSE = " + str(MSE) + "\n\n")
    MSE_out.close()
    print("lambda: " + str(lam) + "\neta: " + str(eta) + "\nk: " + str(num_features) +"\ntrain error: "+ str(train_MSE) +" \nRMSE = " + str(MSE) + "\n\n")



cwd = os.path.abspath(os.path.curdir)
directory = os.path.join(cwd, "FilteredCompleteData")
filename_DC = "DC_Paperback_UserItemReviews.final"
final_filename_DC = os.path.join(directory, filename_DC)

filename_Marvel = "Marvel_Paperback_UserItemReviews.final"
final_filename_Marvel = os.path.join(directory, filename_Marvel)

# print (cwd)
print("Reading data...")
data_DC = pickle.load(open(final_filename_DC, 'rb'))
data_Marvel = pickle.load(open(final_filename_Marvel, 'rb'))
print("done")



X_all_DC, y_all_DC, X_all_Marvel, y_all_Marvel = process_into_trainable_data(data_DC, data_Marvel)


num_users_DC_DC = 0
num_books_pooled_DC = 0
num_books_unique_DC = 0
num_reviews_DC = 0
users_DC = set()
unique_books_DC = set()

for book in data_DC:
    for review in data_DC[book]:
        users_DC.add(review[0])
        unique_books_DC.add(review[1])
        num_reviews_DC +=1
    num_books_pooled_DC +=1

print ("DC Metrics")
print("Users: " + str(len(users_DC)))
print("Unique Books: " + str(len(unique_books_DC)))
print("Pooled Books: " + str(num_books_pooled_DC))
print("num_reviews: " + str(num_reviews_DC))
print("Average Review: " + str(sum(y_all_DC)/len(y_all_DC)))
print()
num_users_Marvel_Marvel = 0
num_books_pooled_Marvel = 0
num_books_unique_Marvel = 0
num_reviews_Marvel = 0
users_Marvel = set()
unique_books_Marvel = set()

for book in data_Marvel:
    for review in data_Marvel[book]:
        users_Marvel.add(review[0])
        unique_books_Marvel.add(review[1])
        num_reviews_Marvel +=1
    num_books_pooled_Marvel +=1
print ("Marvel Metrics")
print("Users: " + str(len(users_Marvel)))
print("Unique Books: " + str (len(unique_books_Marvel)))
print("Pooled Books: " + str(num_books_pooled_Marvel))
print("num_reviews: " + str(num_reviews_Marvel))
print("Average Review: " + str(sum(y_all_Marvel)/len(y_all_Marvel)))
print()
num_users = len(users_Marvel) + len(users_DC)
num_books_pooled = num_books_pooled_DC + num_books_pooled_Marvel
# num_books_unique = num_books_unique_DC + num_books_unique_Marvel
num_reviews = num_reviews_DC + num_reviews_Marvel
users = users_DC.union(users_Marvel)
unique_books = unique_books_DC.union(unique_books_Marvel)

print ("Total Metrics")
print("Users: " + str(len(users)))
print("Unique Books: " + str (len(unique_books)))
print("Pooled Books: " + str(num_books_pooled))
print("num_reviews: " + str(num_reviews))
print("Average Review: " + str(sum(y_all_Marvel + y_all_DC)/len(y_all_Marvel + y_all_DC)))
print()

for publisher in [
    "DC",
    "Marvel",
                  "Both"
                  ]:
    if publisher == "DC":
        X_all = X_all_DC
        y_all = y_all_DC
        X_all_cross_publisher = X_all_Marvel
        y_all_cross_publisher = y_all_Marvel
    elif publisher == "Marvel":
        X_all = X_all_Marvel
        y_all = y_all_Marvel
        X_all_cross_publisher = X_all_DC          
        y_all_cross_publisher = y_all_DC  

    else:
        X_all = X_all_DC + X_all_Marvel
        y_all = y_all_DC + y_all_Marvel
        X_all_cross_publisher = None
        y_all_cross_publisher = None
    # print(len(data))
    # partition_size_train = len(X_all)
    # X_all, y_all = shuffle(X_all, y_all)

    X_rating, X_out_rating, y_rating, y_out_rating = train_test_split(X_all, y_all,test_size=0.30, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_out_rating, y_out_rating,test_size=0.33, random_state=42)

    best_lam = 0.2
    best_eta = 0.005
    # best_num_features = round(min(num_books_pooled_DC, num_users_DC_DC) / (50 * 10000))#?
    best_num_features = 100
    print (best_num_features)

    cwd = os.path.abspath(os.path.curdir)
    predictions_Rating_file = os.path.join(cwd, "assignment1","predictions_Rating_lfm.txt")

    file_out = "lfm_model_{publisher}_{num_features}.lfm".format(publisher=publisher, num_features = best_num_features)

    if not os.path.isfile(file_out):
        # train model
        alpha, bias_by_user, bias_by_business, user_gamma, business_gamma = train_latent_factor_model(X_rating, y_rating, best_lam, best_eta, num_features= best_num_features)
        update_biases_with_avg_values(bias_by_user, bias_by_business, user_gamma, business_gamma, best_num_features)
        userAverage, businessAverage= get_averages(X_rating, y_rating)

        model = {'alpha': alpha,
                 'bias_by_user': bias_by_user,
                 'bias_by_business': bias_by_business,
                 'user_gamma': user_gamma,
                 'business_gamma': business_gamma,
                 'userAverage': userAverage,
                 'businessAverage': businessAverage}

        pickle.dump(model, open(file_out, "wb" ))
    else:
        model = pickle.load(open( file_out, "rb" ))

        alpha = model['alpha']
        bias_by_user = model['bias_by_user']
        bias_by_business = model['bias_by_business']
        user_gamma = model['user_gamma']
        business_gamma = model['business_gamma']
        userAverage = model['userAverage']
        businessAverage = model['businessAverage']

    #run inference
    predictions_zipped_valid = run_inference_lfm_ensemble(X_valid, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma, userAverage, businessAverage)
    predictions_zipped_test = run_inference_lfm_ensemble(X_test, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma, userAverage, businessAverage)
    predictions_zipped_train = run_inference_lfm_ensemble(X_all, alpha, bias_by_user, bias_by_business, user_gamma,
                                                         business_gamma, userAverage, businessAverage)

    lfm_predictions_train = []
    baseline_predictions_train = []
    for lfm_predict, baseline_predict in predictions_zipped_train:
        lfm_predictions_train.append(lfm_predict)
        baseline_predictions_train.append(baseline_predict)
         

    lfm_predictions_valid = []
    baseline_predictions_valid = []
    for lfm_predict, baseline_predict in predictions_zipped_valid:
        lfm_predictions_valid.append(lfm_predict)
        baseline_predictions_valid.append(baseline_predict)

    lfm_predictions_test = []                              
    baseline_predictions_test = []
    for lfm_predict, baseline_predict in predictions_zipped_test:
        lfm_predictions_test.append(lfm_predict)           
        baseline_predictions_test.append(baseline_predict) 
    
    print(publisher + " Metrics:")
    print("RMSE Train (LFM) = " + str(rmse(y_all, lfm_predictions_train)))
    print("RMSE Test (LFM) = " + str(rmse(y_test, lfm_predictions_test)))
    print("RMSE Validation (LFM) = " + str(rmse(y_valid, lfm_predictions_valid)))

    print("RMSE Train (Baseline) = " + str(rmse(y_all, baseline_predictions_train)))
    print("RMSE Test (Baseline) = " + str(rmse(y_test, baseline_predictions_test)))
    print("RMSE Validation (Baseline) = " + str(rmse(y_valid, baseline_predictions_valid)))

    if publisher is not "Both":
        X_all_cross_publisher_filtered = X_all_cross_publisher
        y_all_cross_publisher_filtered = y_all_cross_publisher
        # for i in range (0, len(X_all_cross_publisher)):
        #     datum = X_all_cross_publisher[i][1]
        #     rating = y_all_cross_publisher[i]
        #     if X_all_cross_publisher[i][1] in user_gamma:
        #         X_all_cross_publisher_filtered.append(datum)
        #         y_all_cross_publisher_filtered.append(rating)
        predictions_zipped_cross = run_inference_lfm_ensemble(X_all_cross_publisher_filtered, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma, userAverage, businessAverage)
        lfm_predictions_cross = []       
        baseline_predictions_cross = []  
        for lfm_predict, baseline_predict in predictions_zipped_cross:
            lfm_predictions_cross.append(lfm_predict)
            baseline_predictions_cross.append(baseline_predict)
        print("Number of Samples = " + str(len(X_all_cross_publisher_filtered)))
        print("RMSE Cross Test (LFM) = " + str(rmse(y_all_cross_publisher_filtered, lfm_predictions_cross)))
        print("RMSE Cross Test (Baseline) = " + str(rmse(y_all_cross_publisher_filtered, baseline_predictions_cross)))



    # # predictions = run_inference_lfm(X_to_infer, alpha, bias_by_user, bias_by_business, user_gamma, business_gamma)
    #
    #
    # #write predictions out to file
    # predictions_out = open(predictions_Rating_file, 'w')
    # predictions_out.write(header)
    # for i in range(0, len(X_to_infer)):
    #     user = X_to_infer[i][1]
    #     business = X_to_infer[i][2]
    #     predictions_out.write(user + '-' + business + ',' + str((predictions[i])) + '\n')
    # predictions_out.close()
    # print("done")
    #

#
# if __name__ == '__main__':
#
#     freeze_support()
#     lambdas = [1/lam for lam in range(1, 6)]
#     lfm_params = []
#     for eta in [0.005 ]:
#         for num_features in [1]:
#             for lam in lambdas:
#                lfm_params.append([lam, eta, num_features])
#     for params in lfm_params:
#         write_MSE_lfm(params)
#     processes = Pool(processes=len(lfm_params))
#
#
#     processes.map_async(write_MSE_lfm, lfm_params)
#     processes.close()
#     processes.join()
#
#
#
# # In[ ]:
# # #
# # #
#