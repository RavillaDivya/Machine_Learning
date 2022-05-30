from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn import tree
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import requests
import json
from IPython.display import JSON
# from reviews_analysis import get_review

def get_review(lat,lon):
  #Load Dataset
  dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

  # MAY NEED THIS STATEMENT FOR NEW SYSTEMS, SHOULD MOVE THIS PROCESS TO A NEW FILE LATER.
  #Preprocess Dataset
  # nltk.download('stopwords')

  #Stemming and Lemmatization
  corpus = []
  for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    #Change the not to be a influence of word rather than removing it.
    latest = []
    for i in range(0, len(review)):
        if review[i] == 'not': 
            review[i+1] = 'un' + review[i+1]
        else:
            latest.append(review[i])
    review = latest
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

  # Creating the Bag of Words model using CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  y = dataset.iloc[:, 1].values

  #Training Classifiers
  Multi_NB_classifier = MultinomialNB(alpha=0.1)
  Multi_NB_classifier.fit(X, y)

  Bernoulli_NB_classifier = BernoulliNB(alpha=0.8)
  Bernoulli_NB_classifier.fit(X, y)

  Logistic_reg_classifier = linear_model.LogisticRegression(C=1.5)
  Logistic_reg_classifier.fit(X, y)

  Decicion_tree_classifier = tree.DecisionTreeClassifier()
  Decision_tree_classifier = Decicion_tree_classifier.fit(X,y)

  #Making a call to nearby places
  url_nearby = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude}%2C{longtitude}&radius=10000&type={type}&key=AIzaSyBTdCD9cR4rmv92IBisCQApkTYIAbYBLSU"

  find_places = url_nearby.format(latitude = lat, longtitude = lon, type = "restaurant");

  response = requests.get(url = find_places)

  pretty_json = json.loads(response.text)

  #Getting the ratings to pick restaurant
  ratings = []
  for i in range(len(pretty_json['results'])):
    company = pretty_json['results'][i]
    if 'rating' in company:
      rating = company['rating']
    else:
      rating = 0
    ratings.append(rating)

  list_of_restaurants = []
  for i in range(len(pretty_json['results'])):
    company = pretty_json['results'][i]
    if 'name' in company:
      name = company['name']
    else:
      name = 0
    list_of_restaurants.append(name)

  max_value = max(ratings)
  max_index = ratings.index(max_value)

  #Get the name of the restaurant and place id
  if pretty_json['status'] == 'OK':

    place_name = pretty_json['results'][max_index]['name']
    place_id = pretty_json['results'][max_index]['place_id']

  #Get reviews of nearest restaurant
  details_url = "https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key=AIzaSyBTdCD9cR4rmv92IBisCQApkTYIAbYBLSU"

  reviews_url = details_url.format(place_id = place_id)

  response = requests.get(url = reviews_url)

  pretty_json = json.loads(response.text)

  reviews = []
  if pretty_json['status'] == 'OK':
    for i in range(len(pretty_json['result']['reviews'])):
      review_rating = pretty_json['result']['reviews'][i]['rating']
      review_text = pretty_json['result']['reviews'][i]['text']
      reviews.append(review_text)
  else:
    print('Failed to get json response:')

  #Preprocess extracted reviews
  test_corpus = []
  for i in range(len(reviews)):
    review = re.sub('[^a-zA-Z]', ' ', reviews[i])
    review = review.lower()
    review = review.split()
    #Change the not to be a influence of word rather than removing it.
    latest = []
    for i in range(0, len(review)):
        if review[i] == 'not': 
            review[i+1] = 'un' + review[i+1]
        else:
            latest.append(review[i])
    review = latest
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)

  transformed_reviews = cv.transform(test_corpus).toarray()

  y_pred_multinomial = Multi_NB_classifier.predict(transformed_reviews)
  y_pred_bernoulli = Bernoulli_NB_classifier.predict(transformed_reviews)
  y_pred_logistic = Logistic_reg_classifier.predict(transformed_reviews)
  y_pred_decision = Decicion_tree_classifier.predict(transformed_reviews)
  predictions = []
  
  for i in range(len(reviews)):
    if (y_pred_multinomial[i] + y_pred_bernoulli[i] + y_pred_logistic[i] + y_pred_decision[i]) >= 4:
      predictions.append(1)
    else:
      predictions.append(0)

  loc = predictions.index(1)
  positive_review = reviews[loc]
  return place_name,positive_review, list_of_restaurants