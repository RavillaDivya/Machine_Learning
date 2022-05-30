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
from reviews_analysis import get_review


lat = 40.748817
lon = -73.985428

place_name,positive_review = get_review(lat,lon)

print(place_name)
positive_review = positive_review.replace('\n', '')
print(positive_review)