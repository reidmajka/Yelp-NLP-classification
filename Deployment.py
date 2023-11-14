#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import sqlite3
import pickle
from scipy.sparse import hstack, csr_matrix

#packages for preprocessing class uploaded
from sklearn.base import BaseEstimator, TransformerMixin
import nltk 
import re
from nltk import WordNetLemmatizer, pos_tag # lemmatizer using WordNet, nltk's native part of speech tagging
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize # nltk's gold standard word tokenizer
from nltk.tokenize import sent_tokenize # nltk's sentence tokenizer
from nltk.corpus import stopwords, wordnet # imports WordNet and stopwords

#packages to run NLP prediction on texts
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import folium

import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud

#imported classes for text preprocessing
from TextPreprocessors import TextPreprocessorSTEM, TextPreprocessorLEM


# In[30]:


#dropdown packages
import ipywidgets as widgets
from ipywidgets import Label
from traitlets import observe
from IPython.display import display

import pandas as pd
import json
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from ipyleaflet import Map, DrawControl, Marker
from shapely.geometry import Point, Polygon


# In[3]:


#import the businesses json file to populate dropdown list
with open('data/yelp_academic_dataset_business.json', 'r', encoding='utf-8') as file:
    business_data = [json.loads(line) for line in file]
    
business_df = pd.DataFrame(business_data)
business_df.head()


# In[5]:


# List to store selected coordinates
selected_coordinates = []

def handle_draw(self, action, geo_json):
    global selected_coordinates
    coordinates = geo_json['geometry']['coordinates']
    selected_coordinates = coordinates
    label.value = f"Selected Area Coordinates: {coordinates}"

# Create an interactive map
m = Map(center=(40, -75), zoom=8)

draw_control = DrawControl()
m.add_control(draw_control)

label = Label()
draw_control.on_draw(handle_draw)
display(m, label)


# In[6]:


selected_coordinates = [(lat, lon) for [lon, lat] in selected_coordinates[0]]


# In[7]:


selected_coordinates


# In[8]:


polygon = Polygon(selected_coordinates[:-1])


# In[9]:


polygon


# In[10]:


# Function to filter DataFrame based on selected area
def filter_df_by_area(df, selected_coordinates):
    filtered_rows = []
    polygon = Polygon(selected_coordinates[:-1])  #set polygon of area
    for _, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if polygon.contains(Point(lat, lon)):
            filtered_rows.append(row)
    return pd.DataFrame(filtered_rows)


# In[11]:


filtered_df = filter_df_by_area(business_df, selected_coordinates)
filtered_df


# In[12]:


autofill_input = widgets.Text(placeholder='Type here...')
display(autofill_input)


# In[13]:


saved_input = (autofill_input.value).lower()
saved_input


# In[14]:


# Convert the 'name' and 'categories' columns to lowercase
filtered_df['name_lower'] = filtered_df['name'].str.lower()
filtered_df['categories_lower'] = filtered_df['categories'].str.lower()

# Filter rows based on the saved_input in lowercase columns
result_df = filtered_df[
    (filtered_df['name_lower'].str.contains(saved_input)) |
    (filtered_df['categories_lower'].str.contains(saved_input))
]

result_df = result_df[result_df['is_open'] == 1]

# Drop the temporary lowercase columns if needed
result_df = result_df.drop(['name_lower', 'categories_lower'], axis=1)


# In[15]:


result_df


# In[16]:


#retrieve the reviews pertaining to the dropdown field selected and save into dataframe
bus_ids = set(result_df['business_id'])

with open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as review_file:
    review_data = [json.loads(line) for line in review_file if json.loads(line)['business_id'] in bus_ids]
    
review_df = pd.DataFrame(review_data)
review_df.head()


# In[17]:


review_df.shape


# In[18]:


review_df['review_count'] = review_df['business_id'].map(
    lambda x: business_df.loc[business_df['business_id'] == x, 'review_count'].values[0])

review_df['business_rating_avg'] = review_df['business_id'].map(
    lambda x: business_df.loc[business_df['business_id'] == x, 'stars'].values[0])


# In[19]:


#run NLP model to predict businesses that aren't open
NLP_model = joblib.load('GB_model_100k_yelp.pkl')

with open('count_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file) 

other_cols = ['stars', 'business_rating_avg', 'review_count']    
NLP_array = review_df['text']

stemmer = TextPreprocessorSTEM()
stemmed_array = stemmer.fit_transform(NLP_array)

#vectorize the text data
text_vec = vectorizer.transform(stemmed_array)
text_vec = pd.DataFrame.sparse.from_spmatrix(text_vec)
X_values = pd.concat([review_df[other_cols].reset_index(drop=True), text_vec], axis=1)
X_values = csr_matrix(X_values.values)

NLP_preds = NLP_model.predict(X_values)
review_df['prediction_open'] = NLP_preds


# In[20]:


#apply the predictions to the result_df
result_preds = review_df.groupby('business_id')['prediction_open'].mean().reset_index()
result_preds = result_preds[result_preds['prediction_open'] <= 0.5]

#map values back to business_df
proxy_df = result_df[result_df['business_id'].isin(set(result_preds['business_id']))]
proxy_df['prediction_open'] = proxy_df['business_id'].map(
    lambda x: result_preds.loc[result_preds['business_id'] == x, 'prediction_open'].values[0])


# In[21]:


proxy_df


# In[ ]:





# In[31]:


# Calculate bounding box of the polygon
min_lat = min(lat for lat, lon in selected_coordinates)
max_lat = max(lat for lat, lon in selected_coordinates)
min_lon = min(lon for lat, lon in selected_coordinates)
max_lon = max(lon for lat, lon in selected_coordinates)

# Calculate the center of the bounding box
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# Calculate the zoom level based on the bounding box dimensions
bbox_width = max_lon - min_lon
bbox_height = max_lat - min_lat
max_bbox_dimension = max(bbox_width, bbox_height)
zoom = 14 - max_bbox_dimension  # Adjust the factor as needed

# Create an interactive map centered on the polygon with calculated zoom level
m2 = Map(center=(center_lat, center_lon), zoom=zoom)
#m2.add_layer(polygon_layer)

# Create Marker layers from the DataFrame coordinates
for index, row in proxy_df.iterrows():
    marker = Marker(location=(row['latitude'], row['longitude']))
    m2.add_layer(marker)

# Display the map
display(m2)


# In[ ]:





# In[ ]:




