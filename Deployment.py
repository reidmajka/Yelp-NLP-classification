#!/usr/bin/env python
# coding: utf-8

# In[4]:
import streamlit as st
import folium
from streamlit_folium import folium_static

import streamlit as st
import folium
import pandas as pd
import numpy as np
import json
import sqlite3
import pickle
from scipy.sparse import hstack, csr_matrix

st.title('Finding Businesses in Need of Development')

business_input = st.text_input("Search by Type of Business:")
if st.button("Submit"):
# Trigger action when the button is clicked
    process_input(business_input)
    
business_input = business_input.lower()

geo_input = st.text_input("Input Location Coordinates (latitude, longitude):", key="lat, lon")
[lat, lon] = geo_input.split(',')
lat = float(lat)
lon = float(lon)
selected_coordinates = [(lat+0.05, lon-0.05), (lat+0.05, lon+0.05), (lat-0.05, lon+0.05), (lat-0.05, lon-0.05), (lat+0.05, lon-0.05)]

bounds = [(lat-0.05, lon-0.05), (lat+0.05, lon+0.05)]
map_center = [lat, lon]
m = folium.Map(location=map_center, zoom_start=12)
# Add a rectangle to represent the bounds
folium.Rectangle(bounds, color='blue', fill=True, fill_opacity=0.2).add_to(m)

# Render the folium map in Streamlit
folium_static(m)

# In[1]:

#packages to run NLP prediction on texts
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud

#imported classes for text preprocessing
from TextPreprocessors import TextPreprocessorSTEM, TextPreprocessorLEM

#dropdown packages
import ipywidgets as widgets
from ipywidgets import Label
from traitlets import observe
from IPython.display import display

from ipyleaflet import Map, DrawControl, Marker
from shapely.geometry import Point, Polygon


# In[2]:


#import the businesses json file to populate dropdown list
with open('data/yelp_academic_dataset_business.json', 'r', encoding='utf-8') as file:
    business_data = [json.loads(line) for line in file]
    
business_df = pd.DataFrame(business_data)

polygon = Polygon(selected_coordinates[:-1])

# Function to filter DataFrame based on selected area
def filter_df_by_area(df, selected_coordinates):
    filtered_rows = []
    for _, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if polygon.contains(Point(lat, lon)):
            filtered_rows.append(row)
    return pd.DataFrame(filtered_rows)

filtered_df = filter_df_by_area(business_df, selected_coordinates)

# Convert the 'name' and 'categories' columns to lowercase
filtered_df['name_lower'] = filtered_df['name'].str.lower()
filtered_df['categories_lower'] = filtered_df['categories'].str.lower()

# Filter rows based on the saved_input in lowercase columns
result_df = filtered_df[
    (filtered_df['name_lower'].str.contains(business_input)) |
    (filtered_df['categories_lower'].str.contains(business_input))
]

result_df = result_df[result_df['is_open'] == 1]
# Drop the temporary lowercase columns if needed
result_df = result_df.drop(['name_lower', 'categories_lower'], axis=1)

st.write(f"Found {len(result_df)} businesses in area:")

for index, row in result_df.iterrows():
    bus_coords = [row['latitude'], row['longitude']]
    marker = folium.Marker(location=bus_coords, popup="Center")
    marker.add_to(m)
st.subheader('Businesses in Scope:')
folium_static(m)

#retrieve the reviews pertaining to the dropdown field selected and save into dataframe
bus_ids = set(result_df['business_id'])

with open('data/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as review_file:
    review_data = [json.loads(line) for line in review_file if json.loads(line)['business_id'] in bus_ids]
    
review_df = pd.DataFrame(review_data)
st.write(f"Found {len(review_df)} reviews related to businesses in area")

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
#proxy_df

map_center = [lat, lon]
m2 = folium.Map(location=map_center, zoom_start=12)
# Add a rectangle to represent the bounds
folium.Rectangle(bounds, color='blue', fill=True, fill_opacity=0.2).add_to(m2)
# Create Marker layers from the DataFrame coordinates
for index, row in proxy_df.iterrows():
    bus_coords2 = [row['latitude'], row['longitude']]
    marker = folium.Marker(location=bus_coords2, popup="Center")
    marker.add_to(m2)

st.subheader('Businesses at Risk of Closure:')
folium_static(m2)
