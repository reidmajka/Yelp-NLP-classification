# Providing Insights for Business Development

![crowdsourced stock imagery](readme-images/readme-header.png)

*Using data derived from social media companies that empower crowd-sourced information and insights, is it possible to find trends in texts that indicate a need for business improvement? Can we develop text classification models that can unearth trends otherwise manually compiled from individual posts and reviews? Utilizing review and business data from Yelp, we are looking to train a language model on businesses that are both open and have closed, apply to currently running businesses, and focus on falsely-classified closed businesses to see if there are trends in the reviews that hint at upcoming issues.*

## Background & Objectives
The power of crowdsourced information and reviews can be utilized to help businesses thrive in the communities they choose to set themselves in. Popular platforms like Yelp, Twitter, Foursquare etc. allow the community to provide their input on establishments once limited to word-of-mouth and surveys conducted by either the businesses themselves, or community organizations - both of which inherently will result in bias, lack of extensive data, and actionable insights. While the internet introduces nuances with anonymization and noise, large datasets can help indicate trends from community input that may help businesses change structure in order to avoid a shutdown.

### Objective: 
*  create models to identify reviews that are associated with companies that have since gone out of business
*  retrieve business and review data for a given region defined as in scope for a Business Development consulting firm
*  identify characteristics of businesses in that area that are succeeding and apply to similar companies at risk of default

NOTE: the project was originally planned on different sources of data, including: Twitter, Meta (Facebook/Instagram), and Yelp. Due to recent changes in the open-source community, this project was not able to move forward with Twitter data, as the new API pricing structure was not feasible for an educational project of this scope. Meta had similar restrictions, though this model architecture can be retro-fitted to other sources once available with time and financial resources available.

### The Data
in order to train these various Machine Learning models, open-source data provided by Yelp was utilized to train the following models:
*  NLP classification model on reviews to predict whether the business in question has gone out of business since the review (assuming reviews were not left on businesses that have already shut down)
*  K-nearest neighbors model to predict attributes of successful businesses based off the attributes of a given business that is deemed at-risk by the above model

The open-source data files contain different JSON files that are inter-linked, including more than six million reviews for over 100k businesses, all throughout the United States. 

## Modeling process
### NLP Classification Model on Yelp Reviews
The first model required training a Machine Learning classification model on reviews data, using the feature "is_open" as our target variable. This indicates, for all the business ID's in the dataset, if the business (as of July 2023) is still open. While there are a few practicality issues with the dataset, we will assume for this project that there were no external factors to a business closing outside of business performance (i.e., that the pandemic did not happen in this universe). For X-features, the reviews were used, along with the reiew rating, average business rating, and review count for the business in question for each review.

The process to determine optimal model results is as follows:
1.  the initial GridSearchCV will cover the following: two forms of text preprocessing (Stemming and Lemmatization), two forms of vectorization (Count Vectorization and Tf-Idf Vectorization), and five different ML classification models (Logistic Regression, Decision Tree, Random Forest, Multinomial Naïve Bayes, Gradient Boosting)
2.  Given the large amount of data, take a small sample `(n=1,000)` to perform a GridSearchCV on all combinations of the above Pipeline objects
  a.  the resulting data showed that the top text preprocessor was through Stemming, vectorization through CountVec, and models using Decision Tree and Gradient Boosted trees
4.  The top performing objexts from the `n=1,000` iteration were applied to a larger dataset, this time a sample of the total dataframe of `n=10,000`. The same process was followed, showing improving scores for all iterations.
5.  the final iteration used a Gradient-Boosted Trees model, on stemmed data fed through a Count Vectorizer. the final (and best) AUC score was recorded at 0.77.

6.  A test model fit was done on the entire dataset. Time restraints allowed this to be run once, with just the review text, using a Hashing Vectorizer that did not allow the token features to be identified (due to a local memory issue), and resulted in and AUC score of 0.61. For future improvement, the data and model can be fed into an external processing application like Google Colab, but for now will have to utilize the model in #5 above.

### Clustering model 
Using a KMeans clustering model and geographic data of businesses "at risk", determine where to focus efforts and determine if neighborhood characters factor into business success. This model is largely used as a reference.

## Output and Recommendations
Overall, while the NLP classification model is working in theory at 0.77 AUC, applying to unseen reviews and assessing trends in the texts provided to be not as helpful as imagined. This makes sense, in that there are likely many different factors that play into a business's demise than just the reviews provided by consumers (a big note in this use case is that 2023 closed businesses will likely include closures due to the pandemic, for example). While it certainly helps to understand the trends within community-sourced data, a more robust model would include many other features that depend on business demographics.

## Next steps
Given the resources available to complete the first pass of the project, next steps include:
*  applying connection to Yelp API to do live searches on all businesses
*  include user features to determine clintele associated with successful businesses
*  apply external datasets associated with community attributes (neighborhood businesses, residents, regional attributes etc.) to better investigate which factors attribute to business issues

### Sources
Yelp dataset: https://www.yelp.com/dataset

### Reproduction 

To obtain the dataset, download json files at above URL, and run the notebooks in following order:
1.  Yelp-NLP-preprocessing
2.  Yelp-NLP-classification
3.  Yelp-KNN-visualization
4.  Wordcloud-Notebook
