# NLP-Sentiment-Analysis-with-Naive-Bayes-Restaurant-Reviews

This Python script demonstrates a Natural Language Processing (NLP) task using the Naive Bayes classifier. Here's a breakdown of what the script does:

Importing Libraries: The necessary libraries such as NumPy, Matplotlib, and Pandas are imported.

Importing the Dataset: The script reads a dataset from a file named 'Restaurant_Reviews.tsv' where each review is stored along with its sentiment (positive or negative).

Cleaning the Texts: The text data is preprocessed by removing non-alphabetic characters, converting text to lowercase, removing stopwords (commonly occurring words like 'the', 'is', etc.), and stemming (reducing words to their root form).

Creating the Bag of Words Model: The Bag of Words model is created using the CountVectorizer from Scikit-learn, which converts text data into numerical format suitable for machine learning algorithms.

Splitting the Dataset: The dataset is split into training and testing sets.

Training the Naive Bayes Model: A Naive Bayes classifier is trained on the training set.

Predicting Test Set Results: The trained model is used to predict the sentiment of reviews in the test set.

Evaluating the Model: The confusion matrix and accuracy score are calculated to evaluate the performance of the model.

This script serves as a simple example of how to perform sentiment analysis using NLP techniques and a Naive Bayes classifier.
