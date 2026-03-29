# Machine Learning Assignments

This repository contains a collection of machine learning assignments implemented in Python using Jupyter Notebooks. Each notebook focuses on a specific concept or technique, ranging from supervised learning models to recommendation systems and ensemble methods.

## 📂 Project Structure

The repository consists of five main notebooks:
- CA01 - House Prices EDA
- CA02 – Naive Bayes Spam Filter
- CA03 – Decision Trees
- CA04 – Ensemble Learning
- CA05 - KNN Movie Recommender System

## 📊 CA01 - Housing Prices EDA

This notebook explores a housing dataset to understand how different features relate to property prices.

Overview:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Visualization of key variables affecting price
- Correlation and pattern analysis

Goal: Identify trends and relationships in housing data to understand what factors influence prices.

## 🤖 CA02 – Classification Assignment
This project implements a Naive Bayes supervised machine learning model to classify emails as either "Spam" or "Not Spam". The classifier processes raw email text, extracts the most frequent words as features, and utilizes the GaussianNB algorithm from Scikit-Learn to perform binary classification.

Overview:
- Dictionary Creation: Processes training emails to build a dictionary of the 3000 most common alphabetical words, filtering out single-letter entries.
- Feature Extraction: Converts email content into a feature matrix where rows represent emails and columns represent the frequency of the top 3000 words.
- Automated Labeling: Automatically identifies spam emails based on file naming conventions (e.g., files starting with "spmsg").
- Performance Evaluation: Calculates an Accuracy Score by comparing predicted labels against actual test data labels.

Results:
- The model achieves a high classification accuracy (approximately 96.5% based on sample output) using the Gaussian Naive Bayes approach

## 🌳 CA03 – Decision Trees
This project utilizes a Decision Tree classifier to predict whether an individual's income exceeds $50K per year based on census data.

Overview: 
- Data Preparation: Includes loading data from GitHub, performing Data Quality Analysis (DQA), and using LabelEncoder to convert categorical text bins into a numerical format for model compatibility.
- Modeling: Implements the DecisionTreeClassifier from Scikit-Learn. The notebook explores hyperparameter tuning to identify the "best" tree by balancing complexity and performance.
- Evaluation & Visualization: Uses performance metrics like classification reports and confusion matrices to assess accuracy. It also provides a visual representation of the final decision tree to show how key features like marital status, education, and capital gains drive the classification.

Goal:
- The primary goal is to build an interpretable predictive model that accurately classifies individuals into income brackets (<=50K or >50K). It specifically aims to identify the most influential socioeconomic factors while maintaining a model that generalizes well without overfitting.

## 🌲🌲CA04 – Ensemble Learning
This project compares several high-performance Ensemble Learning algorithms to predict whether an individual's annual income exceeds $50K using census data.

Overview:
- Preprocessing Pipeline: Uses LabelEncoder to transform categorical data into a machine-readable format and splits the data into training and testing sets based on the dataset's internal flags.
- Algorithm Implementation: Trains and evaluates four major ensemble models:
- Random Forest: A bagging method using multiple decision trees.
- AdaBoost: An adaptive boosting technique that focuses on previously misclassified instances.
- Gradient Boosting: A boosting method that optimizes differentiable loss functions.
- XGBoost: An optimized distributed gradient boosting library designed for efficiency.
- Performance Tracking: For each algorithm, the notebook iterates through different quantities of "estimators" (n_estimators) to plot accuracy and AUC (Area Under the Curve) trends.

Goal:
- The primary goal is to identify which ensemble learning technique provides the highest predictive accuracy and best distinguishes between income classes (AUC score) for this specific dataset. By comparing these models side-by-side, the project aims to demonstrate the strengths and trade-offs of different ensemble strategies in a real-world classification task.

## 🎬 KNN Recommender System
This project implements a content-based recommendation engine using the K-Nearest Neighbors (KNN) algorithm to suggest movies based on their metadata and ratings.

Overview
- Data Handling: Loads and cleans a movie dataset, focusing on numerical features such as IMDB ratings and binary-encoded genres (e.g., Action, Drama, Sci-Fi).
- Feature Engineering: Prepares a specific feature vector for the target movie ("The Post") to be used as the query point for the model.
- Model Implementation: Utilizes Scikit-Learn’s NearestNeighbors unsupervised learner. It specifically employs the Euclidean Distance metric to calculate similarity between movies.
- Recommendation Logic: The model searches the feature space for the five data points with the smallest distance to the input vector and returns the titles and ratings of those movies.

Goal:
- To build a functional recommendation system that leverages mathematical distance to find similarities between items. It serves as a practical application of the KNN algorithm for solving information filtering and personalized user experience challenges.


## ⚙️ Technologies Used
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
