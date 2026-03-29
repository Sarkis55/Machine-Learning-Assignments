# CA03---Decision-Tree_Census_Data

By Sarkis Shil-Gevorkyan and Shahzeb Ather

## Overview
This assignment implements a Decision Tree Classifier to predict income levels based on census data. The model determines whether an individual's income exceeds $50K/year (represented by the target variable y) using various demographic and socioeconomic features.
  - Dataset
  - Key Features
  - Technologies Used
  - Implementation Steps
  - Model Evaluation

## Objective
The objective of this project is to build, evaluate, and visualize a Decision Tree model. It involves data preprocessing, handling categorical variables through discretization, and hyperparameter tuning to find the "best tree" for accurate classification.

### Dataset
- The dataset is retrieved from a public GitHub repository and contains 48,842 records and 11 columns.
     - Source: https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true
- Structure: The data includes a flag column to distinguish between training (train) and testing (test) sets.

### Key Features
 - ```hours_per_week_bin```: Binned hours worked per week.
 - ```occupation_bin```: Categorized occupation levels.
 - ```msr_bin```: Marital status categories.
 - ```capital_gl_bin```: Capital gains and losses categories.
 - ```race_sex_bin```: Combined race and sex categories.
 - ```education_num_bin```: Binned years of education.
 - ```education_bin```: Categorized education levels (e.g., Bachelors, Masters).
 - ```workclass_bin```: Income-based workclass categories.
 - ```age_bin```: Binned age ranges.

### Technologies Used
- Python
- Pandas & NumPy: For data manipulation and analysis.
- Scikit-Learn:
   - DecisionTreeClassifier: For model building.
   - LabelEncoder: For feature discretization.
   - classification_report, confusion_matrix: For performance evaluation.
- Matplotlib & Seaborn: For data visualization.

### Implementation Steps
- Data Loading & DQA: Loading the dataset and performing initial Data Quality Assessment (checking for nulls, value counts, etc.).
- Preprocessing: Using LabelEncoder to convert categorical object columns into integer representations for machine learning.
- Data Splitting: Separating the dataset into training and testing sets based on the internal flag column.
- Model Training: Implementing the DecisionTreeClassifier.
- Hyperparameter Tuning: Analyzing performance across various hyperparameter settings to optimize the model.
- Visualization: Using ```plot_tree``` to visualize the final decision-making process of the model.

### Model Evaluation
The notebook includes detailed performance metrics:
  - Confusion Matrix: To track true positives, false positives, etc.
  - Classification Report: Including Precision, Recall, and F1-score.
