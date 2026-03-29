# CA04---Ensemble_Learning_Census_Data
By Sarkis Shil-Gevorkyan and Shahzeb Ather

## Overview
This project builds and evaluates multiple ensemble learning models on a census income dataset to predict whether an individual earns above or below a specified income threshold. The notebook walks through:
  - Data loading and inspection
  - Data preprocessing and feature engineering
  - Handling class imbalance
  - Training multiple ensemble models
  - Hyperparameter comparison (n_estimators)
  - Performance evaluation using Accuracy and AUC
  - Final model comparisons

**Objective:** Identify which ensemble method performs best and under what configuration.

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
   - ```RandomForestClassifier```, ```GradientBoostingClassifier``` and ```AdaboostClassifier``` For model building.
   - ```LabelEncoder``` and ```KBinsDiscretizer```: For feature discretization and binning.
   - ```accuracy_score```, ```roc_auc_score```: For performance evaluation.
- XGBoost Library for ```XGBoostClassifier``` 
- Matplotlib & Seaborn: For data visualization.

## Project Structure

### Data Loading
  - Load CSV into a Pandas DataFrame
  - Initial inspection using:
    - ```.shape```
    - ```.info()```
    - ```.describe()```
    - Missing value checks
    - Value counts

### Preprocessing & Data Quality Analysis (DQA)
  - Encoding
    - All categorical (object) columns are transformed using LabelEncoder.

  - Correlation Analysis
    - Heatmap visualization to inspect feature relationships.
    
  - Feature Engineering
    - Combined capital_gl_bin into a binary feature ```capital_gl_combined``` to address the imbalance.
    - Dropped redundant features: ```capital_gl_bin``` and ```education_bin``` (duplicate of education_num_bin)
  
  - Binning
    - Applied ```KBinsDiscretizer``` (quantile strategy) to further process ```education_num_bin``` and ```hours_per_week_bin```.
  
### Train/Test Split
  - Instead of using train_test_split, the dataset is manually split using the provided flag column:
    - Train:   flag = 1
    - Test:   flag = 0
     
### Models Implemented
The project focuses on comparing the performance of the following ensemble methods based on different values of the ```n_estimator``` hyperparameter:
  - ***Random Forest:*** A bagging technique using multiple decision trees.
  - ***Gradient Boosting:*** A boosting technique that optimizes for the residual errors of previous trees.
  - ***AdaBoost:*** An adaptive boosting technique that focuses on incorrectly classified instances.
  - ***XGBoost:*** An optimized distributed gradient boosting library designed for efficiency.

### Metrics
Models are evaluated based on two primary metrics:
  - ***Accuracy Score:*** The percentage of correct predictions.
  - ***AUC (Area Under the Curve):*** Measures the ability of the classifier to distinguish between classes.

### Model Evaluation
  - For each model, the accuracy and AUC score are recorded for each value of ```n_estimator``` options: ```[50,100,150,200,250,300,350,400,450,500]```
  - Then observe the performance of the model as the number of trees increases and determine an optimal value for ```n_estimator```
  - Best Accuracy and AUC are recorded per algorithm.

### Final Comparison Table Summarizing Results
| Metric   | Random Forest | Gradient Boost | AdaBoost | XGBoost |
| -------- | ------------- | -------------- | -------- | ------- |
| Accuracy |    0.84190    |    0.84608     | 0.84417  | 0.84331 |
| AUC      |    0.75181    |    0.75876     | 0.75769  | 0.75391 |
