# BSAN-6070---CA02

# CA02 — Spam Email Detection using Naive Bayes

## Overview
This repository contains my solution for **CA02: Spam eMail Detection using the Naive Bayes Classification Algorithm**.

The goal is to:
- Build a word-frequency feature representation from email text (top **3000** most frequent words)
- Train a **Naive Bayes** classifier on labeled emails (Spam vs Not Spam)
- Predict labels on a held-out test set and report **accuracy**

## Dataset
The dataset is provided via the course (Brightspace) as a ZIP file containing two folders:
- `train-mails/` (training emails)
- `test-mails/` (test emails)

### File naming convention (provided dataset)
- Non-spam emails: `number-numbermsg<number>.txt` (example: `3-1msg1.txt`)
- Spam emails: `spmsg<number>.txt` (example: `spmsga162.txt`)


## How It Works

### 1. Dictionary Creation (`make_Dictionary`)
The model first identifies the most important words in the training set:
* Iterates through every email file in the `train-mails` folder.
* Removes non-alphabetical characters and single-letter words to reduce noise.
* Extracts the **3,000 most common words** to serve as the feature set for the classifier.

### 2. Feature Extraction (`extract_features`)
To feed text into a machine learning model, we must convert it into numbers. This is done via a **Bag of Words** approach:
* **Feature Matrix**: Creates a matrix where each row is an email and each column represents one of the 3,000 words from our dictionary.
* **Word Counting**: The function counts how many times each dictionary word appears in an email and stores that count in the matrix.
* **Labeling**: Automatically assigns a `1` for spam or `0` for ham by checking if the filename starts with `spmsg`.

### 3. Model Training & Testing
We use the **Gaussian Naive Bayes** algorithm to perform the classification:
* **Training**: The model is trained using the `features_matrix` and `labels` from the training data.
* **Prediction**: The trained model predicts whether the emails in the `test-mails` folder are spam.
* **Evaluation**: The model's predictions are compared against the actual labels to generate an accuracy score.

## Performance
The classifier achieves a high level of accuracy on the test dataset:
* **Accuracy Score**: ~96.5%.

## Installation & Usage
1.  Clone the repository.
2.  Ensure you have `numpy` and `scikit-learn` installed.
3.  Place the `train-mails` and `test-mails` folders in the same directory as the `.ipynb` file to ensure the relative paths work correctly.
4.  Run the Jupyter Notebook `CA02_NB_assignment.ipynb`.

**Important:** Do **not** rename or modify any data folders/files. The notebook is graded against the original structure.
