# KNN Movie Recommender

A simple **content-based movie recommender** built in Python with **pandas**, **NumPy**, and **scikit-learn**.  
This project uses a **K-Nearest Neighbors (KNN)** model to find movies similar to a target movie based on selected numerical and genre-related features.

## Overview

The notebook loads a movie dataset, trains a KNN model on selected features, and returns the **5 nearest movies** to a custom movie feature vector.

In the notebook, the example target movie is **"The Post"**, represented by its IMDb rating and genre flags.

## Features Used

The recommender is trained on the following attributes:

- IMDB Rating
- Biography
- Drama
- Thriller
- Comedy
- Crime
- Mystery
- History

These features are used to measure similarity between movies.

## Dataset

The dataset is loaded directly from:

`movies_recommendation_data.csv`

Source used in the notebook:

```python
df = pd.read_csv('https://github.com/ArinB/MSBA-CA-Data/raw/main/CA05/movies_recommendation_data.csv')
```

## Requirements

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook / Google Colab

## How It Works

1. Import required libraries.
2. Load the movie dataset.
3. Inspect the dataset shape and column information.
4. Drop the `Label` column since it is not used for recommendations.
5. Train a `NearestNeighbors` model with `n_neighbors=5`.
6. Create a custom feature vector for a movie.
7. Use `kneighbors()` to find the 5 most similar movies.
8. Print the recommended movie titles and IMDb ratings.

## Model

The project uses:

```python
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(n_neighbors=5)
```

This is a **content-based filtering** approach, not collaborative filtering.  
Recommendations are based on movie attributes rather than user ratings or user behavior.

## Example: Recommending Movies Similar to "The Post"

Example custom input from the notebook:

```python
The_Post_features = {
    'IMDB Rating': 7.2,
    'Biography': 1,
    'Drama': 1,
    'Thriller': 0,
    'Comedy': 0,
    'Crime': 0,
    'Mystery': 0,
    'History': 1
}
```

The model converts this dictionary into a feature vector and returns the nearest matching movies from the dataset.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/KNN_Movie_Recommender.git
cd KNN_Movie_Recommender
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn notebook
```



#### **By Sarkis Shil-Gevorkyan and Shahzeb Aether**

