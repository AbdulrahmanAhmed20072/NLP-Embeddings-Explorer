# Word Embeddings Analysis with PCA

This notebook implements word embedding analysis and dimensionality reduction using Principal Component Analysis (PCA).

## Overview

The project includes:
- Word embeddings analysis using cosine similarity
- Country prediction based on city-country relationships
- PCA implementation from scratch
- Comparison between PCA and SVD
- Visualization of reduced word embeddings

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Pickle

## Data Files

- `capitals.txt`: Contains city-country pairs
- `word_embeddings_subset.p`: Pretrained word embeddings

## Main Functions

### Similarity Metrics
```python
# Cosine similarity between word vectors
def cosine_similarity(A, B):
    dot_product = np.dot(A,B)
    norm_A = np.sqrt(np.dot(A,A))
    norm_B = np.sqrt(np.dot(B,B))
    return dot_product / (norm_A * norm_B)

# Euclidean distance between vectors
def Euclidean(A,B):
    diff = A - B
    return np.linalg.norm(diff)
```

### Country Prediction
```python
# Predicts country2 based on the relationship: city1:country1 :: city2:country2
def get_country(city1, country1, city2, embeddings):
    # Returns predicted country and similarity score
```

### PCA Implementation
```python
def compute_pca(X, n_components):
    # Standardize data
    # Compute covariance matrix
    # Calculate eigenvalues and eigenvectors
    # Project data onto principal components
    return X_reduced
```

## Usage Example

```python
# Load data
data = pd.read_csv('capitals.txt', delimiter=' ')
word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))

# Compute similarity between words
similarity = cosine_similarity(word_embeddings["king"], word_embeddings["queen"])

# Predict country
result = get_country('Thailand', 'Bangkok', 'Athens', word_embeddings)

# Perform PCA
words = ['oil', 'gas', 'happy', 'sad', 'city', 'town']
X = get_vectors(word_embeddings, words)
X_reduced = compute_pca(X, 2)
```

## Key Features

### Word Embeddings Analysis
- Loading and processing word vectors
- Computing similarities between words
- Analyzing relationships between cities and countries

### Dimensionality Reduction
- Custom PCA implementation
- Comparison with SVD
- Visualization of reduced embeddings
- Analysis of semantic relationships in reduced space

## Visualization

The notebook includes visualization capabilities for:
- Reduced word embeddings in 2D space
- Comparison between PCA and SVD results
- Semantic clustering of related words

## Model Evaluation

Accuracy is calculated by:
- Predicting countries based on city-country relationships
- Comparing predicted countries with ground truth
- Computing overall accuracy across the dataset
