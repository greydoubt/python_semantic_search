# this is a "not hooked up to a DB" version 
# this relies on data_prep.py
# reference expected_output.txt for the vector representation

import numpy as np
from scipy.spatial.distance import cdist


def vector_search(query_vector, vectors, similarity_threshold):
    distances = cdist([query_vector], vectors, metric='cosine')[0]
    similar_indices = np.where(distances < similarity_threshold)[0]
    return similar_indices


if __name__ == '__main__':
    # Example usage
    query_vector = np.array([0.5, 0.2, 0.7, ...])  # Example query vector
    similarity_threshold = 0.8

    # Load the vectors from file
    vectors = np.load('vectors.npy')

    # Perform the vector search
    similar_indices = vector_search(query_vector, vectors, similarity_threshold)

    # Print the similar indices
    print(similar_indices)
