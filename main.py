

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
import re


def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text


def load_data():
    # Load the reviews from file
    with open('reviews.txt', 'r') as file:
        reviews = file.readlines()

    # Preprocess the reviews
    preprocessed_reviews = [preprocess_text(review) for review in reviews]

    return preprocessed_reviews


def cluster_reviews(reviews):
    # Convert reviews to feature vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    review_vectors = vectorizer.fit_transform(reviews)

    # Normalize the feature vectors
    review_vectors = normalize(review_vectors)

    # Perform clustering using K-means
    num_clusters = 5  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(review_vectors)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    return cluster_labels


def count_rides(cluster_labels):
    # Count the number of reviews in each cluster
    ride_counts = Counter(cluster_labels)

    return ride_counts


def generate_report(ride_counts):
    print("Ride Report:")
    print("-------------")
    for ride_id, count in ride_counts.items():
        print(f"Ride {ride_id + 1}: {count} mentions")
    print()


if __name__ == '__main__':
    # Load and preprocess the reviews
    reviews = load_data()

    # Cluster the reviews
    cluster_labels = cluster_reviews(reviews)

    # Count the rides mentioned in the reviews
    ride_counts = count_rides(cluster_labels)

    # Generate and print the report
    generate_report(ride_counts)
