import numpy as np


def verify_data():
    vectors = np.load('vectors.npy')
    labels = np.load('labels.npy')

    num_vectors = len(vectors)
    num_labels = len(labels)

    if num_vectors != num_labels:
        print("Error: Mismatch between the number of vectors and labels.")
        return

    print(f"Number of vectors: {num_vectors}")
    print(f"Number of labels: {num_labels}")

    for i in range(num_vectors):
        vector = vectors[i]
        label = labels[i]

        print(f"Vector {i}: {vector}")
        print(f"Label {i}: {label}")
        print()

    print("Data verification completed successfully.")


if __name__ == '__main__':
    verify_data()
