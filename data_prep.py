import numpy as np


def prepare_data():
    # Generate example vectors and labels
    vectors = []
    labels = []

    # Generate 20 example vectors
    for _ in range(20):
        vector = np.random.rand(10)  # Example vector of length 10
        vectors.append(vector)

    # Assign labels to vectors
    for i in range(20):
        if i < 10:
            labels.append('Roller Coaster')
        else:
            labels.append('Ferris Wheel')

    # Save vectors and labels to files
    np.save('vectors.npy', vectors)
    np.save('labels.npy', labels)


if __name__ == '__main__':
    prepare_data()
