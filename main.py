import numpy as np
import csv
import os


def generate_vector_dataset(
    num_vectors=1000, dimensions=256, filename="pgvector_test_data.csv"
):
    """
    Generate a CSV file with random float vectors for pgvector performance testing.

    Args:
        num_vectors: Number of vectors to generate (default: 1000)
        dimensions: Dimension of each vector (default: 256)
        filename: Output CSV filename (default: pgvector_test_data.csv)
    """
    print(f"Generating {num_vectors} vectors with {dimensions} dimensions...")

    # Generate random vectors using numpy
    # Using normal distribution with mean=0, std=1
    vectors = np.random.normal(0, 1, (num_vectors, dimensions))

    # Normalize vectors to unit length (optional, but often helpful for vector search)
    # Comment out this line if you don't want normalized vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    # Write vectors to CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row (optional)
        header = [f"dim_{i}" for i in range(dimensions)]
        writer.writerow(header)

        # Write each vector as a row
        for i, vector in enumerate(vectors):
            writer.writerow(vector)

    file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
    print(f"Generated {filename} ({file_size:.2f} MB)")
    print(f"File contains {num_vectors} vectors with {dimensions} dimensions each")


if __name__ == "__main__":
    generate_vector_dataset()

    # Example of using custom parameters:
    # generate_vector_dataset(num_vectors=5000, dimensions=512, filename="custom_vectors.csv")
