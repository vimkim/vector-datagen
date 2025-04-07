import numpy as np
import csv
import os
import argparse


def generate_vector_dataset(
    num_vectors, dimensions, seed=None, base_filename="vector_data"
):
    """
    Generate a CSV file with random float vectors for pgvector performance testing.

    Vectors are generated using a normal distribution (mean=0, std=1) and normalized to unit length.

    Args:
        num_vectors (int): Number of vectors (rows) to generate.
        dimensions (int): Dimension of each vector.
        seed (int, optional): Random seed for reproducibility.
        base_filename (str): Base name for the output file.
    """
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}")

    print(f"Generating {num_vectors} vectors with {dimensions} dimensions...")

    # Generate random vectors from a normal distribution
    vectors = np.random.normal(0, 1, (num_vectors, dimensions))

    # Normalize vectors to unit length
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    vectors = vectors / norms

    # Construct the filename with parameters engraved
    seed_part = f"_seed{seed}" if seed is not None else ""
    filename = f"{base_filename}_{num_vectors}rows_{dimensions}dim{seed_part}.csv"

    # Write vectors to CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [f"dim_{i}" for i in range(dimensions)]
        writer.writerow(header)
        writer.writerows(vectors)

    file_size = os.path.getsize(filename) / (1024 * 1024)  # size in MB
    print(f"Generated {filename} ({file_size:.2f} MB)")
    print(f"File contains {num_vectors} vectors with {dimensions} dimensions each")

    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with random float vectors for vector database benchmarking."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of vectors (rows) to generate (default: 1000)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=256,
        help="Dimension of each vector (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--base_filename",
        type=str,
        default="pgvector_test_data",
        help="Base name for the output file (default: pgvector_test_data)",
    )

    args = parser.parse_args()

    generate_vector_dataset(
        num_vectors=args.rows,
        dimensions=args.dimensions,
        seed=args.seed,
        base_filename=args.base_filename,
    )
