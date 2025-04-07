import numpy as np
import csv
import os
import argparse
import sys
from tqdm import tqdm


def generate_vector_dataset_chunked(
    num_vectors,
    dimensions,
    seed=None,
    base_filename="pgvector_test_data",
    to_stdout=False,
    chunk_size=10000,  # Number of rows to generate per chunk
):
    """
    Generate random float vectors for pgvector performance testing in chunks,
    displaying a progress bar during the writing process.

    Vectors are generated from a normal distribution (mean=0, std=1) and normalized to unit length.

    Args:
        num_vectors (int): Number of vectors (rows) to generate.
        dimensions (int): Dimension of each vector.
        seed (int, optional): Random seed for reproducibility.
        base_filename (str): Base name for the output file.
        to_stdout (bool): If True, print CSV output to stdout instead of writing to a file.
        chunk_size (int): Number of rows to process per chunk.
    """
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}", file=sys.stderr)

    total_chunks = (num_vectors + chunk_size - 1) // chunk_size
    print(
        f"Generating {num_vectors} vectors with {dimensions} dimensions in {total_chunks} chunks...",
        file=sys.stderr,
    )

    header = [f"dim_{i}" for i in range(dimensions)]

    if to_stdout:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        # Use tqdm for progress bar (updates on stderr)
        for start in tqdm(
            range(0, num_vectors, chunk_size), file=sys.stderr, desc="Processing chunks"
        ):
            end = min(start + chunk_size, num_vectors)
            chunk = np.random.normal(0, 1, (end - start, dimensions))
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            normalized_chunk = chunk / norms
            writer.writerows(normalized_chunk)
        print(
            "\nInspection complete. No file was created as --stdout flag was provided.",
            file=sys.stderr,
        )
        return None
    else:
        seed_part = f"_seed{seed}" if seed is not None else ""
        filename = f"{base_filename}_{num_vectors}rows_{dimensions}dim{seed_part}.csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for start in tqdm(
                range(0, num_vectors, chunk_size),
                file=sys.stderr,
                desc="Processing chunks",
            ):
                end = min(start + chunk_size, num_vectors)
                chunk = np.random.normal(0, 1, (end - start, dimensions))
                norms = np.linalg.norm(chunk, axis=1, keepdims=True)
                normalized_chunk = chunk / norms
                writer.writerows(normalized_chunk)

        file_size = os.path.getsize(filename) / (1024 * 1024)  # size in MB
        print(f"\nGenerated {filename} ({file_size:.2f} MB)", file=sys.stderr)
        print(
            f"File contains {num_vectors} vectors with {dimensions} dimensions each",
            file=sys.stderr,
        )
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
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print CSV output to stdout instead of creating a file",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of rows to generate per chunk (default: 10000)",
    )

    args = parser.parse_args()

    generate_vector_dataset_chunked(
        num_vectors=args.rows,
        dimensions=args.dimensions,
        seed=args.seed,
        base_filename=args.base_filename,
        to_stdout=args.stdout,
        chunk_size=args.chunk_size,
    )
