import numpy as np
import pandas as pd
import os
import argparse
import sys
from tqdm import tqdm
import multiprocessing as mp


def generate_chunk(chunk_id, start, end, dimensions, seed=None):
    """Generate a single chunk of normalized vectors"""
    if seed is not None:
        # Use a different seed for each chunk to maintain randomness
        np.random.seed(seed + chunk_id)

    # Generate random vectors
    chunk = np.random.normal(0, 1, (end - start, dimensions))

    # Normalize in one step using broadcasting
    chunk = chunk / np.linalg.norm(chunk, axis=1)[:, np.newaxis]

    return chunk


def generate_vector_dataset_chunked(
    num_vectors,
    dimensions,
    seed=None,
    base_filename="vector_data",
    to_stdout=False,
    chunk_size=10000,
    use_parallel=True,
    num_processes=None,
    output_format="csv",
):
    """
    Generate random float vectors for pgvector performance testing in chunks,
    with optimized performance.

    Args:
        num_vectors (int): Number of vectors (rows) to generate.
        dimensions (int): Dimension of each vector.
        seed (int, optional): Random seed for reproducibility.
        base_filename (str): Base name for the output file.
        to_stdout (bool): If True, print CSV output to stdout instead of writing to a file.
        chunk_size (int): Number of rows to process per chunk.
        use_parallel (bool): Whether to use parallel processing.
        num_processes (int): Number of processes to use (default: CPU count - 1).
        output_format (str): Format to save the data ('csv', 'npy', or 'parquet').
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

    # Determine number of processes for parallel processing
    if use_parallel and not to_stdout:
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)
            num_processes = min(num_processes, total_chunks)
        print(
            f"Using {num_processes} processes for parallel chunk generation",
            file=sys.stderr,
        )

    seed_part = f"_seed{seed}" if seed is not None else ""

    if output_format == "npy" and not to_stdout:
        # For NPY format, we'll write directly to a binary file
        filename = f"{base_filename}_{num_vectors}rows_{dimensions}dim{seed_part}.npy"
        data_chunks = []

        with tqdm(
            total=total_chunks, desc="Processing chunks", file=sys.stderr
        ) as pbar:
            for chunk_id in range(total_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, num_vectors)
                chunk = generate_chunk(chunk_id, start, end, dimensions, seed)

                if chunk_id == 0:
                    # First chunk, create the file
                    np.save(filename, chunk)
                else:
                    # Append to existing file
                    with open(filename, "ab") as f:
                        np.save(f, chunk)
                pbar.update(1)

        file_size = os.path.getsize(filename) / (1024 * 1024)  # size in MB
        print(f"\nGenerated {filename} ({file_size:.2f} MB)", file=sys.stderr)

    elif to_stdout:
        # Output to stdout, process sequentially
        print(",".join(header))  # Write header

        for chunk_id in tqdm(
            range(total_chunks), desc="Processing chunks", file=sys.stderr
        ):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, num_vectors)
            chunk = generate_chunk(chunk_id, start, end, dimensions, seed)

            # Convert to DataFrame and output as CSV
            pd.DataFrame(chunk, columns=header).to_csv(
                sys.stdout, header=False, index=False
            )

        print("\nOutput to stdout complete.", file=sys.stderr)
        return None

    else:
        # CSV or Parquet output
        if output_format == "csv":
            filename = (
                f"{base_filename}_{num_vectors}rows_{dimensions}dim{seed_part}.csv"
            )
        else:  # parquet
            filename = (
                f"{base_filename}_{num_vectors}rows_{dimensions}dim{seed_part}.parquet"
            )

        if use_parallel:
            # Parallel processing for chunk generation
            pool = mp.Pool(processes=num_processes)
            results = []

            for chunk_id in range(total_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, num_vectors)
                results.append(
                    pool.apply_async(
                        generate_chunk, args=(chunk_id, start, end, dimensions, seed)
                    )
                )

            # Write out chunks as they complete
            if output_format == "csv":
                with tqdm(
                    total=total_chunks, desc="Processing chunks", file=sys.stderr
                ) as pbar:
                    # Write header first
                    pd.DataFrame(columns=header).to_csv(filename, index=False)

                    for i, result in enumerate(results):
                        chunk = result.get()
                        # Convert to DataFrame and append to CSV
                        pd.DataFrame(chunk, columns=header).to_csv(
                            filename, mode="a", header=False, index=False
                        )
                        pbar.update(1)
            else:  # parquet
                # For parquet, collect all chunks and write at once
                all_data = []
                with tqdm(
                    total=total_chunks, desc="Processing chunks", file=sys.stderr
                ) as pbar:
                    for result in results:
                        all_data.append(result.get())
                        pbar.update(1)

                    # Combine all chunks and write to parquet
                    full_data = np.vstack(all_data)
                    pd.DataFrame(full_data, columns=header).to_parquet(
                        filename, index=False
                    )

            pool.close()
            pool.join()

        else:
            # Sequential processing
            if output_format == "csv":
                # Write header first
                pd.DataFrame(columns=header).to_csv(filename, index=False)

                for chunk_id in tqdm(
                    range(total_chunks), desc="Processing chunks", file=sys.stderr
                ):
                    start = chunk_id * chunk_size
                    end = min(start + chunk_size, num_vectors)
                    chunk = generate_chunk(chunk_id, start, end, dimensions, seed)

                    # Convert to DataFrame and append to CSV
                    pd.DataFrame(chunk, columns=header).to_csv(
                        filename, mode="a", header=False, index=False
                    )
            else:  # parquet
                all_data = []
                for chunk_id in tqdm(
                    range(total_chunks), desc="Processing chunks", file=sys.stderr
                ):
                    start = chunk_id * chunk_size
                    end = min(start + chunk_size, num_vectors)
                    all_data.append(
                        generate_chunk(chunk_id, start, end, dimensions, seed)
                    )

                # Combine all chunks and write to parquet
                full_data = np.vstack(all_data)
                pd.DataFrame(full_data, columns=header).to_parquet(
                    filename, index=False
                )

        file_size = os.path.getsize(filename) / (1024 * 1024)  # size in MB
        print(f"\nGenerated {filename} ({file_size:.2f} MB)", file=sys.stderr)
        print(
            f"File contains {num_vectors} vectors with {dimensions} dimensions each",
            file=sys.stderr,
        )
        return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate vectors for vector database benchmarking with optimized performance."
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
        default="vector_data",
        help="Base name for the output file (default: vector_data)",
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
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes to use for parallel processing (default: CPU count - 1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "npy", "parquet"],
        default="csv",
        help="Output format (csv, npy, or parquet) (default: csv)",
    )

    args = parser.parse_args()

    generate_vector_dataset_chunked(
        num_vectors=args.rows,
        dimensions=args.dimensions,
        seed=args.seed,
        base_filename=args.base_filename,
        to_stdout=args.stdout,
        chunk_size=args.chunk_size,
        use_parallel=not args.no_parallel,
        num_processes=args.processes,
        output_format=args.format,
    )
