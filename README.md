# (Partitioned) (Learned) Bloom Filters Implementation for CS2241 Final Project

This project provides Python implementations of a Learned Bloom Filters (LBFs, PLBFs). It includes:

1.  An efficient standalone `BloomFilter` class (`bloom_filter.py`).
2.  Implementations of PLBF using both:
    *   Standard Dynamic Programming (O(N²k)) for threshold optimization (`PLBF` class in `fast_plbf.py`).
    *   Faster Dynamic Programming (O(Nk log N)) leveraging matrix monotonicity (`FastPLBF` class in `fast_plbf.py`).

PLBFs combine a learned model (predictor) with multiple Bloom filters to achieve better space/performance trade-offs compared to standard Bloom filters, especially when key distributions are non-uniform.

## Features

*   Implements the Partitioned Learned Bloom Filter structure.
*   Uses a provided score predictor function to guide key partitioning.
*   Offers two dynamic programming algorithms for finding optimal region thresholds:
    *   Standard DP (`PLBF`)
    *   Faster DP using SMAWK-like algorithm (`FastPLBF`)
*   Includes an efficient underlying `BloomFilter` implementation:
    *   Uses `xxhash` (xxh64) for fast hashing.
    *   Uses the `bitarray` package for memory-efficient, C-optimized bit operations.
    *   Provides type-specific `add_*`/`contains_*` methods (bytes, str, int) for performance.
*   Command-line interface (`fast_plbf.py`) for building and evaluating the filter using data from a CSV file.
*   Type-hinted codebase with MyPy configuration (`mypy.ini`).

## Setup and Installation

1.  **Install Git LFS:** The datasets used in this project are large and managed using Git Large File Storage (LFS). You must install Git LFS *before* cloning the repository.
    *   **macOS (using Homebrew):**
        ```bash
        brew install git-lfs
        ```
    *   **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt-get update
        sudo apt-get install git-lfs
        ```
    *   **Linux (Fedora/CentOS/RHEL):**
        ```bash
        sudo dnf install git-lfs
        # or sudo yum install git-lfs
        ```
    *   **Initialize Git LFS for your user:** After installing, run this command once to set up the necessary Git hooks globally:
        ```bash
        git lfs install --system
        ```

2.  **Clone the repository (if you haven't already):**
    ```bash
    git clone git@github.com:MKJM2/cs2241-latency.git
    cd cs2241-latency
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # On Windows use: .venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script for building and evaluating the PLBF is `fast_plbf.py`. It requires a dataset provided as a CSV file containing columns for `key`, `score` (between 0 and 1), and `label` (1 for positive keys, other values for negative keys).

**Command-line Arguments:**

```text
usage: fast_plbf.py [-h] --data_path DATA_PATH --N N --k K --F F [--test_split TEST_SPLIT] [--seed SEED] [--use_fast_dp]

Construct and test a Fast Partitioned Learned Bloom Filter.

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path of the dataset CSV file (needs 'key', 'score', 'label' columns)
  --N N                 N: The number of initial segments for discretization
  --k K                 k: The number of final regions (partitions)
  --F F                 F: The target overall false positive rate
  --test_split TEST_SPLIT
                        Fraction of negative samples to use for testing (default: 0.7)
  --seed SEED           Random seed for train/test split (default: 0)
  --use_fast_dp         Use the FastPLBF (O(NklogN) DP) implementation as opposed to standard PLBF (O(N^2k) DP). Default is FastPLBF.
```

**Example:**

Assuming you have a dataset `data/url_dataset.csv`:

```bash
python fast_plbf.py \
    --data_path data/url_dataset.csv \
    --N 1000 \
    --k 10 \
    --F 0.01 \
    --test_split 0.5 \
    --seed 42 \
    --use_fast_dp
```

This command will:
*   Load data from `data/url_dataset.csv`.
*   Use N=1000 initial segments for score discretization.
*   Create k=10 final regions (partitions).
*   Target an overall False Positive Rate (FPR) of F=0.01.
*   Use 50% of the negative samples for testing the final FPR (the other 50% are used for learning the negative distribution `h`).
*   Set the random seed for the train/test split to 42.
*   Use the faster DP algorithm (`FastPLBF`) for optimization (this is the default). To use the standard DP, remove the `--use_fast_dp` flag or explicitly set it if the default changes.

The script will output the construction time, verification results (checking for false negatives), the measured FPR on the test set, the estimated memory usage of the backup Bloom filters, and the calculated optimal thresholds (`t`) and FPRs per region (`f`).


## Datasets used for experiments

### URL
As per the PLBF and FastPLBF literature, we use the [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
we acquired from: 
```
```
Manu Siddhartha. Malicious urls dataset | kaggle. URL
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset [Online; accessed
22-April-2025], 2025.
```
```

## Project Structure

```
./
├── data/                   # Directory for datasets (managed by Git LFS)
│   └── malicious_phish.csv # Example dataset
├── bloom_filter.py       # Efficient Bloom Filter implementation (used by PLBF)
├── requirements.txt      # Python package dependencies
├── README.md             # This file
├── mypy.ini              # MyPy type checker configuration
├── fast_plbf.py          # Main script for PLBF construction, training, and testing
