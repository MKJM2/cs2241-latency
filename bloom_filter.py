#!/usr/bin/env python3
import argparse
import math
import time
import xxhash  # type: ignore[import-not-found]
import struct
from bitarray import bitarray  # type: ignore[import-not-found]
from typing import (
    Callable,
    Final,
    Generic,
    List,
    Tuple,
    TypeVar,
    Optional,
)
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import sizeof

# --- Type variables and aliases ---
KeyType = TypeVar("KeyType")
# Define a specific type for the serializer function
Serializer = Callable[[KeyType], bytes]

# --- Constants ---
XXH_SEED1: Final[int] = 0
XXH_SEED2: Final[int] = 6917


# --- Generic Bloom Filter ---
class BloomFilter(Generic[KeyType]):
    """
    A generic, high-performance Bloom filter optimized for speed.

    Requires the user to provide a `serializer` function during initialization
    to convert items of `KeyType` into bytes before hashing. The core filter
    logic operates exclusively on these bytes.

    Features:
    - Generic over KeyType.
    - Requires user-provided serialization function (KeyType -> bytes).
    - xxhash (xxh64) for fast hashing.
    - bitarray package for C-optimized bit manipulation.
    - Kirsch-Mitzenmacher optimization (double hashing).
    - No runtime type checks in hot paths.
    """

    __slots__ = (
        "capacity",
        "error_rate",
        "serializer",
        "size",
        "num_hashes",
        "bit_array",
        "num_items",
        "_hasher1_intdigest",
        "_hasher2_intdigest",
    )

    # Type alias for the internal hash function signature (bytes -> int)
    _BytesHasher = Callable[[bytes], int]

    def __init__(
        self,
        capacity: int,
        error_rate: float,
        serializer: Optional[Serializer[KeyType]] = None,
    ):
        """
        Initializes the generic Bloom filter.

        Args:
            capacity: The expected number of items to be stored (n).
            error_rate: The desired false positive probability (p), e.g., 0.001.
            serializer: An optional function that takes an item of KeyType and returns bytes.
                        If None, built-in support for int, float, str, bytes is used.

        Raises:
            ValueError: If capacity is non-positive or error_rate is not in (0, 1).
            TypeError: If serializer is provided but not callable, or if an
                       unsupported key type is encountered at insertion.
        """
        if not capacity > 0:
            raise ValueError("Capacity must be positive")
        if not 0 < error_rate < 1:
            raise ValueError("Error rate must be between 0 and 1")

        if serializer is None:
            serializer = self._default_serializer
        self.serializer: Final[Serializer[KeyType]] = serializer

        self.capacity: Final[int] = capacity
        self.error_rate: Final[float] = error_rate

        size, num_hashes = self._calculate_optimal_params(capacity, error_rate)
        self.size: Final[int] = size
        self.num_hashes: Final[int] = num_hashes

        # Initialize bit array using the C-backed bitarray
        self.bit_array: bitarray = bitarray(self.size)
        self.bit_array.setall(0)

        self.num_items: int = 0

        # Initialize hashers using xxh64_intdigest for direct integer output
        # These always operate on bytes internally.
        self._hasher1_intdigest: BloomFilter._BytesHasher = (
            lambda b: xxhash.xxh64_intdigest(b, seed=XXH_SEED1)
        )
        self._hasher2_intdigest: BloomFilter._BytesHasher = (
            lambda b: xxhash.xxh64_intdigest(b, seed=XXH_SEED2)
        )

    @staticmethod
    def _default_serializer(item: KeyType) -> bytes:
        """
        Default serialization for int, float, str, bytes.
        Raises TypeError on other types.
        """
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)  # no-op
        if isinstance(item, str):
            return item.encode("utf-8")
        if isinstance(item, float): # float: 8-byte IEEE-754 big-endian
            return struct.pack(">d", item)
        if isinstance(item, int): # int: two's-complement 64-bit little-endian
            return item.to_bytes(8, byteorder="little", signed=True)
        raise TypeError(
            f"No default serializer for type {type(item).__name__}; "
            "please provide a custom serializer"
        )

    @staticmethod
    def _calculate_optimal_params(capacity: int, error_rate: float) -> Tuple[int, int]:
        """Calculates optimal size (m) and hash count (k)."""
        # m = - (n * ln(p)) / (ln(2)^2)
        m_float: float = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        size: int = max(1, int(math.ceil(m_float)))  # Ensure size is at least 1

        # k = (m / n) * ln(2)
        # Handle potential division by zero if capacity is somehow <= 0 despite check
        k_float: float = (size / capacity) * math.log(2) if capacity > 0 else 1.0
        num_hashes: int = max(1, int(math.ceil(k_float)))  # Ensure at least 1 hash

        return size, num_hashes

    def _get_indices(self, item_bytes: bytes) -> List[int]:
        """Generates k indices using double hashing with xxhash on bytes."""
        h1: int = self._hasher1_intdigest(item_bytes)
        h2: int = self._hasher2_intdigest(item_bytes)
        m: int = self.size
        # Generate k indices using Kirsch-Mitzenmacher optimization
        return [(h1 + i * h2) % m for i in range(self.num_hashes)]

    def _add_indices(self, indices: List[int]) -> None:
        """Sets the bits at the given indices in the bit array."""
        bit_arr: bitarray = self.bit_array
        for index in indices:
            bit_arr[index] = 1

    def _check_indices(self, indices: List[int]) -> bool:
        """Checks if all bits at the given indices are set."""
        bit_arr: bitarray = self.bit_array
        for index in indices:
            if not bit_arr[index]:
                return False  # Definitely not present (early exit)
        return True  # Possibly present

    # --- Public Add/Contains Methods ---

    def add(self, item: KeyType) -> None:
        """
        Adds an item to the Bloom filter.

        The item is first converted to bytes using the serializer provided
        during initialization.

        Args:
            item: The item of KeyType to add.
        """
        try:
            item_bytes: bytes = self.serializer(item)
        except Exception as e:
            raise TypeError(
                f"Failed to serialize item of type {type(item).__name__} with provided serializer: {e}"
            ) from e

        indices: List[int] = self._get_indices(item_bytes)
        self._add_indices(indices)
        self.num_items += 1

    def __contains__(self, item: KeyType) -> bool:
        """
        Checks if an item might be in the Bloom filter.

        The item is first converted to bytes using the serializer provided
        during initialization.

        Args:
            item: The item of KeyType to check.

        Returns:
            True if the item is possibly in the set (may be a false positive).
            False if the item is definitely not in the set.
        """
        try:
            item_bytes: bytes = self.serializer(item)
        except Exception as e:
            # If serialization fails, the item cannot have been added
            raise TypeError(
                f"Warning: Failed to serialize item for checking. Returning False. Error: {e}"
            ) from e

        indices: List[int] = self._get_indices(item_bytes)
        return self._check_indices(indices)

    # --- Other Public Methods ---

    def __len__(self) -> int:
        """Returns the number of items added."""
        return self.num_items

    @property
    def bit_size(self) -> int:
        """Returns the size of the underlying bit array (m)."""
        return self.size

    def __sizeof__(self) -> int:
        """Returns the size of the underlying bit array in bytes"""
        return math.ceil(self.bit_size / 8)

    def get_current_false_positive_rate(self) -> float:
        """
        Estimates the current theoretical false positive rate based on the
        number of items added (`num_items`).

        Formula: (1 - exp(-k * n / m))^k
        Where: k = num_hashes, n = num_items, m = size

        Returns:
            The estimated false positive probability (float between 0.0 and 1.0).
        """
        k: int = self.num_hashes
        n: int = self.num_items
        m: int = self.size

        if m == 0 or n == 0:  # Avoid division by zero or calculation for empty filter
            return 0.0

        try:
            exponent: float = -k * n / float(m)
            rate: float = (1.0 - math.exp(exponent)) ** k
        except (OverflowError, ValueError):
            rate = 1.0  # Theoretical rate approaches 1 if calculations fail

        return max(0.0, min(1.0, rate))  # Clamp result

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the filter."""
        # Determine serializer name if possible, otherwise show type
        serializer_name = getattr(
            self.serializer, "__name__", str(type(self.serializer))
        )
        return (
            f"{self.__class__.__name__}("
            f"capacity={self.capacity}, "
            f"error_rate={self.error_rate:.2e}, "
            f"serializer={serializer_name}, "
            f"size={self.size}, "
            f"num_hashes={self.num_hashes}, "
            f"num_items={self.num_items})"
        )


# --- Main Driver Block ---
def main() -> None:
    """Main function to test the generic Bloom Filter."""
    parser = argparse.ArgumentParser(description="Test a generic Bloom Filter.")
    parser.add_argument(
        "--data_path",
        action="store",
        dest="data_path",
        type=str,
        required=True,
        help="Path of the dataset CSV file (needs 'key', 'label' columns)",
    )
    parser.add_argument(
        "--error_rate",
        action="store",
        dest="error_rate",
        type=float,
        required=True,
        help="Target false positive rate for the Bloom Filter (e.g., 0.01)",
    )
    parser.add_argument(
        "--test_split",
        action="store",
        dest="test_split",
        type=float,
        default=0.2,
        help="Fraction of negative samples held out for testing FPR (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    results: argparse.Namespace = parser.parse_args()

    DATA_PATH: Final[str] = results.data_path
    ERROR_RATE: Final[float] = results.error_rate
    TEST_SPLIT: Final[float] = results.test_split
    SEED: Final[int] = results.seed

    # --- Data Loading and Preparation ---
    print(f"Loading data from: {DATA_PATH}")
    try:
        data: pd.DataFrame = pd.read_csv(DATA_PATH)
        required_cols: Final[set[str]] = {"key", "label"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
        data["key"] = data["key"].astype(str)  # Ensure keys are strings
        data["label"] = pd.to_numeric(data["label"], errors="coerce")
        if data["label"].isnull().any():
            raise ValueError("Non-numeric values found in 'label' column.")
        # Use original label to identify positives (==1) vs negatives (!=1)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        exit(1)
    except ValueError as e:
        print(f"Error loading or validating data: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit(1)

    positive_sample: pd.DataFrame = data.loc[(data["label"] == 1)].copy()
    negative_sample: pd.DataFrame = data.loc[
        (data["label"] != 1)
    ].copy()  # Use original label

    if len(positive_sample) == 0:
        print("Error: No positive samples (label=1) found.")
        exit(1)

    # --- Split Data for Bloom Filter Testing ---
    # We only need positives (to add) and a held-out set of negatives (to test FPR)
    neg_keys_test: List[str]  # Held-out negatives for FPR testing

    if len(negative_sample) == 0:
        print("Warning: No negative samples found. Cannot evaluate FPR.")
        neg_keys_test = []
    else:
        # Split negatives: Hold out TEST_SPLIT fraction for final testing
        # The rest of the negatives are simply ignored for standard BF test
        _, neg_test_df = train_test_split(  # Use _ for the part we discard
            negative_sample, test_size=TEST_SPLIT, random_state=SEED
        )
        neg_keys_test = list(neg_test_df["key"])

    pos_keys: List[str] = list(positive_sample["key"])

    print(f"Total positive samples (capacity): {len(pos_keys)}")
    print(f"Total negative samples: {len(negative_sample)}")
    print(f"  - Negatives for final testing: {len(neg_keys_test)}")

    # --- Construct Bloom Filter ---
    print("\nConstructing Generic Bloom Filter...")
    print(f"Capacity={len(pos_keys)}, Target Error Rate={ERROR_RATE:.6f}")
    construction_start_time: float = time.time()
    try:
        # Instantiate the generic BloomFilter with the specific KeyType (str)
        bf = BloomFilter[str](capacity=len(pos_keys), error_rate=ERROR_RATE)
        # Add positive keys
        for key in pos_keys:
            bf.add(key)

    except Exception as e:
        print(f"An unexpected error occurred during Bloom Filter construction: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    construction_end_time: float = time.time()
    print(
        f"Bloom Filter construction (adding {len(pos_keys)} items) took {construction_end_time - construction_start_time:.4f} seconds."
    )
    print(
        f"Theoretical parameters: size(m)={bf.bit_size}, num_hashes(k)={bf.num_hashes}"
    )

    # --- Verification: No False Negatives ---
    print("\nVerifying no false negatives...")
    fn_cnt: int = 0
    verify_start_time = time.time()
    for key in pos_keys:
        if key not in bf:  # Uses the __contains__ method
            print(f"False Negative detected: Key={key}")
            fn_cnt += 1
    verify_end_time = time.time()
    if fn_cnt == 0:
        print(
            f"Verification successful: No false negatives found ({verify_end_time - verify_start_time:.2f}s)."
        )
    else:
        # This should never happen with a standard Bloom Filter
        print(
            f"Error: {fn_cnt} false negatives detected out of {len(pos_keys)} ({verify_end_time - verify_start_time:.2f}s)."
        )

    # --- Testing: False Positive Rate ---
    print("\nCalculating false positive rate on test set...")
    fp_cnt: int = 0
    test_start_time = time.time()
    if len(neg_keys_test) > 0:
        for key in neg_keys_test:
            if key in bf:  # Uses the __contains__ method
                fp_cnt += 1
        measured_fpr: float = fp_cnt / len(neg_keys_test)
    else:
        print("Warning: No test negative keys available to measure FPR.")
        measured_fpr = 0.0
    test_end_time = time.time()
    print(f"FPR calculation took {test_end_time - test_start_time:.2f}s.")

    # --- Output Results ---
    print("\n--- Bloom Filter Results ---")
    print(f"Target Error Rate: {ERROR_RATE:.6f}")
    print(
        f"Measured FPR on test set: {measured_fpr:.6f} [{fp_cnt} / {len(neg_keys_test)}]"
    )
    print(
        f"Construction Time (adding items): {construction_end_time - construction_start_time:.4f} seconds"
    )
    # Report theoretical size (m bits)
    mem_bits: int = bf.bit_size
    mem_kib: float = mem_bits / 8.0 / 1024.0
    mem_mib: float = mem_kib / 1024.0
    if mem_mib >= 1.0:
        print(f"Theoretical Memory Usage (m): {mem_mib:.2f} MiB ({mem_bits} bits)")
    elif mem_kib >= 0.01:
        print(f"Theoretical Memory Usage (m): {mem_kib:.2f} KiB ({mem_bits} bits)")
    else:
        print(f"Theoretical Memory Usage (m): {mem_bits} bits")
    print(f"Total Memory Usage: {sizeof(bf)} bytes")
    print(f"Number of Hash Functions (k): {bf.num_hashes}")
    print(f"Number of Items Added (n): {len(bf)}")
    print(f"Estimated Current FP Rate: {bf.get_current_false_positive_rate():.6f}")
    print("----------------")


if __name__ == "__main__":
    main()
