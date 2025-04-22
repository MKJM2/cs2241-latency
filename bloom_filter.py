import math
import xxhash  # type: ignore[import-not-found]
import struct  # For integer serialization
from bitarray import bitarray  # type: ignore[import-not-found]
from typing import List, Final, Tuple, Callable, Any

# --- Constants ---
XXH_SEED1: Final[int] = 0
XXH_SEED2: Final[int] = 6917
# Use struct for consistent integer serialization (e.g., 64-bit signed long long)
# '<q' = little-endian signed long long (8 bytes)
# '>q' = big-endian signed long long (8 bytes)
# Choose one and stick to it. Big-endian is common network order.
INT_STRUCT_FORMAT: Final[str] = ">q"
INT_BYTE_LENGTH: Final[int] = struct.calcsize(INT_STRUCT_FORMAT)  # Should be 8


class BloomFilter:
    """
    A production-ready, high-performance Bloom filter optimized for speed
    by using type-specific methods, avoiding runtime checks in core paths.

    Requires callers to use the appropriate add_* and contains_* methods
    for the type of item being processed (bytes, str, int).

    Features:
    - xxhash (xxh64) for fast hashing.
    - bitarray package for C-optimized bit manipulation.
    - Kirsch-Mitzenmacher optimization (double hashing).
    - Type-specific methods for add/contains operations.
    - __slots__ for reduced memory overhead and faster attribute access.
    """

    __slots__ = (
        "capacity",
        "error_rate",
        "size",
        "num_hashes",
        "bit_array",
        "num_items",
        "_hasher1_intdigest",
        "_hasher2_intdigest",
    )

    # Type alias for the hash function signature
    IntDigestHasher = Callable[[bytes], int]

    def __init__(self, capacity: int, error_rate: float):
        """
        Initializes the Bloom filter.

        Args:
            capacity: The expected number of items to be stored (n).
            error_rate: The desired false positive probability (p), e.g., 0.001.

        Raises:
            ValueError: If capacity is non-positive or error_rate is not in (0, 1).
        """
        if not capacity > 0:
            raise ValueError("Capacity must be positive")
        if not 0 < error_rate < 1:
            raise ValueError("Error rate must be between 0 and 1")

        self.capacity = capacity
        self.error_rate = error_rate

        self.size, self.num_hashes = self._calculate_optimal_params(
            capacity, error_rate
        )

        # Initialize bit array using the C-backed bitarray
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

        self.num_items = 0

        # Initialize hashers using xxh64_intdigest for direct integer output
        self._hasher1_intdigest: BloomFilter.IntDigestHasher = (
            lambda b: xxhash.xxh64_intdigest(b, seed=XXH_SEED1)
        )
        self._hasher2_intdigest: BloomFilter.IntDigestHasher = (
            lambda b: xxhash.xxh64_intdigest(b, seed=XXH_SEED2)
        )

    @staticmethod
    def _calculate_optimal_params(capacity: int, error_rate: float) -> Tuple[int, int]:
        """Calculates optimal size (m) and hash count (k)."""
        if capacity <= 0:  # Should be caught by __init__, but defensive check
            return 1, 1

        # m = - (n * ln(p)) / (ln(2)^2)
        m_float = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        size = max(1, int(math.ceil(m_float)))  # Ensure size is at least 1

        # k = (m / n) * ln(2)
        k_float = (size / capacity) * math.log(2)
        num_hashes = max(1, int(math.ceil(k_float)))  # Ensure at least 1 hash

        return size, num_hashes

    def _get_indices(self, item_bytes: bytes) -> List[int]:
        """Generates k indices using double hashing with xxhash."""
        # This is the core hashing logic, kept tight.
        h1 = self._hasher1_intdigest(item_bytes)
        h2 = self._hasher2_intdigest(item_bytes)
        m = self.size
        # List comprehension is generally efficient here
        return [(h1 + i * h2) % m for i in range(self.num_hashes)]

    # --- Core Add/Contains Logic (Internal, operates on bytes) ---

    def _add_indices(self, indices: List[int]) -> None:
        """Sets the bits at the given indices in the bit array."""
        # This loop relies on the speed of bitarray's C implementation
        bit_arr = self.bit_array  # Local reference slight optimization
        for index in indices:
            bit_arr[index] = 1

    def _check_indices(self, indices: List[int]) -> bool:
        """Checks if all bits at the given indices are set."""
        bit_arr = self.bit_array  # Local reference
        for index in indices:
            if not bit_arr[index]:
                return False  # Definitely not present (early exit)
        return True  # Possibly present

    # --- Public Type-Specific Add Methods ---

    def add_bytes(self, item: bytes) -> None:
        """
        Adds raw bytes to the Bloom filter.

        Args:
            item: The bytes object to add.
        """
        indices = self._get_indices(item)
        self._add_indices(indices)
        self.num_items += 1

    def add_str(self, item: str, encoding: str = "utf-8") -> None:
        """
        Adds a string to the Bloom filter after encoding it.

        Args:
            item: The string to add.
            encoding: The encoding to use (default: 'utf-8').
        """
        item_bytes = item.encode(encoding)
        indices = self._get_indices(item_bytes)
        self._add_indices(indices)
        self.num_items += 1

    def add_int(self, item: int) -> None:
        """
        Adds an integer to the Bloom filter after serializing it to bytes.

        Args:
            item: The integer to add. Uses struct packing based on
                  INT_STRUCT_FORMAT (default: 8-byte big-endian).
        """
        try:
            item_bytes = struct.pack(INT_STRUCT_FORMAT, item)
        except struct.error as e:
            # Handle cases where the integer might be too large for the format
            raise ValueError(
                f"Integer {item} cannot be packed into format "
                f"'{INT_STRUCT_FORMAT}'. Error: {e}"
            ) from e
        indices = self._get_indices(item_bytes)
        self._add_indices(indices)
        self.num_items += 1

    def add(self, item: Any) -> None:
        """
        Adds an item to the Bloom filter, automatically handling type conversion.

        Args:
            item: The item to add. Supported types are bytes, str, and int.
                Strings are encoded using UTF-8 by default.
        """
        if isinstance(item, bytes):
            self.add_bytes(item)
        elif isinstance(item, str):
            self.add_str(item)
        elif isinstance(item, int):
            self.add_int(item)
        else:
            raise TypeError(
                f"Unsupported type: {type(item).__name__!r}. Supported types: bytes, str, int"
            )

    # --- Public Type-Specific Contains Methods ---

    def contains_bytes(self, item: bytes) -> bool:
        """
        Checks if raw bytes might be in the Bloom filter.

        Args:
            item: The bytes object to check.

        Returns:
            True if the item is possibly in the set (may be a false positive).
            False if the item is definitely not in the set.
        """
        indices = self._get_indices(item)
        return self._check_indices(indices)

    def contains_str(self, item: str, encoding: str = "utf-8") -> bool:
        """
        Checks if a string might be in the Bloom filter after encoding it.

        Args:
            item: The string to check.
            encoding: The encoding to use (default: 'utf-8').

        Returns:
            True if the item is possibly in the set (may be a false positive).
            False if the item is definitely not in the set.
        """
        item_bytes = item.encode(encoding)
        indices = self._get_indices(item_bytes)
        return self._check_indices(indices)

    def contains_int(self, item: int) -> bool:
        """
        Checks if an integer might be in the Bloom filter after serializing it.

        Args:
            item: The integer to check. Uses struct packing based on
                  INT_STRUCT_FORMAT.

        Returns:
            True if the item is possibly in the set (may be a false positive).
            False if the item is definitely not in the set.
        """
        try:
            item_bytes = struct.pack(INT_STRUCT_FORMAT, item)
        except struct.error as _e:
            # If packing fails, it cannot have been added with add_int
            # Log warning or re-raise depending on desired strictness
            # print(f"Warning: Integer {item} cannot be packed. Returning False.")
            return False  # Or raise ValueError if invalid input shouldn't be checked
        indices = self._get_indices(item_bytes)
        return self._check_indices(indices)

    def __contains__(self, item: Any) -> bool:
        """
        Checks if an item might be in the Bloom filter.

        Args:
            item: The item to check. Supported types are bytes, str, and int.

        Returns:
            True if the item is possibly in the set (may be a false positive).
            False if the item is definitely not in the set.
        """
        if isinstance(item, bytes):
            return self.contains_bytes(item)
        elif isinstance(item, str):
            return self.contains_str(item)
        elif isinstance(item, int):
            return self.contains_int(item)
        else:
            raise TypeError(
                f"Unsupported type: {type(item).__name__!r}. Supported types: bytes, str, int"
            )

    # --- Other Public Methods ---

    def __len__(self) -> int:
        """Returns the number of items added (approximate capacity usage)."""
        return self.num_items

    def get_current_false_positive_rate(self) -> float:
        """
        Estimates the current theoretical false positive rate based on the
        number of items added (`num_items`). This rate increases as more
        items are added than the initial capacity.

        Formula: (1 - exp(-k * n / m))^k
        Where:
            k = num_hashes
            n = num_items (current number added)
            m = size (bits in filter)

        Returns:
            The estimated false positive probability (float between 0.0 and 1.0).
            Returns 1.0 if parameters lead to calculation errors (e.g., overflow).
        """
        k = self.num_hashes
        n = self.num_items
        m = self.size

        if m == 0:  # Avoid division by zero
            return 1.0

        try:
            # Use floating point division
            exponent = -k * n / float(m)
            # Calculate (1 - (1/e)^(kn/m))^k which is equivalent but potentially
            # more stable if exp(-kn/m) is very small.
            # Or stick to the direct formula:
            rate = (1.0 - math.exp(exponent)) ** k
        except (OverflowError, ValueError):
            # If intermediate calculations overflow or result in math domain errors,
            # the theoretical FP rate is effectively 1.0
            rate = 1.0

        # Clamp result just in case of floating point inaccuracies
        return max(0.0, min(1.0, rate))

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the filter."""
        return (
            f"{self.__class__.__name__}("
            f"capacity={self.capacity}, "
            f"error_rate={self.error_rate:.2e}, "
            f"size={self.size}, "
            f"num_hashes={self.num_hashes}, "
            f"num_items={self.num_items})"
        )


if __name__ == "__main__":
    # --- Example Usage ---
    bf = BloomFilter(capacity=100000, error_rate=0.01)  # 100k items, 1% FP

    # Must use type-specific methods
    bf.add_str("hello world")
    bf.add_bytes(b"raw data")
    bf.add_int(1234567890)
    bf.add_int(-98765)  # Works with signed format '>q'

    print(bf)
    print(f"Contains 'hello world': {bf.contains_str('hello world')}")
    print(f"Contains b'raw data': {bf.contains_bytes(b'raw data')}")
    print(f"Contains 1234567890: {bf.contains_int(1234567890)}")
    print(f"Contains -98765: {bf.contains_int(-98765)}")
    print(f"Contains 'missing str': {bf.contains_str('missing str')}")
    print(f"Contains 9999: {bf.contains_int(9999)}")

    # Example of adding many items
    for i in range(50000):
        bf.add_int(i)

    print(f"\nAfter adding 50k integers:")
    print(bf)
    print(f"Estimated current FP rate: {bf.get_current_false_positive_rate():.4f}")

    # Checking a non-member integer
    print(f"Contains 1000000 (not added): {bf.contains_int(1000000)}")
