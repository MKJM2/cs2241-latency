#!/usr/bin/env python3
import unittest
import math
import random
import string
from bitarray import bitarray
from typing import Any

from bloom_filter import BloomFilter

# --- Helper Function ---
def generate_random_string(length: int) -> str:
    """Generates a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


# --- Test Class ---
class TestBloomFilter(unittest.TestCase):
    """Unit tests for the generic BloomFilter class."""

    DEFAULT_CAPACITY: int = 1000
    DEFAULT_ERROR_RATE: float = 0.01

    def test_initialization_valid(self) -> None:
        """Test successful initialization with valid parameters."""
        capacity = self.DEFAULT_CAPACITY
        error_rate = self.DEFAULT_ERROR_RATE
        bf = BloomFilter[str](capacity, error_rate)

        self.assertEqual(bf.capacity, capacity)
        self.assertEqual(bf.error_rate, error_rate)
        self.assertGreater(bf.size, 0)
        self.assertGreater(bf.num_hashes, 0)
        self.assertEqual(bf.num_items, 0)
        self.assertEqual(len(bf), 0)
        self.assertEqual(bf.bit_array.count(1), 0)  # Check all bits are 0
        self.assertEqual(bf.bit_size, bf.size)
        self.assertIsInstance(bf.bit_array, type(bitarray()))

        # Check calculated params are reasonable (derived from formulas)
        expected_m = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        expected_k = (bf.size / capacity) * math.log(2)
        self.assertAlmostEqual(bf.size, math.ceil(expected_m), delta=1)
        self.assertAlmostEqual(bf.num_hashes, math.ceil(expected_k), delta=1)
        # Ensure at least 1 for size and hashes
        self.assertGreaterEqual(bf.size, 1)
        self.assertGreaterEqual(bf.num_hashes, 1)

    def test_initialization_invalid_capacity(self) -> None:
        """Test initialization with invalid capacity values."""
        with self.assertRaisesRegex(ValueError, "Capacity must be positive"):
            BloomFilter[str](0, self.DEFAULT_ERROR_RATE)
        with self.assertRaisesRegex(ValueError, "Capacity must be positive"):
            BloomFilter[str](-100, self.DEFAULT_ERROR_RATE)

    def test_initialization_invalid_error_rate(self) -> None:
        """Test initialization with invalid error rate values."""
        with self.assertRaisesRegex(
            ValueError, "Error rate must be between 0 and 1"
        ):
            BloomFilter[str](self.DEFAULT_CAPACITY, 0.0)
        with self.assertRaisesRegex(
            ValueError, "Error rate must be between 0 and 1"
        ):
            BloomFilter[str](self.DEFAULT_CAPACITY, 1.0)
        with self.assertRaisesRegex(
            ValueError, "Error rate must be between 0 and 1"
        ):
            BloomFilter[str](self.DEFAULT_CAPACITY, -0.1)
        with self.assertRaisesRegex(
            ValueError, "Error rate must be between 0 and 1"
        ):
            BloomFilter[str](self.DEFAULT_CAPACITY, 1.1)

    def test_calculate_optimal_params_edge_cases(self) -> None:
        """Test _calculate_optimal_params static method edge cases."""
        # Capacity 1
        size, num_hashes = BloomFilter._calculate_optimal_params(1, 0.01)
        self.assertGreaterEqual(size, 1)
        self.assertGreaterEqual(num_hashes, 1)

        # Very low error rate (requires larger size/hashes)
        size_low_err, num_hashes_low_err = BloomFilter._calculate_optimal_params(
            1000, 1e-9
        )
        self.assertGreaterEqual(size_low_err, 1)
        self.assertGreaterEqual(num_hashes_low_err, 1)

        # Very high error rate (requires smaller size/hashes, but still >= 1)
        size_high_err, num_hashes_high_err = BloomFilter._calculate_optimal_params(
            1000, 0.99
        )
        self.assertGreaterEqual(size_high_err, 1)
        self.assertGreaterEqual(num_hashes_high_err, 1)

    def test_add_and_contains_basic_str(self) -> None:
        """Test adding and checking string items."""
        bf = BloomFilter[str](100, 0.01)
        item1 = "hello"
        item2 = "world"
        item_not_added = "test"

        self.assertFalse(item1 in bf)
        self.assertFalse(item2 in bf)
        self.assertFalse(item_not_added in bf)
        self.assertEqual(len(bf), 0)

        bf.add(item1)
        self.assertTrue(item1 in bf)
        self.assertFalse(item2 in bf)  # Should still be false
        self.assertFalse(item_not_added in bf)
        self.assertEqual(len(bf), 1)
        self.assertEqual(bf.num_items, 1)

        bf.add(item2)
        self.assertTrue(item1 in bf)
        self.assertTrue(item2 in bf)
        self.assertFalse(item_not_added in bf)
        self.assertEqual(len(bf), 2)
        self.assertEqual(bf.num_items, 2)

    def test_add_and_contains_basic_int(self) -> None:
        """Test adding and checking integer items."""
        bf = BloomFilter[int](100, 0.01)
        item1 = 12345
        item2 = 98765
        item_not_added = 54321

        self.assertFalse(item1 in bf)
        self.assertFalse(item2 in bf)
        self.assertFalse(item_not_added in bf)
        self.assertEqual(len(bf), 0)

        bf.add(item1)
        self.assertTrue(item1 in bf)
        self.assertFalse(item2 in bf)
        self.assertFalse(item_not_added in bf)
        self.assertEqual(len(bf), 1)

        bf.add(item2)
        self.assertTrue(item1 in bf)
        self.assertTrue(item2 in bf)
        self.assertFalse(item_not_added in bf)
        self.assertEqual(len(bf), 2)

    def test_no_false_negatives(self) -> None:
        """Verify that items added are always reported as present."""
        capacity = 5000
        bf = BloomFilter[str](capacity, 0.001)
        items_to_add = [generate_random_string(20) for _ in range(capacity)]

        for item in items_to_add:
            bf.add(item)

        self.assertEqual(len(bf), capacity)

        false_negatives = 0
        for item in items_to_add:
            if item not in bf:
                false_negatives += 1
                print(f"False negative detected for: {item}") # Log for debugging

        self.assertEqual(
            false_negatives, 0, "Bloom filter should not have false negatives"
        )

    def test_add_duplicate_items(self) -> None:
        """Test adding the same item multiple times."""
        bf = BloomFilter[str](100, 0.01)
        item = "duplicate"

        bf.add(item)
        self.assertEqual(len(bf), 1)
        self.assertTrue(item in bf)
        # Check bit count - should be exactly num_hashes bits set if no collisions yet
        initial_set_bits = bf.bit_array.count(1)
        self.assertLessEqual(initial_set_bits, bf.num_hashes)

        bf.add(item) # Add again
        self.assertEqual(len(bf), 2, "num_items should increment even for duplicates")
        self.assertTrue(item in bf)
        # Bit count should not change when adding a duplicate
        self.assertEqual(bf.bit_array.count(1), initial_set_bits)

    def test_get_indices_consistency(self) -> None:
        """Test that _get_indices returns consistent results for the same input."""
        bf = BloomFilter[str](100, 0.01)
        item_bytes = b"test_bytes"
        indices1 = bf._get_indices(item_bytes)
        indices2 = bf._get_indices(item_bytes)

        self.assertEqual(indices1, indices2)
        self.assertEqual(len(indices1), bf.num_hashes)
        # Check indices are within bounds
        for index in indices1:
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, bf.size)

    def test_add_check_indices_internal(self) -> None:
        """Test the internal _add_indices and _check_indices methods."""
        bf = BloomFilter[str](100, 0.01)
        item_bytes = b"another_test"
        indices = bf._get_indices(item_bytes)

        # Initially, indices should not be set
        self.assertFalse(bf._check_indices(indices))

        # Add indices
        bf._add_indices(indices)

        # Now, check should return True
        self.assertTrue(bf._check_indices(indices))

        # Check individual bits
        for index in indices:
            self.assertTrue(bf.bit_array[index])

        # Check with a modified list (missing one index)
        if len(indices) > 1:
            self.assertTrue(bf._check_indices(indices[:-1]))

        # Check with an extra, likely unset index
        extra_index = (indices[0] + 1) % bf.size
        if not bf.bit_array[extra_index]: # Ensure it's actually unset
             self.assertFalse(bf._check_indices(indices + [extra_index]))


    def test_serializer_error_add(self) -> None:
        """Test that add handles serializer exceptions."""

        def failing_serializer(x: Any) -> bytes:
            raise ValueError("Serialization failed!")

        bf = BloomFilter[Any](100, 0.01, failing_serializer)
        item = object()

        with self.assertRaisesRegex(
            TypeError, "Failed to serialize item.*Serialization failed!"
        ):
            bf.add(item)
        # Ensure item count didn't change
        self.assertEqual(len(bf), 0)

    def test_serializer_error_contains(self) -> None:
        """Test that __contains__ handles serializer exceptions."""

        def failing_serializer(x: Any) -> bytes:
            raise ValueError("Serialization failed!")

        item = object()

        # Add an item with a working serializer first (optional but good practice)
        bf_working = BloomFilter[str](100, 0.01)
        bf_working.add("test")

        # Now test the failing one
        bf_failing = BloomFilter[Any](100, 0.01, failing_serializer)

        with self.assertRaisesRegex(
            TypeError, "Warning: Failed to serialize item.*Serialization failed!"
        ):
             _ = item in bf_failing # Evaluate __contains__

    def test_len_method(self) -> None:
        """Test the __len__ method."""
        bf = BloomFilter[int](100, 0.01)
        self.assertEqual(len(bf), 0)
        bf.add(1)
        self.assertEqual(len(bf), 1)
        bf.add(2)
        self.assertEqual(len(bf), 2)
        bf.add(1) # Add duplicate
        self.assertEqual(len(bf), 3)

    def test_sizeof_method(self) -> None:
        """Test the __sizeof__ method."""
        bf = BloomFilter[str](100, 0.01)
        expected_bytes = math.ceil(bf.bit_size / 8.0)
        self.assertEqual(bf.__sizeof__(), expected_bytes)

        # Test with a size not divisible by 8
        bf2 = BloomFilter[str](10, 0.1) # Likely results in odd size
        expected_bytes2 = math.ceil(bf2.bit_size / 8.0)
        self.assertEqual(bf2.__sizeof__(), expected_bytes2)


    def test_get_current_false_positive_rate(self) -> None:
        """Test the theoretical FPR calculation."""
        capacity = 1000
        error_rate = 0.01
        bf = BloomFilter[str](capacity, error_rate)

        # 1. Empty filter
        self.assertEqual(bf.get_current_false_positive_rate(), 0.0)

        # 2. Add some items (less than capacity)
        num_added_half = capacity // 2
        for i in range(num_added_half):
            bf.add(f"item_{i}")
        self.assertEqual(len(bf), num_added_half)
        fpr_half = bf.get_current_false_positive_rate()
        self.assertGreater(fpr_half, 0.0)
        self.assertLess(fpr_half, error_rate) # Should be lower than target FPR

        # 3. Add items up to capacity
        for i in range(num_added_half, capacity):
             bf.add(f"item_{i}")
        self.assertEqual(len(bf), capacity)
        fpr_full = bf.get_current_false_positive_rate()
        # Theoretical FPR should be close to the target error rate at capacity
        self.assertAlmostEqual(fpr_full, error_rate, delta=error_rate * 0.5) # Allow some tolerance

        # 4. Add items exceeding capacity
        num_added_over = int(capacity * 1.5)
        for i in range(capacity, num_added_over):
             bf.add(f"item_{i}")
        self.assertEqual(len(bf), num_added_over)
        fpr_over = bf.get_current_false_positive_rate()
        # FPR should increase significantly when overloaded
        self.assertGreater(fpr_over, error_rate)

        # 5. Test formula directly (matches implementation)
        k = bf.num_hashes
        n = bf.num_items
        m = bf.size
        expected_fpr = (1.0 - math.exp(-k * n / float(m))) ** k
        self.assertAlmostEqual(fpr_over, expected_fpr, places=9)

    def test_repr_method(self) -> None:
        """Test the __repr__ method executes and returns a string."""
        bf = BloomFilter[str](50, 0.05)
        bf.add("repr_test")
        representation = repr(bf)

        self.assertIsInstance(representation, str)
        self.assertIn("BloomFilter", representation)
        self.assertIn("capacity=50", representation)
        self.assertIn("error_rate=5.00e-02", representation) # Check scientific notation format
        self.assertIn(f"size={bf.size}", representation)
        self.assertIn(f"num_hashes={bf.num_hashes}", representation)
        self.assertIn("num_items=1", representation)

        # Test with a lambda serializer (no __name__)
        bf_lambda = BloomFilter[int](10, 0.1, lambda x: bytes([x]))
        repr_lambda = repr(bf_lambda)
        self.assertIn("serializer=<lambda>", repr_lambda) # Checks type fallback

        def some_int_serializer(key: int) -> bytes:
            return b"test"
        bf_func = BloomFilter[int](10, 0.1, some_int_serializer)
        repr_func = repr(bf_func)
        self.assertIn("serializer=some_int_serializer", repr_func)


# --- Run Tests ---
if __name__ == "__main__":
    unittest.main()
