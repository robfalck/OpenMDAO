"""Unit tests for the OptimizerVector class."""

import unittest
import numpy as np

from openmdao.vectors.optimizer_vector import OptimizerVector
from openmdao.utils.assert_utils import assert_near_equal


class TestOptimizerVectorBasics(unittest.TestCase):
    """Test basic OptimizerVector functionality without driver integration."""

    def setUp(self):
        """Create simple test metadata and data arrays."""
        # Simple metadata for testing
        self.metadata = {
            'x': {'start_idx': 0, 'end_idx': 1, 'size': 1},
            'z': {'start_idx': 1, 'end_idx': 3, 'size': 2},
        }
        # Create data array: [x_value, z1_value, z2_value]
        self.data = np.array([1.0, 5.0, 2.0])
        self.vec = OptimizerVector(self.data, self.metadata)

    def test_getitem_single_value(self):
        """Test __getitem__ for retrieving single-element values by name."""
        x_val = self.vec['x']
        self.assertIsInstance(x_val, np.ndarray)
        assert_near_equal(x_val, np.array([1.0]))

    def test_getitem_multi_value(self):
        """Test __getitem__ for retrieving multi-element values by name."""
        z_val = self.vec['z']
        self.assertIsInstance(z_val, np.ndarray)
        assert_near_equal(z_val, np.array([5.0, 2.0]))

    def test_getitem_invalid_key(self):
        """Test __getitem__ raises KeyError for invalid variable name."""
        with self.assertRaises(KeyError):
            _ = self.vec['invalid_name']

    def test_setitem_single_value(self):
        """Test __setitem__ for setting single-element values by name."""
        self.vec['x'] = 3.5
        assert_near_equal(self.vec['x'], np.array([3.5]))
        assert_near_equal(self.data[0], 3.5)

    def test_setitem_multi_value(self):
        """Test __setitem__ for setting multi-element values by name."""
        self.vec['z'] = np.array([7.0, 8.0])
        assert_near_equal(self.vec['z'], np.array([7.0, 8.0]))
        assert_near_equal(self.data[1:3], np.array([7.0, 8.0]))

    def test_setitem_broadcasts_scalar(self):
        """Test __setitem__ broadcasts scalar to multi-element values."""
        self.vec['z'] = 5.0
        assert_near_equal(self.vec['z'], np.array([5.0, 5.0]))

    def test_setitem_invalid_key(self):
        """Test __setitem__ raises KeyError for invalid variable name."""
        with self.assertRaises(KeyError):
            self.vec['invalid_name'] = 1.0

    def test_contains(self):
        """Test __contains__ for checking variable existence."""
        self.assertIn('x', self.vec)
        self.assertIn('z', self.vec)
        self.assertNotIn('invalid_name', self.vec)

    def test_len(self):
        """Test __len__ returns number of variables."""
        self.assertEqual(len(self.vec), 2)

    def test_iter(self):
        """Test __iter__ for iterating over variable names."""
        names = list(self.vec)
        self.assertEqual(len(names), 2)
        self.assertIn('x', names)
        self.assertIn('z', names)

    def test_keys(self):
        """Test keys() returns variable names."""
        keys = list(self.vec.keys())
        self.assertEqual(len(keys), 2)
        self.assertIn('x', keys)
        self.assertIn('z', keys)

    def test_values(self):
        """Test values() iterates over variable values."""
        values = list(self.vec.values())
        self.assertEqual(len(values), 2)
        # First value should be x
        assert_near_equal(values[0], np.array([1.0]))
        # Second value should be z
        assert_near_equal(values[1], np.array([5.0, 2.0]))

    def test_items(self):
        """Test items() iterates over (name, value) pairs."""
        items = list(self.vec.items())
        self.assertEqual(len(items), 2)

        # Check that we can iterate and get correct name-value pairs
        items_dict = dict(items)
        assert_near_equal(items_dict['x'], np.array([1.0]))
        assert_near_equal(items_dict['z'], np.array([5.0, 2.0]))

    def test_asarray(self):
        """Test asarray() returns underlying numpy array."""
        array = self.vec.asarray()
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.ndim, 1)
        assert_near_equal(array, np.array([1.0, 5.0, 2.0]))

    def test_asarray_returns_reference(self):
        """Test asarray() returns reference, not a copy."""
        array_ref = self.vec.asarray()
        array_ref[0] = 999.0
        assert_near_equal(self.vec['x'], np.array([999.0]))

    def test_get_metadata_all(self):
        """Test get_metadata() without name returns all metadata."""
        all_meta = self.vec.get_metadata()
        self.assertIsInstance(all_meta, dict)
        self.assertIn('x', all_meta)
        self.assertIn('z', all_meta)
        self.assertEqual(len(all_meta), 2)

    def test_get_metadata_specific(self):
        """Test get_metadata(name) returns metadata for specific variable."""
        x_meta = self.vec.get_metadata('x')
        self.assertIsInstance(x_meta, dict)
        self.assertEqual(x_meta['start_idx'], 0)
        self.assertEqual(x_meta['end_idx'], 1)
        self.assertEqual(x_meta['size'], 1)

        z_meta = self.vec.get_metadata('z')
        self.assertEqual(z_meta['start_idx'], 1)
        self.assertEqual(z_meta['end_idx'], 3)
        self.assertEqual(z_meta['size'], 2)

    def test_get_metadata_invalid_key(self):
        """Test get_metadata() raises KeyError for invalid variable name."""
        with self.assertRaises(KeyError):
            self.vec.get_metadata('invalid_name')


class TestOptimizerVectorReshaping(unittest.TestCase):
    """Test OptimizerVector handles reshaping correctly."""

    def test_scalar_access_returns_flat_array(self):
        """Test accessing scalar variables returns 1D array."""
        metadata = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        data = np.array([5.0])
        vec = OptimizerVector(data, metadata)

        x = vec['x']
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape[0], 1)

    def test_array_access_returns_flat_array(self):
        """Test accessing array variables returns 1D array."""
        metadata = {'z': {'start_idx': 0, 'end_idx': 3, 'size': 3}}
        data = np.array([1.0, 2.0, 3.0])
        vec = OptimizerVector(data, metadata)

        z = vec['z']
        self.assertEqual(z.ndim, 1)
        self.assertEqual(z.shape[0], 3)


class TestOptimizerVectorConstraintMetadata(unittest.TestCase):
    """Test OptimizerVector with constraint metadata."""

    def setUp(self):
        """Create metadata for constraints with bounds."""
        self.metadata = {
            'con1': {
                'start_idx': 0,
                'end_idx': 1,
                'size': 1,
                'type': 'constraint',
                'linear': False,
                'equals': None,
                'lower': 0.0,
                'upper': 100.0,
            },
            'con2': {
                'start_idx': 1,
                'end_idx': 3,
                'size': 2,
                'type': 'constraint',
                'linear': True,
                'equals': 5.0,
                'lower': None,
                'upper': None,
            },
        }
        self.data = np.array([50.0, 3.0, 4.0])
        self.vec = OptimizerVector(self.data, self.metadata)

    def test_constraint_metadata_accessible(self):
        """Test constraint metadata is accessible via get_metadata."""
        con1_meta = self.vec.get_metadata('con1')
        self.assertEqual(con1_meta['type'], 'constraint')
        self.assertEqual(con1_meta['lower'], 0.0)
        self.assertEqual(con1_meta['upper'], 100.0)
        self.assertIsNone(con1_meta['equals'])

        con2_meta = self.vec.get_metadata('con2')
        self.assertEqual(con2_meta['type'], 'constraint')
        self.assertIsNone(con2_meta['lower'])
        self.assertIsNone(con2_meta['upper'])
        self.assertEqual(con2_meta['equals'], 5.0)

    def test_constraint_values_accessible(self):
        """Test constraint values are accessible by name."""
        con1_val = self.vec['con1']
        assert_near_equal(con1_val, np.array([50.0]))

        con2_val = self.vec['con2']
        assert_near_equal(con2_val, np.array([3.0, 4.0]))


class TestOptimizerVectorObjectiveMetadata(unittest.TestCase):
    """Test OptimizerVector with objective metadata."""

    def setUp(self):
        """Create metadata for objectives."""
        self.metadata = {
            'obj1': {
                'start_idx': 0,
                'end_idx': 1,
                'size': 1,
                'type': 'objective',
            },
            'obj2': {
                'start_idx': 1,
                'end_idx': 2,
                'size': 1,
                'type': 'objective',
            },
        }
        self.data = np.array([100.0, 200.0])
        self.vec = OptimizerVector(self.data, self.metadata)

    def test_objective_metadata_accessible(self):
        """Test objective metadata is accessible."""
        obj1_meta = self.vec.get_metadata('obj1')
        self.assertEqual(obj1_meta['type'], 'objective')

        obj2_meta = self.vec.get_metadata('obj2')
        self.assertEqual(obj2_meta['type'], 'objective')

    def test_objective_values_accessible(self):
        """Test objective values are accessible by name."""
        obj1_val = self.vec['obj1']
        assert_near_equal(obj1_val, np.array([100.0]))

        obj2_val = self.vec['obj2']
        assert_near_equal(obj2_val, np.array([200.0]))


class TestOptimizerVectorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_vector(self):
        """Test OptimizerVector with no variables."""
        metadata = {}
        data = np.array([])
        vec = OptimizerVector(data, metadata)

        self.assertEqual(len(vec), 0)
        self.assertEqual(list(vec.keys()), [])

    def test_single_variable(self):
        """Test OptimizerVector with single variable."""
        metadata = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        data = np.array([42.0])
        vec = OptimizerVector(data, metadata)

        self.assertEqual(len(vec), 1)
        assert_near_equal(vec['x'], np.array([42.0]))

    def test_large_array_access(self):
        """Test accessing large arrays."""
        size = 1000
        metadata = {'large_var': {'start_idx': 0, 'end_idx': size, 'size': size}}
        data = np.arange(size, dtype=float)
        vec = OptimizerVector(data, metadata)

        large_val = vec['large_var']
        self.assertEqual(large_val.shape[0], size)
        assert_near_equal(large_val, np.arange(size, dtype=float))


if __name__ == '__main__':
    unittest.main()
