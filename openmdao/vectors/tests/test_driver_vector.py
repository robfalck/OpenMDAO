"""Unit tests for the DriverVector class."""

import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.test_suite.components.sellar import SellarDerivatives


@use_tempdirs
class TestDriverVector(unittest.TestCase):
    """Test the DriverVector class functionality."""

    def test_driver_vector_getitem(self):
        """Test DriverVector.__getitem__ for retrieving values by name."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 24.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test retrieving variables by name
        x_val = vec['x']
        z_val = vec['z']

        # Verify they are numpy arrays with correct values
        self.assertIsInstance(x_val, np.ndarray)
        self.assertIsInstance(z_val, np.ndarray)
        self.assertEqual(x_val.shape[0], 1)
        self.assertEqual(z_val.shape[0], 2)

        # Verify KeyError for invalid name
        with self.assertRaises(KeyError):
            _ = vec['invalid_name']

    def test_driver_vector_setitem(self):
        """Test DriverVector.__setitem__ for setting values by name."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()
        orig_array = vec.asarray().copy()

        # Set new values by name
        vec['x'] = 2.5
        vec['z'] = [1.0, 2.0]

        # Verify underlying array was modified
        new_array = vec.asarray()
        self.assertFalse(np.allclose(orig_array, new_array))

        # Verify values are correct
        assert_near_equal(vec['x'], np.array([2.5]))
        assert_near_equal(vec['z'], np.array([1.0, 2.0]))

        # Verify KeyError for invalid name
        with self.assertRaises(KeyError):
            vec['invalid_name'] = 1.0

    def test_driver_vector_contains(self):
        """Test DriverVector.__contains__ for checking variable existence."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test __contains__
        self.assertIn('z', vec)
        self.assertNotIn('x', vec)
        self.assertNotIn('obj', vec)

    def test_driver_vector_len(self):
        """Test DriverVector.__len__ for getting number of variables."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        model.add_objective('obj')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test __len__
        self.assertEqual(len(vec), 2)  # z and x

    def test_driver_vector_iter(self):
        """Test DriverVector.__iter__ for iterating over variable names."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test __iter__
        names = list(vec)
        self.assertEqual(len(names), 2)
        self.assertIn('z', names)
        self.assertIn('x', names)

    def test_driver_vector_keys(self):
        """Test DriverVector.keys() for getting variable names."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test keys()
        keys = list(vec.keys())
        self.assertEqual(len(keys), 2)
        self.assertIn('z', keys)
        self.assertIn('x', keys)

    def test_driver_vector_values(self):
        """Test DriverVector.values() for iterating over variable values."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 24.0]))
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test values()
        values = list(vec.values())
        self.assertEqual(len(values), 1)  # Only z
        self.assertIsInstance(values[0], np.ndarray)
        self.assertEqual(values[0].shape[0], 2)

    def test_driver_vector_items(self):
        """Test DriverVector.items() for iterating over (name, value) pairs."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test items()
        items = list(vec.items())
        self.assertEqual(len(items), 2)

        for name, value in items:
            self.assertIn(name, ['z', 'x'])
            self.assertIsInstance(value, np.ndarray)

    def test_driver_vector_asarray(self):
        """Test DriverVector.asarray() returns underlying array."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test asarray()
        array = vec.asarray()
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.ndim, 1)

        # Modify array and verify it's reflected in vector access
        orig_val = vec['x'].copy()
        array[0] = 999.0  # Assuming 'x' is first in alphabetical order after unpacking
        # The modification should be reflected

    def test_driver_vector_get_metadata(self):
        """Test DriverVector.get_metadata() for retrieving metadata."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_design_var('x')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_design_var_vector()

        # Test get_metadata with specific name
        z_meta = vec.get_metadata('z')
        self.assertIsInstance(z_meta, dict)
        self.assertIn('start_idx', z_meta)
        self.assertIn('end_idx', z_meta)
        self.assertIn('size', z_meta)

        # Test get_metadata without name (get all)
        all_meta = vec.get_metadata()
        self.assertIsInstance(all_meta, dict)
        self.assertIn('z', all_meta)
        self.assertIn('x', all_meta)

        # Test KeyError for invalid name
        with self.assertRaises(KeyError):
            vec.get_metadata('invalid_name')

    def test_driver_vector_response_vector(self):
        """Test DriverVector with response vector."""
        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        vec = prob.driver.get_response_vector()

        # Test accessing response values by name
        obj_val = vec['obj']
        con_val = vec['con1']

        self.assertIsInstance(obj_val, np.ndarray)
        self.assertIsInstance(con_val, np.ndarray)

        # Test metadata includes type information
        all_meta = vec.get_metadata()
        self.assertEqual(all_meta['obj']['type'], 'objective')
        self.assertEqual(all_meta['con1']['type'], 'constraint')


if __name__ == '__main__':
    unittest.main()
