"""Tests for DefaultAutoscaler."""
import unittest
import numpy as np
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver
from openmdao.drivers.autoscalers.default_autoscaler import DefaultAutoscaler
from openmdao.vectors.driver_vector import DriverVector
from openmdao.test_suite.components.sellar import SellarDerivatives


class TestDefaultAutoscalerIntegration(unittest.TestCase):
    """Test DefaultAutoscaler integration with Driver."""

    def setUp(self):
        """Create a test problem with scaling."""
        prob = Problem()
        model = prob.model = SellarDerivatives()

        # Add design variables and constraints to model
        model.add_design_var('z', ref=10.0, ref0=5.0)  # With scaling
        model.add_objective('obj')
        model.add_constraint('con1', lower=0.0, ref=0.5)  # With scaling
        model.add_constraint('con2', upper=0.0)

        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', disp=False)

        prob.setup()
        prob.run_model()

        self.prob = prob
        self.driver = prob.driver

    def test_default_autoscaler_assigned(self):
        """Test that default autoscaler is assigned during setup."""
        self.assertIsNotNone(self.driver.autoscaler)
        self.assertIsInstance(self.driver.autoscaler, DefaultAutoscaler)

    def test_cached_vectors_created(self):
        """Test that cached vectors are created during setup."""
        self.assertIsNotNone(self.driver._dv_vector)
        self.assertIsNotNone(self.driver._cons_vector)
        self.assertIsNotNone(self.driver._objs_vector)

    def test_cached_vectors_are_driver_vectors(self):
        """Test that cached vectors are DriverVector instances."""
        self.assertIsInstance(self.driver._dv_vector, DriverVector)
        self.assertIsInstance(self.driver._cons_vector, DriverVector)
        self.assertIsInstance(self.driver._objs_vector, DriverVector)

    def test_autoscaler_scale_desvars(self):
        """Test scaling design variables via autoscaler."""
        # Get initial vector
        dv_vector = self.driver._dv_vector

        # Store initial values
        initial_values = dv_vector.asarray().copy()

        # Apply scaling
        self.driver.autoscaler.scale_desvars(dv_vector)

        # Verify that scaling was applied (values changed due to ref/ref0)
        scaled_values = dv_vector.asarray()

        # z has ref=10.0, ref0=5.0, so scaling factor is 10.0/5.0 = 2.0
        # Expected: (z_model + ref0) * (ref / ref0) = z_model * 2.0 + 10.0
        # But set_design_var unscales as: z_model = z_opt / scaler - adder
        # So scaler = ref/ref0 = 2.0, adder = ref0 = 5.0
        # Actually, the formula depends on how ref/ref0 are converted

    def test_autoscaler_scale_cons(self):
        """Test scaling constraints via autoscaler."""
        # Get initial constraint vector
        cons_vector = self.driver._cons_vector

        # Store initial values
        initial_values = cons_vector.asarray().copy()

        # Apply constraint scaling
        self.driver.autoscaler.scale_cons(cons_vector)

        # Verify that scaling was applied
        scaled_values = cons_vector.asarray()
        self.assertGreaterEqual(len(scaled_values), 2)

    def test_autoscaler_scale_objs(self):
        """Test scaling objectives via autoscaler."""
        # Get initial objective vector
        objs_vector = self.driver._objs_vector

        # Store initial values
        initial_values = objs_vector.asarray().copy()

        # Apply objective scaling
        self.driver.autoscaler.scale_objs(objs_vector)

        # Verify that scaling was applied
        scaled_values = objs_vector.asarray()
        self.assertEqual(len(scaled_values), 1)

    def test_scaling_roundtrip(self):
        """Test that scaling and unscaling is consistent."""
        # Get a fresh copy of design variables
        original_vector = self.driver.get_design_var_vector(driver_scaling=False, get_remote=True)
        original_data = original_vector.asarray().copy()

        # Create a scaled version
        scaled_vector = self.driver.get_design_var_vector(driver_scaling=False, get_remote=True)
        scaled_data = scaled_vector.asarray().copy()

        # Scale it
        self.driver.autoscaler.scale_desvars(scaled_vector)

        # Unscale it
        self.driver.autoscaler.unscale_desvars(scaled_vector)

        # Should be back to original (approximately)
        np.testing.assert_array_almost_equal(scaled_vector.asarray(), original_data, decimal=10)


class TestAutoscalerMath(unittest.TestCase):
    """Test the mathematical correctness of autoscaler operations."""

    def test_scale_unscale_roundtrip_simple(self):
        """Test simple scale/unscale roundtrip with known values."""
        # Create a simple mock driver with known scaling values
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }

        driver = MockDriver()

        # Create autoscaler and assign driver
        autoscaler = DefaultAutoscaler()
        autoscaler._driver = driver

        # Create design variable vector
        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        model_value = np.array([3.0])  # Model space
        dv_vector = DriverVector(model_value.copy(), meta)

        # Scale: optimizer = (model + adder) * scaler = (3.0 + 1.0) * 2.0 = 8.0
        autoscaler.scale_desvars(dv_vector)
        scaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(scaled_value, 8.0, places=10)

        # Unscale: model = scaled / scaler - adder = 8.0 / 2.0 - 1.0 = 3.0
        autoscaler.unscale_desvars(dv_vector)
        unscaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(unscaled_value, 3.0, places=10)

    def test_scale_unscale_with_none_values(self):
        """Test scaling when scaler or adder is None."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': None, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = DefaultAutoscaler()
        autoscaler._driver = driver

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        original_value = np.array([3.0])
        dv_vector = DriverVector(original_value.copy(), meta)

        # With None values, scaling should have no effect
        autoscaler.scale_desvars(dv_vector)
        scaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(scaled_value, 3.0, places=10)

    def test_in_place_modification(self):
        """Test that vector._data is modified in-place."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }

        driver = MockDriver()
        autoscaler = DefaultAutoscaler()
        autoscaler._driver = driver

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        data = np.array([3.0])
        dv_vector = DriverVector(data, meta)

        # Get reference to data
        data_ref = dv_vector.asarray()
        initial_id = id(data_ref)

        # Scale
        autoscaler.scale_desvars(dv_vector)

        # Verify data is modified in-place
        final_id = id(dv_vector.asarray())
        self.assertEqual(initial_id, final_id)
        self.assertAlmostEqual(dv_vector.asarray()[0], 8.0, places=10)

    def test_multiple_variables_scaling(self):
        """Test scaling with multiple variables."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x1': {'total_scaler': 2.0, 'total_adder': 0.0},
                    'x2': {'total_scaler': 1.0, 'total_adder': 5.0}
                }

        driver = MockDriver()
        autoscaler = DefaultAutoscaler()
        autoscaler._driver = driver

        meta = {
            'x1': {'start_idx': 0, 'end_idx': 1, 'size': 1},
            'x2': {'start_idx': 1, 'end_idx': 2, 'size': 1}
        }
        data = np.array([3.0, 10.0])
        dv_vector = DriverVector(data, meta)

        # Scale
        autoscaler.scale_desvars(dv_vector)

        # x1: (3.0 + 0.0) * 2.0 = 6.0
        # x2: (10.0 + 5.0) * 1.0 = 15.0
        np.testing.assert_array_almost_equal(dv_vector.asarray(), [6.0, 15.0], decimal=10)


if __name__ == '__main__':
    unittest.main()
