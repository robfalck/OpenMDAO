import unittest
import numpy as np
from smt.applications import EGO
import openmdao.api as om
import matplotlib.pyplot as plt

from openmdao.utils.assert_utils import assert_near_equal


def function_test_1d(x):
    # function xsinx
    import numpy as np

    x = np.reshape(x, (-1,))
    y = np.zeros(x.shape)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    return y.reshape((-1, 1))


class TestEGODriver(unittest.TestCase):

    def test_smt_ego_example(self):
        p = om.Problem()

        p.driver = om.EGODriver(max_iter=10, criterion='EI', random_state=42)
        p.driver.add_recorder(om.SqliteRecorder('ego_driver_test.sql'))

        p.model.add_subsystem('exec_comp', om.ExecComp('y = (x - 3.5) * sin((x - 3.5) / pi)'),
                            promotes_inputs=['x'], promotes_outputs=['y'])

        p.model.add_objective('y')  # Minimize
        p.model.add_design_var('x', lower=0.0, upper=25.0)

        # By not providing training data for the objective here, the driver will
        # first run the model to obtain the objective training data.
        p.driver.set_training_data({'x': np.array([[0, 7, 25]]).T})
        p.driver.options['random_state'] = 42

        p.setup()
        p.run_driver()

        assert_near_equal(p.get_val('x'), 18.9479, tolerance=1.0E-4)
        assert_near_equal(p.get_val('y'), -15.125, tolerance=1.0E-4)

        # print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(p.get_val('x')), float(p.get_val('y'))))

    def test_smt_ego_example_with_ydoe(self):
        p = om.Problem()

        p.driver = EGODriver(max_iter=6, criterion='EI', random_state=42)
        p.driver.add_recorder(om.SqliteRecorder('ego_driver_test.sql'))

        p.model.add_subsystem('exec_comp', om.ExecComp('y = (x - 3.5) * sin((x - 3.5) / pi)'),
                            promotes_inputs=['x'], promotes_outputs=['y'])

        p.model.add_objective('y')  # Minimize
        p.model.add_design_var('x', lower=0.0, upper=25.0)

        # By not providing training data for the objective here, the driver will
        # first run the model to obtain the objective training data.
        p.driver.set_training_data({'x': np.array([[0, 7, 25]]).T,
                                    'y': np.array([[ 3.14127616, 3.14127616, 11.42919546]]).T})
        p.driver.options['random_state'] = 42

        p.setup()
        p.run_driver()

        assert_near_equal(p.get_val('x'), 18.9479, tolerance=1.0E-4)
        assert_near_equal(p.get_val('y'), -15.125, tolerance=1.0E-4)


if __name__ == '__main__':
    unittest.main()
