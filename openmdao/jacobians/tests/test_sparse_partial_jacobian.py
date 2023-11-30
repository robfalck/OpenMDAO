import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class _InputResidComp(om.ImplicitComponent):

    def __init__(self, jac):
        self._jac = jac
        super().__init__()

    def add_residual_from_input(self, name, **kwargs):
        resid_name = 'resid_' + name
        shape = kwargs['shape'] if 'shape' in kwargs else (1,)
        size = np.prod(shape)
        ar = np.arange(size, dtype=int)

        self.add_input(name, **kwargs)
        self.add_residual(resid_name, **kwargs)

        if self._jac in ('fd', 'cs'):
            self.declare_partials(of=resid_name, wrt=name, method=self._jac)
        elif self._jac == 'dense':
            self.declare_partials(of=resid_name, wrt=name, val=np.eye(size))
        elif self._jac == 'sparse':
            self.declare_partials(of=resid_name, wrt=name, rows=ar, cols=ar, val=1.0)
        else:
            raise ValueError('invalid value for jac use one of ', ['fd', 'cs', 'dense', 'sparse'])


    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals.set_val(inputs.asarray())


class _TestGroup(om.Group):

    def __init__(self, jac='fd'):
        self._jac = jac
        super().__init__()

    def setup(self):
        self.add_subsystem('exec_com', om.ExecComp(['res_a = a - x[0]', 'res_b = b - x[1:]'],
                                                   a={'shape': (1,)},
                                                   b={'shape': (2,)},
                                                   res_a={'shape': (1,)},
                                                   res_b={'shape': (2,)},
                                                   x={'shape':3}),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('resid_comp', _InputResidComp(jac=self._jac),
                           promotes_inputs=['*'], promotes_outputs=['*'])

    def configure(self):
        resid_comp = self._get_subsystem('resid_comp')
        resid_comp.add_output('x', shape=(3,))
        resid_comp.add_residual_from_input('res_a', shape=(1,))
        resid_comp.add_residual_from_input('res_b', shape=(2,))
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()


class TestAddResidualConfigure(unittest.TestCase):

    def test_add_residual_configure_with_jacs(self):

        for jac in ['dense']:#, 'sparse']:
            with self.subTest(msg=jac):
                p = om.Problem()
                p.model.add_subsystem('test_group', _TestGroup(jac=jac))
                p.setup()

                p.set_val('test_group.a', 3.0)
                p.set_val('test_group.b', [4.0, 5.0])

                p.run_model()

                a = p.get_val('test_group.a')
                b = p.get_val('test_group.b')
                x = p.get_val('test_group.x')

                assert_near_equal(a, x[0], tolerance=1.0E-9)
                assert_near_equal(b, x[1:], tolerance=1.0E-9)


class TestSparsePartialJacobian(unittest.TestCase):

    def test_sparse_partial_jac(self):
        pass
