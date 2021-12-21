import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


class TestPassthruComp(unittest.TestCase):

    def test_values_and_partials(self):
        p = om.Problem()

        ptc = om.PassthruComp()
        ptc._no_check_partials = False

        p.model.add_subsystem('ptc', ptc)

        ptc.add_var('scalar', val=3.14159, units=None)
        ptc.add_var('vector', val=22/7, shape=(3,), units='m')
        ptc.add_var('tensor', shape=(10, 3, 3), units='kg')

        p.setup(force_alloc_complex=True)

        p.set_val('ptc.vector', np.random.rand(3))
        p.set_val('ptc.tensor', np.random.rand(10, 3, 3))

        p.run_model()

        assert_near_equal(p.get_val('ptc.scalar'), p.get_val('ptc.scalar_value'))
        assert_near_equal(p.get_val('ptc.vector', units='ft'), p.get_val('ptc.vector', units='ft'))
        assert_near_equal(p.get_val('ptc.tensor', units='lbm'), p.get_val('ptc.tensor', units='lbm'))

        cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)

        assert_check_partials(cpd)

    def test_rename(self):
        p = om.Problem()

        ptc = om.PassthruComp()
        ptc._no_check_partials = False

        p.model.add_subsystem('ptc', ptc)

        ptc.add_var('tensor', output_name='T', shape=(10, 3, 3), units='kg*m/s**2')

        p.setup(force_alloc_complex=True)
        p.set_val('ptc.tensor', np.random.rand(10, 3, 3))

        p.run_model()

        assert_near_equal(p.get_val('ptc.tensor'), p.get_val('ptc.T'))

    def test_tags(self):
        p = om.Problem()

        ptc = om.PassthruComp()
        ptc._no_check_partials = False

        p.model.add_subsystem('ptc', ptc)

        ptc.add_var('a', val=10, units=None, tags='a_common', input_tags=['a_only_input'], output_tags=['a_only_output'])
        ptc.add_var('b', val=20, units=None, tags='b_common', input_tags='b_only_input', output_tags='b_only_output')

        p.setup(force_alloc_complex=True)

        p.run_model()

        meta = p.model.get_io_metadata()

        self.assertSetEqual(meta['ptc.a']['tags'], {'a_common', 'a_only_input'})
        self.assertSetEqual(meta['ptc.a_value']['tags'], {'a_common', 'a_only_output'})

        self.assertSetEqual(meta['ptc.b']['tags'], {'b_common', 'b_only_input'})
        self.assertSetEqual(meta['ptc.b_value']['tags'], {'b_common', 'b_only_output'})


if __name__ == '__main__':
    unittest.main()

