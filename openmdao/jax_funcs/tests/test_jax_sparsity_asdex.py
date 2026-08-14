"""
Tests for asdex based jacobian sparsity detection on jax components.

The asdex path determines sparsity by abstract interpretation of the jaxpr, which yields a
global pattern rather than one sampled at a point.  These tests cover pattern correctness for
explicit and implicit components, point independence (the correctness bug that motivated the
feature), the dense fallback, dynamic shapes, and configuration/fallback behavior.
"""
import sys
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.jax_utils import _asdex_available

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None

ASDEX = _asdex_available()
asdex_required = unittest.skipUnless(ASDEX and jax is not None,
                                     'asdex and jax are required for these tests')
jax_required = unittest.skipUnless(jax is not None, 'jax is required for these tests')


# ---------------------------------------------------------------------------------------
# test components
# ---------------------------------------------------------------------------------------

class ElementwiseComp(om.JaxExplicitComponent):
    """y = x * z, elementwise. dy/dx and dy/dz are both diagonal."""

    def initialize(self):
        self.options.declare('vec_size', default=5, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', shape=(n,))
        self.add_input('z', shape=(n,))
        self.add_output('y', shape=(n,))

    def compute_primal(self, x, z):
        return x * z


class MixedScalarVectorComp(om.JaxExplicitComponent):
    """y = x * s, with s scalar. dy/dx diagonal, dy/ds a dense column."""

    def initialize(self):
        self.options.declare('vec_size', default=5, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', shape=(n,))
        self.add_input('s', shape=())
        self.add_output('y', shape=(n,))

    def compute_primal(self, x, s):
        return x * s


class DenseComp(om.JaxExplicitComponent):
    """Every output depends on every input."""

    def initialize(self):
        self.options.declare('vec_size', default=4, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', shape=(n,))
        self.add_output('y', shape=(n,))

    def compute_primal(self, x):
        return jnp.sum(x) * jnp.ones(self.options['vec_size'])


class VanishingDerivComp(om.JaxExplicitComponent):
    """
    y = v * (1 + b * s).

    dy/db = v * s vanishes numerically at s == 0 while remaining structurally nonzero.
    A point sampled detector at s == 0 concludes y does not depend on b.
    """

    def initialize(self):
        self.options.declare('vec_size', default=5, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('v', shape=(n,))
        self.add_input('b', shape=(n,))
        self.add_input('s', shape=(), val=0.0)
        self.add_output('y', shape=(n,))

    def compute_primal(self, v, b, s):
        return v * (1.0 + b * s)


class ElementwiseImplicitComp(om.JaxImplicitComponent):
    """R = y**2 - x, elementwise. dR/dy and dR/dx are both diagonal."""

    def initialize(self):
        self.options.declare('vec_size', default=5, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', shape=(n,))
        self.add_output('y', shape=(n,), val=np.ones(n))

    def compute_primal(self, x, y):
        return y ** 2 - x


class MixedImplicitComp(om.JaxImplicitComponent):
    """R = y - x * s, with s scalar, so the input and state blocks differ structurally."""

    def initialize(self):
        self.options.declare('vec_size', default=5, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', shape=(n,))
        self.add_input('s', shape=(), val=2.0)
        self.add_output('y', shape=(n,), val=np.ones(n))

    def compute_primal(self, x, s, y):
        return y - x * s


def _build(comp, **kwargs):
    p = om.Problem()
    p.model.add_subsystem('comp', comp, promotes=['*'])
    p.setup(**kwargs)
    p.final_setup()
    return p


def _subjac(comp, of, wrt):
    return comp._subjacs_info[(f'comp.{of}', f'comp.{wrt}')]


def _pattern(comp, of, wrt, shape):
    """
    Return the dense boolean pattern of a subjac.

    Detected sparsity is recorded under the 'sparsity' key as (rows, cols, shape); 'rows'/'cols'
    hold user declared sparsity.  A subjac with neither is dense.
    """
    meta = _subjac(comp, of, wrt)
    dense = np.zeros(shape, dtype=bool)
    if meta.get('sparsity') is not None:
        rows, cols, _ = meta['sparsity']
        dense[np.asarray(rows), np.asarray(cols)] = True
    elif meta['rows'] is not None:
        dense[meta['rows'], meta['cols']] = True
    else:
        dense[:, :] = True
    return dense


def _is_dense(comp, of, wrt):
    """True if the subjac carries no sparsity, detected or declared."""
    meta = _subjac(comp, of, wrt)
    return meta.get('sparsity') is None and meta['rows'] is None


# ---------------------------------------------------------------------------------------
# A. pattern correctness
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexPatternCorrectness(unittest.TestCase):

    def test_elementwise_explicit_is_diagonal(self):
        n = 6
        comp = ElementwiseComp(vec_size=n)
        p = _build(comp)  # keep ref alive
        sparsity, info = comp.compute_sparsity()

        self.assertEqual(info['method'], 'asdex')
        self.assertFalse(info['dense'])
        for wrt in ('x', 'z'):
            np.testing.assert_array_equal(_pattern(comp, 'y', wrt, (n, n)), np.eye(n, dtype=bool))

    def test_mixed_scalar_vector(self):
        n = 6
        comp = MixedScalarVectorComp(vec_size=n)
        p = _build(comp)  # keep ref alive
        comp.compute_sparsity()

        np.testing.assert_array_equal(_pattern(comp, 'y', 'x', (n, n)), np.eye(n, dtype=bool))
        # scalar column is fully dependent
        np.testing.assert_array_equal(_pattern(comp, 'y', 's', (n, 1)),
                                      np.ones((n, 1), dtype=bool))

    def test_elementwise_implicit_is_diagonal(self):
        n = 6
        comp = ElementwiseImplicitComp(vec_size=n)
        p = _build(comp)  # keep ref alive
        sparsity, info = comp.compute_sparsity()

        self.assertEqual(info['method'], 'asdex')
        for wrt in ('x', 'y'):
            np.testing.assert_array_equal(_pattern(comp, 'y', wrt, (n, n)), np.eye(n, dtype=bool))

    def test_implicit_column_order_outputs_first(self):
        """
        OpenMDAO orders wrt columns as outputs then inputs, compute_primal takes inputs then
        outputs.  R = y - x*s has an identity wrt y and -s*I wrt x, so a swapped block ordering
        would still look diagonal; the scalar s column is what disambiguates.
        """
        n = 5
        comp = MixedImplicitComp(vec_size=n)
        p = _build(comp)  # keep ref alive
        comp.compute_sparsity()

        np.testing.assert_array_equal(_pattern(comp, 'y', 'y', (n, n)), np.eye(n, dtype=bool))
        np.testing.assert_array_equal(_pattern(comp, 'y', 'x', (n, n)), np.eye(n, dtype=bool))
        np.testing.assert_array_equal(_pattern(comp, 'y', 's', (n, 1)),
                                      np.ones((n, 1), dtype=bool))


# ---------------------------------------------------------------------------------------
# B. point independence -- the correctness bug
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexPointIndependence(unittest.TestCase):

    def test_pattern_identical_regardless_of_sample_point(self):
        patterns = {}
        for s_val in (0.0, 0.5):
            comp = VanishingDerivComp(vec_size=5)
            p = _build(comp)
            p.set_val('s', s_val)
            sparsity, _ = comp.compute_sparsity()
            patterns[s_val] = sparsity.toarray()

        np.testing.assert_array_equal(patterns[0.0], patterns[0.5])

    def test_structurally_nonzero_dependency_is_kept_at_vanishing_point(self):
        """dy/db vanishes numerically at s=0 but must remain in the pattern."""
        n = 5
        comp = VanishingDerivComp(vec_size=n)
        p = _build(comp)
        p.set_val('s', 0.0)
        comp.compute_sparsity()

        np.testing.assert_array_equal(_pattern(comp, 'y', 'b', (n, n)), np.eye(n, dtype=bool))

    def test_derivatives_correct_after_moving_off_vanishing_point(self):
        """
        Sparsity is detected at s=0, where dy/db is numerically zero, then the model is
        evaluated at s != 0.  If the dependency had been dropped the derivative would be wrong.
        """
        n = 4
        comp = VanishingDerivComp(vec_size=n)
        p = _build(comp, force_alloc_complex=False)
        p.set_val('s', 0.0)
        comp.compute_sparsity()

        p.set_val('v', np.arange(1., n + 1.))
        p.set_val('b', np.full(n, 3.0))
        p.set_val('s', 0.5)
        p.run_model()

        data = p.check_partials(out_stream=None, method='fd')
        assert_check_partials(data, atol=1e-5, rtol=1e-5)

        J = p.compute_totals(of=['y'], wrt=['b'], return_format='array')
        # dy/db = v * s
        np.testing.assert_allclose(np.diag(J), np.arange(1., n + 1.) * 0.5, rtol=1e-6)


# ---------------------------------------------------------------------------------------
# C. efficacy -- structure is independent of array size
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexEfficacy(unittest.TestCase):

    def test_nnz_scales_linearly_not_quadratically(self):
        """The regression that motivated this: elementwise comps were getting dense subjacs."""
        for n in (10, 200):
            comp = ElementwiseComp(vec_size=n)
            p = _build(comp)  # keep ref alive
            sparsity, info = comp.compute_sparsity()
            # 2 diagonals (wrt x and wrt z), not 2 * n**2
            self.assertEqual(sparsity.nnz, 2 * n)
            self.assertAlmostEqual(info['density'], 2 * n / (n * 2 * n), places=10)

    def test_detected_structure_is_size_independent(self):
        densities = []
        for n in (10, 50):
            comp = ElementwiseComp(vec_size=n)
            p = _build(comp)  # keep ref alive
            _, info = comp.compute_sparsity()
            densities.append(info['nz_entries'] / n)
        # nonzeros per row is the same regardless of n
        self.assertEqual(densities[0], densities[1])


# ---------------------------------------------------------------------------------------
# D. dense fallback
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexDenseFallback(unittest.TestCase):

    def test_dense_jacobian_left_dense(self):
        n = 4
        comp = DenseComp(vec_size=n)
        p = _build(comp)  # keep ref alive
        sparsity, info = comp.compute_sparsity()

        self.assertEqual(info['density'], 1.0)
        self.assertTrue(info['dense'])
        # subjac left dense: no detected sparsity applied
        self.assertTrue(_is_dense(comp, 'y', 'x'))

    def test_threshold_is_configurable(self):
        """A diagonal pattern is sparse, but a threshold below its density forces dense."""
        n = 6
        comp = ElementwiseComp(vec_size=n)
        comp.options['sparsity_density_threshold'] = 1e-6
        p = _build(comp)  # keep ref alive
        _, info = comp.compute_sparsity()

        self.assertTrue(info['dense'])
        self.assertTrue(_is_dense(comp, 'y', 'x'))

    def test_dense_component_still_gives_correct_derivatives(self):
        n = 4
        comp = DenseComp(vec_size=n)
        p = _build(comp)
        p.set_val('x', np.arange(1., n + 1.))
        p.run_model()
        assert_check_partials(p.check_partials(out_stream=None, method='fd'),
                              atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------------------
# E. dynamic shapes -- the reason detection lives in setup_partials
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexDynamicShapes(unittest.TestCase):

    def test_size_known_only_at_setup(self):
        for n in (3, 17):
            comp = ElementwiseComp(vec_size=n)
            p = _build(comp)  # keep ref alive
            sparsity, _ = comp.compute_sparsity()
            self.assertEqual(sparsity.shape, (n, 2 * n))

    def test_two_instances_different_sizes(self):
        p = om.Problem()
        p.model.add_subsystem('a', ElementwiseComp(vec_size=4))
        p.model.add_subsystem('b', ElementwiseComp(vec_size=9))
        p.setup()
        p.final_setup()

        sa, _ = p.model.a.compute_sparsity()
        sb, _ = p.model.b.compute_sparsity()
        self.assertEqual(sa.shape, (4, 8))
        self.assertEqual(sb.shape, (9, 18))


# ---------------------------------------------------------------------------------------
# F. configuration and fallback
# ---------------------------------------------------------------------------------------

@jax_required
class TestAsdexConfiguration(unittest.TestCase):

    def test_perturb_method_does_not_use_asdex(self):
        comp = ElementwiseComp(vec_size=5)
        comp.options['sparsity_method'] = 'perturb'
        p = _build(comp)  # keep ref alive
        _, info = comp.compute_sparsity()
        self.assertNotEqual(info.get('method'), 'asdex')

    @unittest.skipIf(ASDEX, 'only meaningful when asdex is absent')
    def test_explicit_asdex_request_raises_when_missing(self):
        comp = ElementwiseComp(vec_size=5)
        comp.options['sparsity_method'] = 'asdex'
        p = _build(comp)  # keep ref alive
        with self.assertRaises(RuntimeError) as ctx:
            comp.compute_sparsity()
        self.assertIn('asdex', str(ctx.exception))

    def test_lazy_import(self):
        """
        asdex pulls in numba, which is expensive.  A model with no jax components must not
        import it.  This guards against a stray module scope import.
        """
        code = (
            'import sys\n'
            'import openmdao.api as om\n'
            'import numpy as np\n'
            'p = om.Problem()\n'
            'p.model.add_subsystem("c", om.ExecComp("y = 2.0 * x"))\n'
            'p.setup(); p.final_setup(); p.run_model()\n'
            'assert "asdex" not in sys.modules, "asdex was imported by a non-jax model"\n'
            'print("OK")\n'
        )
        import subprocess
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertIn('OK', r.stdout, msg=r.stderr)


# ---------------------------------------------------------------------------------------
# G. interaction with declared partials and matrix_free
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexInteractions(unittest.TestCase):

    def test_user_declared_partials_are_not_discarded(self):
        """Declaring partials in setup_partials must be visible to the framework."""
        n = 5

        class DeclaringComp(ElementwiseComp):
            def setup_partials(self):
                ar = np.arange(self.options['vec_size'])
                self.declare_partials('y', ['x', 'z'], rows=ar, cols=ar)

        comp = DeclaringComp(vec_size=n)
        p = _build(comp)  # keep ref alive

        # the declaration survived, i.e. the subjacs are sparse diagonals
        for wrt in ('x', 'z'):
            meta = _subjac(comp, 'y', wrt)
            self.assertIsNotNone(meta['rows'])
            np.testing.assert_array_equal(meta['rows'], np.arange(n))
            np.testing.assert_array_equal(meta['cols'], np.arange(n))

    def test_matrix_free_skips_asdex(self):
        comp = ElementwiseComp(vec_size=5, matrix_free=True)
        p = _build(comp)  # keep ref alive
        self.assertTrue(comp.matrix_free)
        self.assertFalse(comp._do_sparsity)


# ---------------------------------------------------------------------------------------
# H. end to end numerical agreement
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexNumerics(unittest.TestCase):

    def test_explicit_matches_perturb_path(self):
        n = 5
        results = {}
        for method in ('asdex', 'perturb'):
            comp = ElementwiseComp(vec_size=n)
            comp.options['sparsity_method'] = method
            p = _build(comp)
            p.set_val('x', np.arange(1., n + 1.))
            p.set_val('z', np.linspace(2., 3., n))
            p.run_model()
            results[method] = p.compute_totals(of=['y'], wrt=['x', 'z'], return_format='array')
            assert_check_partials(p.check_partials(out_stream=None, method='fd'),
                                  atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(results['asdex'], results['perturb'], rtol=1e-12)

    def test_implicit_with_newton_and_direct_solver(self):
        n = 4
        p = om.Problem()
        comp = p.model.add_subsystem('comp', ElementwiseImplicitComp(vec_size=n),
                                     promotes=['*'])
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.set_val('x', np.array([4., 9., 16., 25.]))
        p.run_model()

        assert_near_equal(p.get_val('y'), np.array([2., 3., 4., 5.]), tolerance=1e-8)
        assert_check_partials(p.check_partials(out_stream=None, method='fd'),
                              atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------------------
# I. sparsity reaches subjac storage (declared during setup_partials)
# ---------------------------------------------------------------------------------------

@asdex_required
class TestAsdexDeclaresSparseStorage(unittest.TestCase):
    """
    Detection has to happen in setup_partials and go through declare_partials.  Recording a
    pattern later, once subjacs exist, does not change how they are stored, so the assembled
    jacobian would stay dense.
    """

    def test_elementwise_subjacs_are_stored_sparse(self):
        n = 8
        comp = ElementwiseComp(vec_size=n)
        p = _build(comp)
        for wrt in ('x', 'z'):
            meta = _subjac(comp, 'y', wrt)
            self.assertIsNotNone(meta['rows'],
                                 f"subjac (y, {wrt}) should be stored sparse")
            np.testing.assert_array_equal(meta['rows'], np.arange(n))
            np.testing.assert_array_equal(meta['cols'], np.arange(n))

    def test_assembled_jacobian_is_sparse(self):
        """The end to end payoff: an assembled jacobian with O(n) rather than O(n**2) nonzeros."""
        n = 20
        p = om.Problem()
        p.model.add_subsystem('comp', ElementwiseComp(vec_size=n), promotes=['*'])
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        p.setup()
        p.set_val('x', np.arange(1., n + 1.))
        p.set_val('z', np.ones(n))
        p.run_model()
        p.model._linearize(None)

        mtx = p.model.linear_solver._assembled_jac._dr_do_mtx._matrix
        # n identity entries for y itself plus n for dy/dx and n for dy/dz within the block
        self.assertLess(mtx.nnz, n * n,
                        "assembled jacobian should not be dense")

    def test_unused_input_gets_no_subjac(self):
        """A structurally absent dependency should not be declared at all."""

        class PartialDepComp(om.JaxExplicitComponent):
            def setup(self):
                self.add_input('a', shape=(4,))
                self.add_input('unused', shape=(4,))
                self.add_output('y', shape=(4,))

            def compute_primal(self, a, unused):
                return a * 2.0

        comp = PartialDepComp()
        p = _build(comp)
        self.assertIn(('comp.y', 'comp.a'), comp._subjacs_info)
        self.assertNotIn(('comp.y', 'comp.unused'), comp._subjacs_info)

    def test_implicit_subjacs_are_stored_sparse(self):
        n = 7
        comp = ElementwiseImplicitComp(vec_size=n)
        p = _build(comp)
        for wrt in ('x', 'y'):
            meta = _subjac(comp, 'y', wrt)
            self.assertIsNotNone(meta['rows'],
                                 f"implicit subjac (y, {wrt}) should be stored sparse")
            np.testing.assert_array_equal(meta['rows'], np.arange(n))


# ---------------------------------------------------------------------------------------
# J. payoff gate -- detection must be skipped when it cannot help
# ---------------------------------------------------------------------------------------

class SingleRowComp(om.JaxExplicitComponent):
    """
    One scalar output depending on many inputs.

    A dense evaluation costs min(nrows, ncols) = 1 AD pass in reverse mode, so no sparsity
    exploitation can beat it and detection would be pure overhead.
    """

    def initialize(self):
        self.options.declare('vec_size', default=500, types=int)

    def setup(self):
        n = self.options['vec_size']
        self.add_input('x', shape=(n,))
        self.add_output('y', shape=())

    def compute_primal(self, x):
        return jnp.sum(x * x)


@asdex_required
class TestAsdexPayoffGate(unittest.TestCase):

    def test_single_row_jacobian_skips_detection(self):
        comp = SingleRowComp(vec_size=500)
        p = _build(comp)
        self.assertFalse(comp._use_asdex_sparsity(),
                         'detection should be skipped for a single row jacobian')

    def test_gate_applies_to_runtime_path_too(self):
        """
        There are two entry points into asdex, the setup time declaration and the runtime
        compute_sparsity call.  Both consult _use_asdex_sparsity, so a component that cannot
        benefit must not reach asdex through either one.
        """
        comp = SingleRowComp(vec_size=500)
        p = _build(comp)
        p.run_model()
        sparsity, info = comp.compute_sparsity()
        self.assertNotEqual(info.get('method'), 'asdex')

    def test_no_benefit_component_does_not_import_asdex(self):
        """asdex costs ~0.5s to import via numba, so a model that cannot use it must not pay."""
        code = chr(10).join([
            "import sys",
            "import numpy as np",
            "import openmdao.api as om",
            "from openmdao.jax_funcs.tests.test_jax_sparsity_asdex import SingleRowComp",
            "p = om.Problem()",
            'p.model.add_subsystem("c", SingleRowComp(vec_size=500), promotes=["*"])',
            "p.setup(); p.final_setup(); p.run_model()",
            'p.compute_totals(of=["y"], wrt=["x"])',
            'assert "asdex" not in sys.modules, "asdex imported for a no-benefit component"',
            'print("OK")',
        ])
        import subprocess
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertIn('OK', r.stdout, msg=r.stderr)

    def test_explicit_asdex_request_overrides_gate(self):
        """sparsity_method='asdex' forces detection even where the heuristic says skip."""
        comp = SingleRowComp(vec_size=50)
        comp.options['sparsity_method'] = 'asdex'
        p = _build(comp)
        self.assertTrue(comp._use_asdex_sparsity())

    def test_min_dim_is_configurable(self):
        comp = SingleRowComp(vec_size=500)
        comp.options['sparsity_min_dim'] = 1
        p = _build(comp)
        self.assertTrue(comp._use_asdex_sparsity())

    def test_gate_does_not_block_beneficial_components(self):
        comp = ElementwiseComp(vec_size=50)
        p = _build(comp)
        self.assertTrue(comp._use_asdex_sparsity())


if __name__ == '__main__':
    unittest.main()
