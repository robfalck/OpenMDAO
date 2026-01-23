"""
Tests for to_spec() - converting concrete OpenMDAO systems to specs.
"""
import unittest
from openmdao.api import Problem, Group, ExecComp
from openmdao.specs import to_spec, instantiate_from_spec, ExecCompSpec, GroupSpec


class TestExecCompToSpec(unittest.TestCase):
    """Test conversion of ExecComp to ExecCompSpec."""

    def test_simple_execcomp(self):
        """Test converting a simple ExecComp to spec."""
        # Must be part of a Problem to have metadata populated
        prob = Problem()
        prob.model.add_subsystem('comp', ExecComp("y = 2.0 * x"))
        prob.setup()

        comp = prob.model.comp
        spec = to_spec(comp)

        self.assertIsInstance(spec, ExecCompSpec)
        self.assertEqual(spec.exprs, ["y = 2.0 * x"])

    def test_execcomp_with_shape(self):
        """Test converting ExecComp with array variables."""
        prob = Problem()
        prob.model.add_subsystem(
            'comp',
            ExecComp(
                "obj = x**2 + z[1]",
                z={'shape': (2,)},
                x={}
            )
        )
        prob.setup()

        comp = prob.model.comp
        spec = to_spec(comp)

        self.assertIsInstance(spec, ExecCompSpec)
        self.assertEqual(len(spec.inputs), 2)
        self.assertEqual(len(spec.outputs), 1)

        # Check that z has shape (2,)
        z_spec = next((inp for inp in spec.inputs if inp.name == 'z'), None)
        self.assertIsNotNone(z_spec)
        self.assertEqual(z_spec.shape, (2,))

    def test_execcomp_multiple_exprs(self):
        """Test ExecComp with multiple expressions."""
        prob = Problem()
        prob.model.add_subsystem(
            'comp',
            ExecComp([
                "y1 = 2.0 * x",
                "y2 = 3.0 * x"
            ])
        )
        prob.setup()

        comp = prob.model.comp
        spec = to_spec(comp)

        self.assertIsInstance(spec, ExecCompSpec)
        self.assertEqual(len(spec.exprs), 2)


class TestGroupToSpec(unittest.TestCase):
    """Test conversion of Group to GroupSpec."""

    def test_simple_group(self):
        """Test converting a simple group to spec."""
        prob = Problem()
        prob.model.add_subsystem('comp1', ExecComp("y = 2.0 * x"))
        prob.model.add_subsystem('comp2', ExecComp("z = 3.0 * x"))
        prob.setup()

        spec = to_spec(prob.model)

        self.assertIsInstance(spec, GroupSpec)
        self.assertEqual(len(spec.subsystems), 2)

    def test_group_with_promoted_inputs(self):
        """Test group with promoted inputs."""
        prob = Problem()
        prob.model.add_subsystem(
            'comp1',
            ExecComp("y = 2.0 * x"),
            promotes_inputs=['x']
        )
        prob.model.add_subsystem(
            'comp2',
            ExecComp("z = 3.0 * x"),
            promotes_inputs=['x']
        )
        prob.setup()

        spec = to_spec(prob.model)

        self.assertIsInstance(spec, GroupSpec)
        self.assertEqual(len(spec.subsystems), 2)
        # Check that promotions were captured
        self.assertIn('x', spec.subsystems[0].promotes_inputs)
        self.assertIn('x', spec.subsystems[1].promotes_inputs)

    def test_group_with_connections(self):
        """Test group with explicit connections."""
        prob = Problem()
        prob.model.add_subsystem('comp1', ExecComp("y = 2.0 * x"))
        prob.model.add_subsystem('comp2', ExecComp("z = 3.0 * y"))
        prob.model.connect('comp1.y', 'comp2.y')
        prob.setup()

        spec = to_spec(prob.model)

        self.assertIsInstance(spec, GroupSpec)
        self.assertEqual(len(spec.connections), 1)
        # Check connection exists (order and exact format may vary)
        conn = spec.connections[0]
        self.assertIn('comp1.y', [conn.src, conn.tgt])
        self.assertIn('comp2.y', [conn.src, conn.tgt])


class TestRoundTrip(unittest.TestCase):
    """Test round-trip conversion: spec -> system -> spec."""

    def test_roundtrip_simple_execcomp(self):
        """Test round-trip for simple ExecComp."""
        # Create original spec
        original_spec = ExecCompSpec(
            exprs=["y = 2.0 * x"],
            inputs=[],
            outputs=[]
        )

        # Instantiate from spec in a Problem
        prob = Problem()
        prob.model.add_subsystem('comp', instantiate_from_spec(original_spec))
        prob.setup()

        comp = prob.model.comp

        # Convert back to spec
        recovered_spec = to_spec(comp)

        # Check that expressions match
        self.assertEqual(original_spec.exprs, recovered_spec.exprs)

    def test_roundtrip_execcomp_with_inputs_outputs(self):
        """Test round-trip for ExecComp with variable metadata."""
        from openmdao.specs import VariableSpec

        original_spec = ExecCompSpec(
            exprs=["obj = x**2 + z[1]"],
            inputs=[
                VariableSpec(name='x'),
                VariableSpec(name='z', shape=(2,))
            ],
            outputs=[VariableSpec(name='obj')]
        )

        # Instantiate from spec in a Problem
        prob = Problem()
        prob.model.add_subsystem('comp', instantiate_from_spec(original_spec))
        prob.setup()

        comp = prob.model.comp

        # Convert back to spec
        recovered_spec = to_spec(comp)

        # Check basic properties
        self.assertEqual(original_spec.exprs, recovered_spec.exprs)
        self.assertEqual(len(original_spec.inputs), len(recovered_spec.inputs))

        # Check that z shape is preserved
        z_original = next((inp for inp in original_spec.inputs if inp.name == 'z'), None)
        z_recovered = next((inp for inp in recovered_spec.inputs if inp.name == 'z'), None)
        self.assertIsNotNone(z_original)
        self.assertIsNotNone(z_recovered)
        self.assertEqual(z_original.shape, z_recovered.shape)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in to_spec()."""

    def test_unsupported_component_type(self):
        """Test that unsupported component types raise helpful error."""
        from openmdao.core.indepvarcomp import IndepVarComp

        comp = IndepVarComp('x', 1.0)

        with self.assertRaises(ValueError) as cm:
            to_spec(comp)

        self.assertIn("No spec registered", str(cm.exception))
        self.assertIn("IndepVarComp", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
