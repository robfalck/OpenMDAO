"""
Tests for PromotesSpec class.
"""
import unittest
from pydantic import ValidationError

from openmdao.specs.promotes_spec import PromotesSpec
from openmdao.specs.indices_spec import SrcIndicesSpec
from openmdao.specs.subsystem_spec import SubsystemSpec
from openmdao.specs.group_spec import GroupSpec


class TestPromotesSpec(unittest.TestCase):
    """Test cases for PromotesSpec."""

    def test_simple_promotion_string(self):
        """Test creating PromotesSpec with simple string name."""
        pspec = PromotesSpec(name='x', io_type='input')
        self.assertEqual(pspec.name, 'x')
        self.assertEqual(pspec.io_type, 'input')
        self.assertIsNone(pspec.src_indices)
        self.assertIsNone(pspec.src_shape)

    def test_promotion_with_renaming(self):
        """Test creating PromotesSpec with tuple name for renaming."""
        pspec = PromotesSpec(name=('old_x', 'new_x'), io_type='input')
        self.assertEqual(pspec.name, ('old_x', 'new_x'))
        self.assertEqual(pspec.io_type, 'input')

    def test_promotion_with_src_indices(self):
        """Test PromotesSpec with src_indices."""
        src_indices_spec = SrcIndicesSpec(value=[0, 1, 2])
        pspec = PromotesSpec(
            name='z',
            io_type='input',
            src_indices=src_indices_spec
        )
        self.assertEqual(pspec.name, 'z')
        self.assertIsNotNone(pspec.src_indices)
        # Compare arrays using list conversion to avoid numpy ambiguity
        import numpy as np
        if isinstance(pspec.src_indices.value, np.ndarray):
            self.assertTrue(np.array_equal(pspec.src_indices.value, [0, 1, 2]))
        else:
            self.assertEqual(list(pspec.src_indices.value), [0, 1, 2])

    def test_promotion_with_src_shape(self):
        """Test PromotesSpec with src_shape."""
        pspec = PromotesSpec(
            name='arr',
            io_type='input',
            src_shape=(3, 2)
        )
        self.assertEqual(pspec.src_shape, (3, 2))

    def test_promotion_with_src_shape_int(self):
        """Test PromotesSpec with src_shape as integer (normalized to tuple)."""
        pspec = PromotesSpec(
            name='arr',
            io_type='input',
            src_shape=5
        )
        self.assertEqual(pspec.src_shape, (5,))

    def test_src_indices_not_for_outputs(self):
        """Test that src_indices raises error for outputs."""
        src_indices_spec = SrcIndicesSpec(value=[0, 1])
        with self.assertRaises(ValidationError):
            PromotesSpec(
                name='z',
                io_type='output',
                src_indices=src_indices_spec
            )

    def test_src_shape_not_for_outputs(self):
        """Test that src_shape raises error for outputs."""
        with self.assertRaises(ValidationError):
            PromotesSpec(
                name='arr',
                io_type='output',
                src_shape=(2, 3)
            )

    def test_valid_single_char_name(self):
        """Test that single character names are valid."""
        pspec = PromotesSpec(name='x')
        self.assertEqual(pspec.name, 'x')

    def test_promotion_with_subsys_name(self):
        """Test PromotesSpec with subsys_name."""
        src_indices = SrcIndicesSpec(value=[0, 1, 2])
        pspec = PromotesSpec(
            name='z',
            io_type='input',
            subsys_name='comp1',
            src_indices=src_indices
        )
        self.assertEqual(pspec.name, 'z')
        self.assertEqual(pspec.subsys_name, 'comp1')
        self.assertEqual(pspec.io_type, 'input')

    def test_promotion_with_none_subsys_name(self):
        """Test PromotesSpec with None subsys_name (current system)."""
        pspec = PromotesSpec(name='x', io_type='input', subsys_name=None)
        self.assertEqual(pspec.name, 'x')
        self.assertIsNone(pspec.subsys_name)

    def test_invalid_name_tuple_wrong_length(self):
        """Test that tuple name with wrong length is invalid."""
        with self.assertRaises(ValidationError):
            PromotesSpec(name=('a', 'b', 'c'))

    def test_invalid_name_tuple_non_strings(self):
        """Test that tuple name with non-strings is invalid."""
        with self.assertRaises(ValidationError):
            PromotesSpec(name=('old_x', 123))

    def test_default_io_type_is_any(self):
        """Test that default io_type is 'any'."""
        pspec = PromotesSpec(name='x')
        self.assertEqual(pspec.io_type, 'any')

    def test_serialization_string_name(self):
        """Test serialization of PromotesSpec with string name."""
        pspec = PromotesSpec(name='x', io_type='input')
        data = pspec.model_dump()
        self.assertEqual(data['name'], 'x')
        self.assertEqual(data['io_type'], 'input')

    def test_serialization_tuple_name(self):
        """Test serialization of PromotesSpec with tuple name."""
        pspec = PromotesSpec(name=('old_x', 'new_x'), io_type='input')
        data = pspec.model_dump()
        self.assertEqual(data['name'], ('old_x', 'new_x'))

    def test_deserialization_string_name(self):
        """Test deserialization of PromotesSpec with string name."""
        data = {'name': 'x', 'io_type': 'input'}
        pspec = PromotesSpec.model_validate(data)
        self.assertEqual(pspec.name, 'x')
        self.assertEqual(pspec.io_type, 'input')

    def test_deserialization_tuple_name(self):
        """Test deserialization of PromotesSpec with tuple name."""
        data = {'name': ('old_x', 'new_x'), 'io_type': 'input'}
        pspec = PromotesSpec.model_validate(data)
        self.assertEqual(pspec.name, ('old_x', 'new_x'))

    def test_serialization_with_subsys_name(self):
        """Test serialization of PromotesSpec with subsys_name."""
        pspec = PromotesSpec(name='x', io_type='input', subsys_name='comp1')
        data = pspec.model_dump()
        self.assertEqual(data['name'], 'x')
        self.assertEqual(data['subsys_name'], 'comp1')

    def test_deserialization_with_subsys_name(self):
        """Test deserialization of PromotesSpec with subsys_name."""
        data = {'name': 'x', 'io_type': 'input', 'subsys_name': 'comp1'}
        pspec = PromotesSpec.model_validate(data)
        self.assertEqual(pspec.name, 'x')
        self.assertEqual(pspec.subsys_name, 'comp1')


class TestSubsystemSpecBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility of SubsystemSpec with legacy format."""

    def test_legacy_string_list_promotes(self):
        """Test that legacy string list format is converted to PromotesSpec."""
        spec_dict = {
            'name': 'comp1',
            'system': {'type': 'exec_comp', 'exprs': 'y=x', 'inputs': [], 'outputs': []},
            'promotes': ['x', 'y']
        }
        subsys = SubsystemSpec.model_validate(spec_dict)
        self.assertEqual(len(subsys.promotes), 2)
        self.assertIsInstance(subsys.promotes[0], PromotesSpec)
        self.assertEqual(subsys.promotes[0].name, 'x')
        self.assertEqual(subsys.promotes[0].io_type, 'any')

    def test_legacy_tuple_list_promotes(self):
        """Test that legacy tuple list format is converted to PromotesSpec."""
        spec_dict = {
            'name': 'comp1',
            'system': {'type': 'exec_comp', 'exprs': 'y=x', 'inputs': [], 'outputs': []},
            'promotes': [('old_x', 'new_x'), ('old_y', 'new_y')]
        }
        subsys = SubsystemSpec.model_validate(spec_dict)
        self.assertEqual(len(subsys.promotes), 2)
        self.assertIsInstance(subsys.promotes[0], PromotesSpec)
        self.assertEqual(subsys.promotes[0].name, ('old_x', 'new_x'))

    def test_legacy_string_list_promotes_inputs(self):
        """Test legacy promotes_inputs string list conversion."""
        spec_dict = {
            'name': 'comp1',
            'system': {'type': 'exec_comp', 'exprs': 'y=x', 'inputs': [], 'outputs': []},
            'promotes_inputs': ['x']
        }
        subsys = SubsystemSpec.model_validate(spec_dict)
        self.assertEqual(len(subsys.promotes_inputs), 1)
        self.assertIsInstance(subsys.promotes_inputs[0], PromotesSpec)
        self.assertEqual(subsys.promotes_inputs[0].name, 'x')
        self.assertEqual(subsys.promotes_inputs[0].io_type, 'input')

    def test_legacy_string_list_promotes_outputs(self):
        """Test legacy promotes_outputs string list conversion."""
        spec_dict = {
            'name': 'comp1',
            'system': {'type': 'exec_comp', 'exprs': 'y=x', 'inputs': [], 'outputs': []},
            'promotes_outputs': ['y']
        }
        subsys = SubsystemSpec.model_validate(spec_dict)
        self.assertEqual(len(subsys.promotes_outputs), 1)
        self.assertIsInstance(subsys.promotes_outputs[0], PromotesSpec)
        self.assertEqual(subsys.promotes_outputs[0].name, 'y')
        self.assertEqual(subsys.promotes_outputs[0].io_type, 'output')

    def test_new_promotes_spec_list(self):
        """Test that new PromotesSpec list format works."""
        pspec = PromotesSpec(name='x', io_type='input')
        subsys_dict = {
            'name': 'comp1',
            'system': {'type': 'exec_comp', 'exprs': 'y=x', 'inputs': [], 'outputs': []},
            'promotes_inputs': [pspec]
        }
        subsys = SubsystemSpec.model_validate(subsys_dict)
        self.assertEqual(len(subsys.promotes_inputs), 1)
        self.assertEqual(subsys.promotes_inputs[0].name, 'x')


class TestGroupSpecPromotes(unittest.TestCase):
    """Test GroupSpec promotes field."""

    def test_group_spec_with_promotes(self):
        """Test GroupSpec with promotes field containing PromotesSpec with subsys_name."""
        pspec = PromotesSpec(name='x', io_type='input', subsys_name='comp1')

        group_dict = {
            'type': 'group',
            'subsystems': [],
            'promotes': [pspec]
        }
        group = GroupSpec.model_validate(group_dict)
        self.assertEqual(len(group.promotes), 1)
        self.assertEqual(group.promotes[0].subsys_name, 'comp1')

    def test_group_spec_promotes_empty_default(self):
        """Test that promotes defaults to empty list."""
        group = GroupSpec()
        self.assertEqual(group.promotes, [])

    def test_group_spec_multiple_promotes(self):
        """Test GroupSpec with multiple PromotesSpec for different subsystems."""
        pspec1 = PromotesSpec(name='x', io_type='input', subsys_name='comp1')
        pspec2 = PromotesSpec(name='y', io_type='output', subsys_name='comp2')

        group_dict = {
            'type': 'group',
            'subsystems': [],
            'promotes': [pspec1, pspec2]
        }
        group = GroupSpec.model_validate(group_dict)
        self.assertEqual(len(group.promotes), 2)
        self.assertEqual(group.promotes[0].subsys_name, 'comp1')
        self.assertEqual(group.promotes[1].subsys_name, 'comp2')

    def test_group_spec_promotes_without_subsys_name(self):
        """Test PromotesSpec in GroupSpec with None subsys_name (applies to current system)."""
        pspec = PromotesSpec(name='x', io_type='input', subsys_name=None)

        group_dict = {
            'type': 'group',
            'subsystems': [],
            'promotes': [pspec]
        }
        group = GroupSpec.model_validate(group_dict)
        self.assertEqual(len(group.promotes), 1)
        self.assertIsNone(group.promotes[0].subsys_name)


class TestPromotesSpecWithIndices(unittest.TestCase):
    """Test PromotesSpec integration with SrcIndicesSpec."""

    def test_promotes_with_src_indices_spec(self):
        """Test PromotesSpec with SrcIndicesSpec."""
        src_indices = SrcIndicesSpec(value=[0, 2, 4])
        pspec = PromotesSpec(
            name='z',
            io_type='input',
            src_indices=src_indices
        )
        # Compare arrays properly
        import numpy as np
        if isinstance(pspec.src_indices.value, np.ndarray):
            self.assertTrue(np.array_equal(pspec.src_indices.value, [0, 2, 4]))
        else:
            self.assertEqual(list(pspec.src_indices.value), [0, 2, 4])

    def test_promotes_serialization_with_indices(self):
        """Test serialization of PromotesSpec with indices."""
        src_indices = SrcIndicesSpec(value=[0, 1])
        pspec = PromotesSpec(
            name='x',
            io_type='input',
            src_indices=src_indices
        )
        data = pspec.model_dump()
        self.assertIn('src_indices', data)
        self.assertIsNotNone(data['src_indices'])


if __name__ == '__main__':
    unittest.main()
