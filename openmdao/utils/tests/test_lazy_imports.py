import unittest
import importlib.util


class TestLazyImport(unittest.TestCase):
    """Tests for the LazyImport class."""

    def setUp(self):
        """Clear any cached modules that might interfere with tests."""
        # We'll use modules unlikely to be imported by OpenMDAO
        pass

    def test_lazy_import_available_module(self):
        """Test that LazyImport works for an available module."""
        from openmdao.utils.lazy_imports import LazyModule

        # Create lazy import for a module
        lazy_decimal = LazyModule('decimal')

        # The _module attribute should be None before any access
        self.assertIsNone(lazy_decimal._module)

        # Access an attribute - should trigger import
        result = lazy_decimal.Decimal('3.14')

        # Now the _module attribute should be populated
        self.assertIsNotNone(lazy_decimal._module)

        # Verify it works
        self.assertEqual(str(result), '3.14')

    def test_lazy_import_unavailable_module(self):
        """Test that LazyImport raises ModuleNotFoundError for unavailable modules."""
        from openmdao.utils.lazy_imports import LazyModule

        with self.assertRaises(ModuleNotFoundError) as cm:
            LazyModule('nonexistent_package_xyz_12345')

        self.assertIn('nonexistent_package_xyz_12345', str(cm.exception))

    def test_lazy_import_from_module_level_available(self):
        """Test importing available module from lazy_imports module."""
        # json is standard library, should always be available
        from openmdao.utils.lazy_imports import json

        # Should be a LazyImport instance
        from openmdao.utils.lazy_imports import LazyModule
        self.assertIsInstance(json, LazyModule)

        # Should not be loaded yet
        self.assertIsNone(json._module)

        # Access an attribute to trigger load
        result = json.dumps({'key': 'value'})
        self.assertEqual(result, '{"key": "value"}')

        # Now should be loaded
        self.assertIsNotNone(json._module)

    def test_lazy_import_from_module_level_unavailable(self):
        """Test that importing unavailable module from lazy_imports raises error."""
        with self.assertRaises(ModuleNotFoundError):
            from openmdao.utils.lazy_imports import fake_module_om  # noqa: F401

    def test_lazy_import_submodule(self):
        """Test lazy import of submodules."""
        from openmdao.utils.lazy_imports import LazyModule

        # Use a standard library submodule for testing
        lazy_sub = LazyModule('urllib.parse')

        # Should not be loaded yet
        self.assertIsNone(lazy_sub._module)

        # Access should work
        result = lazy_sub.quote('hello world')
        self.assertEqual(result, 'hello%20world')

        # Should be loaded now
        self.assertIsNotNone(lazy_sub._module)

    def test_lazy_import_caching(self):
        """Test that the module is only imported once."""
        from openmdao.utils.lazy_imports import LazyModule

        lazy_pathlib = LazyModule('pathlib')

        # Should not be loaded yet
        self.assertIsNone(lazy_pathlib._module)

        # First access
        path1 = lazy_pathlib.Path

        # Module should be loaded
        self.assertIsNotNone(lazy_pathlib._module)

        # Second access should use cached module
        path2 = lazy_pathlib.Path

        # Should be the same object
        self.assertIs(path1, path2)

        # _module should still be the same
        first_module = lazy_pathlib._module
        lazy_pathlib._load()
        self.assertIs(lazy_pathlib._module, first_module)

    def test_lazy_import_dir(self):
        """Test that __dir__ returns module attributes."""
        from openmdao.utils.lazy_imports import LazyModule

        lazy_sys = LazyModule('sys')

        # Should not be loaded yet
        self.assertIsNone(lazy_sys._module)

        # Calling dir should trigger import and return attributes
        attrs = dir(lazy_sys)

        # Should have standard sys module attributes
        self.assertIn('version', attrs)
        self.assertIn('platform', attrs)

        # Should be loaded now
        self.assertIsNotNone(lazy_sys._module)

    def test_lazy_import_repr(self):
        """Test that __repr__ works correctly."""
        from openmdao.utils.lazy_imports import LazyModule

        lazy_re = LazyModule('re')

        # Should not be loaded yet
        self.assertIsNone(lazy_re._module)

        # Get repr - this will trigger load
        repr_str = repr(lazy_re)

        # Should look like a module repr
        self.assertTrue('module' in repr_str.lower() or 're' in repr_str)

        # Should be loaded now
        self.assertIsNotNone(lazy_re._module)

    def test_lazy_import_bool(self):
        """Test that LazyImport instances are truthy."""
        from openmdao.utils.lazy_imports import LazyModule

        lazy_csv = LazyModule('csv')

        # Should be truthy (instances are truthy by default)
        self.assertTrue(lazy_csv)
        self.assertTrue(bool(lazy_csv))

    def test_lazy_import_multiple_attributes(self):
        """Test accessing multiple attributes from lazy import."""
        from openmdao.utils.lazy_imports import LazyModule

        lazy_random = LazyModule('random')

        # Should not be loaded yet
        self.assertIsNone(lazy_random._module)

        # Access multiple attributes
        lazy_random.seed(42)

        # Should be loaded now
        self.assertIsNotNone(lazy_random._module)

        val1 = lazy_random.random()

        lazy_random.seed(42)
        val2 = lazy_random.random()

        # Should get same random value with same seed
        self.assertEqual(val1, val2)

    def test_lazy_import_attribute_error(self):
        """Test that AttributeError is raised for non-existent attributes."""
        from openmdao.utils.lazy_imports import LazyModule

        lazy_string = LazyModule('string')

        # Should not be loaded yet
        self.assertIsNone(lazy_string._module)

        with self.assertRaises(AttributeError):
            _ = lazy_string.nonexistent_attribute_xyz_999

        # Should be loaded now (even though attribute doesn't exist)
        self.assertIsNotNone(lazy_string._module)

    def test_try_except_import_pattern_success(self):
        """Test that try/except ModuleNotFoundError pattern works for available modules."""
        # This should work - json is available
        try:
            from openmdao.utils.lazy_imports import json
            imported = True
        except ModuleNotFoundError:
            json = None
            imported = False

        self.assertTrue(imported)
        self.assertIsNotNone(json)

        # Verify it's a LazyImport
        from openmdao.utils.lazy_imports import LazyModule
        self.assertIsInstance(json, LazyModule)

    def test_try_except_import_pattern_failure(self):
        """Test that try/except ModuleNotFoundError pattern works for unavailable modules."""
        # This should fail - module doesn't exist
        try:
            from openmdao.utils.lazy_imports import fake_module_xyz_123_never_exists
            imported = True
            fake_module_xyz_123_never_exists = fake_module_xyz_123_never_exists  # suppress unused warning
        except ModuleNotFoundError:
            fake_module_xyz_123_never_exists = None
            imported = False

        self.assertFalse(imported)
        self.assertIsNone(fake_module_xyz_123_never_exists)

    def test_lazy_import_preserves_module_functionality(self):
        """Test that lazy imported modules work the same as regular imports."""
        from openmdao.utils.lazy_imports import LazyModule
        import collections

        lazy_collections = LazyModule('collections')

        # Should not be loaded yet
        self.assertIsNone(lazy_collections._module)

        # Create objects from both
        regular_counter = collections.Counter([1, 2, 2, 3, 3, 3])
        lazy_counter = lazy_collections.Counter([1, 2, 2, 3, 3, 3])

        self.assertEqual(regular_counter, lazy_counter)

        # Should be loaded now
        self.assertIsNotNone(lazy_collections._module)

    @unittest.skipIf(
        importlib.util.find_spec("numpy") is None,
        "numpy not installed"
    )
    def test_lazy_import_with_numpy(self):
        """Test lazy import with a real heavy package (numpy)."""
        from openmdao.utils.lazy_imports import LazyModule

        # Create lazy numpy
        lazy_np = LazyModule('numpy')

        # Should not be loaded yet
        self.assertIsNone(lazy_np._module)

        # Use it
        arr = lazy_np.array([1, 2, 3])

        # Should be loaded now
        self.assertIsNotNone(lazy_np._module)

        self.assertEqual(len(arr), 3)
        self.assertEqual(lazy_np.sum(arr), 6)

    @unittest.skipIf(
        importlib.util.find_spec("mpi4py") is None,
        "mpi4py not installed"
    )
    def test_lazy_import_mpi_via_module_getattr(self):
        """Test lazy import of MPI via module __getattr__."""
        from openmdao.utils.lazy_imports import MPI, LazyModule

        # Should be a LazyImport instance
        self.assertIsInstance(MPI, LazyModule)

        # Should be for mpi4py.MPI
        self.assertEqual(MPI._module_name, 'mpi4py.MPI')

        # Should not be loaded yet
        self.assertIsNone(MPI._module)

    @unittest.skipIf(
        importlib.util.find_spec("petsc4py") is None,
        "petsc4py not installed"
    )
    def test_lazy_import_petsc_via_module_getattr(self):
        """Test lazy import of PETSc via module __getattr__."""
        from openmdao.utils.lazy_imports import PETSc, LazyModule

        # Should be a LazyImport instance
        self.assertIsInstance(PETSc, LazyModule)

        # Should be for petsc4py.PETSc
        self.assertEqual(PETSc._module_name, 'petsc4py.PETSc')

        # Should not be loaded yet
        self.assertIsNone(PETSc._module)


class TestLazyImportModuleAttribute(unittest.TestCase):
    """Tests for the module-level __getattr__ function."""

    def test_getattr_creates_lazy_import(self):
        """Test that accessing module attributes creates LazyImport instances."""
        from openmdao.utils import lazy_imports
        from openmdao.utils.lazy_imports import LazyModule

        # Access an attribute dynamically
        lazy_datetime = getattr(lazy_imports, 'datetime')

        # Should be a LazyImport instance
        self.assertIsInstance(lazy_datetime, LazyModule)

        # Should not be loaded yet
        self.assertIsNone(lazy_datetime._module)

        # Should have correct module name
        self.assertEqual(lazy_datetime._module_name, 'datetime')

    @unittest.skipIf(
        importlib.util.find_spec("mpi4py") is None,
        "mpi4py not installed"
    )
    def test_getattr_special_case_mpi(self):
        """Test that MPI is handled specially."""
        from openmdao.utils import lazy_imports
        from openmdao.utils.lazy_imports import LazyModule

        # Access MPI attribute
        mpi = getattr(lazy_imports, 'MPI')

        # Should be a LazyImport instance
        self.assertIsInstance(mpi, LazyModule)

        # Should point to mpi4py.MPI, not just mpi4py
        self.assertEqual(mpi._module_name, 'mpi4py.MPI')

        # Should not be loaded yet
        self.assertIsNone(mpi._module)

    @unittest.skipIf(
        importlib.util.find_spec("petsc4py") is None,
        "petsc4py not installed"
    )
    def test_getattr_special_case_petsc(self):
        """Test that PETSc is handled specially."""
        from openmdao.utils import lazy_imports
        from openmdao.utils.lazy_imports import LazyModule

        # Access PETSc attribute
        petsc = getattr(lazy_imports, 'PETSc')

        # Should be a LazyImport instance
        self.assertIsInstance(petsc, LazyModule)

        # Should point to petsc4py.PETSc, not just petsc4py
        self.assertEqual(petsc._module_name, 'petsc4py.PETSc')

        # Should not be loaded yet
        self.assertIsNone(petsc._module)

    def test_getattr_general_module(self):
        """Test that general modules work through __getattr__."""
        from openmdao.utils import lazy_imports
        from openmdao.utils.lazy_imports import LazyModule

        # Access a general module
        lazy_textwrap = getattr(lazy_imports, 'textwrap')

        # Should be a LazyImport instance
        self.assertIsInstance(lazy_textwrap, LazyModule)

        # Should have correct module name
        self.assertEqual(lazy_textwrap._module_name, 'textwrap')

        # Should not be loaded yet
        self.assertIsNone(lazy_textwrap._module)

        # Should work when used
        result = lazy_textwrap.wrap('hello world', width=5)
        self.assertEqual(result, ['hello', 'world'])

        # Should be loaded now
        self.assertIsNotNone(lazy_textwrap._module)


class TestLazyImportChaining(unittest.TestCase):
    """Test that chained attribute access works correctly."""

    @unittest.skipIf(
        importlib.util.find_spec("mpi4py") is None,
        "mpi4py not installed"
    )
    def test_mpi_import_via_getattr(self):
        """Test that importing MPI via module __getattr__ works correctly."""
        from openmdao.utils.lazy_imports import MPI, LazyModule

        # Should be a LazyImport instance
        self.assertIsInstance(MPI, LazyModule)

        # Should be for mpi4py.MPI
        self.assertEqual(MPI._module_name, 'mpi4py.MPI')

        # Should have COMM_WORLD attribute when accessed
        self.assertTrue(hasattr(MPI, 'COMM_WORLD'))


if __name__ == '__main__':
    unittest.main()
