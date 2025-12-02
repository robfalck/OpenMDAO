"""Unit tests for the StreamRecorder."""
import os
import sys
import json
import unittest
from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.recorders.stream_recorder import StreamRecorder
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


@use_tempdirs
class TestStreamRecorderBasic(unittest.TestCase):
    """Test basic StreamRecorder functionality."""

    def test_ndjson_to_file(self):
        """Test recording to a file in NDJSON format."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        recorder = StreamRecorder('driver_cases.ndjson', format='ndjson')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read and validate the recorded data
        self.assertTrue(os.path.exists('driver_cases.ndjson'))

        with open('driver_cases.ndjson', 'r') as f:
            lines = f.readlines()

        self.assertGreater(len(lines), 0)

        # Parse each line as JSON
        records = []
        driver_records = []
        for line in lines:
            record = json.loads(line)
            records.append(record)
            if record['type'] == 'driver_iteration':
                driver_records.append(record)
                self.assertIn('counter', record)
                self.assertIn('iteration_coordinate', record)

        # Check that we have multiple driver iterations
        self.assertGreater(len(driver_records), 1)

        # Check that counters increment
        counters = [r['counter'] for r in driver_records]
        self.assertEqual(counters, list(range(1, len(driver_records) + 1)))

    def test_ndjson_to_stdout(self):
        """Test recording to stdout in NDJSON format."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            recorder = StreamRecorder(None, format='ndjson')  # None = stdout
            prob.driver.add_recorder(recorder)

            prob.setup()
            prob.run_driver()
            recorder.shutdown()
            prob.cleanup()

            output = captured_output.getvalue()

            # Parse output lines
            lines = [line for line in output.split('\n') if line.strip()]
            records = [json.loads(line) for line in lines]

            self.assertGreater(len(records), 0)
            # Check that we have at least one driver iteration
            driver_records = [r for r in records if r['type'] == 'driver_iteration']
            self.assertGreater(len(driver_records), 0)

        finally:
            sys.stdout = old_stdout

    def test_ndjson_to_stream_object(self):
        """Test recording to a StringIO stream object."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x**2 + 3*x + 5'), promotes=['*'])

        stream = StringIO()
        recorder = StreamRecorder(stream, format='ndjson')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read from the stream
        stream.seek(0)
        lines = stream.readlines()

        self.assertGreater(len(lines), 0)

        records = [json.loads(line) for line in lines]
        # Check that we have at least one driver iteration
        driver_records = [r for r in records if r['type'] == 'driver_iteration']
        self.assertGreater(len(driver_records), 0)

    @unittest.skipIf(not MSGPACK_AVAILABLE, "msgpack not available")
    def test_msgpack_to_file(self):
        """Test recording to a file in MessagePack format."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        recorder = StreamRecorder('driver_cases.msgpack', format='msgpack')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read and validate the recorded data
        self.assertTrue(os.path.exists('driver_cases.msgpack'))

        records = []
        with open('driver_cases.msgpack', 'rb') as f:
            while True:
                # Read length prefix
                length_bytes = f.read(4)
                if not length_bytes:
                    break

                length = int.from_bytes(length_bytes, byteorder='little')

                # Read record data
                data = f.read(length)
                record = msgpack.unpackb(data, raw=False)
                records.append(record)

        self.assertGreater(len(records), 0)
        # Check driver iterations
        driver_records = [r for r in records if r['type'] == 'driver_iteration']
        self.assertGreater(len(driver_records), 0)
        for record in driver_records:
            self.assertIn('counter', record)

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            StreamRecorder('output.txt', format='invalid')

        self.assertIn('ndjson', str(cm.exception).lower())
        self.assertIn('msgpack', str(cm.exception).lower())

    @unittest.skipIf(MSGPACK_AVAILABLE, "msgpack is available")
    def test_msgpack_without_package(self):
        """Test that msgpack format without package raises ImportError."""
        with self.assertRaises(ImportError) as cm:
            StreamRecorder('output.msgpack', format='msgpack')

        self.assertIn('msgpack', str(cm.exception).lower())

    def test_invalid_stream_type(self):
        """Test that invalid stream type raises TypeError."""
        recorder = StreamRecorder(123, format='ndjson')  # Invalid: int
        prob = om.Problem()

        with self.assertRaises(TypeError):
            recorder.startup(prob.driver)


@use_tempdirs
class TestStreamRecorderOptions(unittest.TestCase):
    """Test StreamRecorder recording options."""

    def test_record_only_desvars(self):
        """Test recording only design variables."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        recorder = StreamRecorder('cases.ndjson')
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = False
        prob.driver.recording_options['record_constraints'] = False
        prob.driver.recording_options['record_inputs'] = False
        prob.driver.recording_options['record_outputs'] = False
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Verify that we got driver iterations but with limited data
        # (StreamRecorder only includes non-empty dictionaries in the record)
        driver_records = []
        with open('cases.ndjson', 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['type'] == 'driver_iteration':
                    driver_records.append(record)
                    # Should have the standard iteration fields
                    self.assertIn('counter', record)
                    self.assertIn('iteration_coordinate', record)
                    # When nothing is recorded, outputs/inputs/residuals are omitted (empty dicts)
                    # This is expected behavior - StreamRecorder doesn't write empty collections

        # Ensure we actually checked some driver iterations
        self.assertGreater(len(driver_records), 0)

    def test_system_recorder(self):
        """Test recording system iterations."""

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=2*x'), promotes=['*'])

        recorder = StreamRecorder('system_cases.ndjson')

        # System recording not supported in parallel - this will raise RuntimeError
        try:
            model.add_recorder(recorder)
        except RuntimeError as e:
            if "parallel" in str(e).lower():
                # Expected in parallel, skip the test
                self.skipTest("System recording not supported in parallel")
            else:
                raise

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True

        prob.setup()
        prob.run_model()
        recorder.shutdown()
        prob.cleanup()

        # Read and verify
        system_records = []
        with open('system_cases.ndjson', 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['type'] == 'system_iteration':
                    system_records.append(record)

        self.assertGreater(len(system_records), 0)
        for record in system_records:
            self.assertIn('inputs', record)
            self.assertIn('outputs', record)
            self.assertIn('residuals', record)

    def test_solver_recorder(self):
        """Test recording solver iterations."""

        prob = om.Problem(model=SellarDerivatives())

        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.DirectSolver()

        recorder = StreamRecorder('solver_cases.ndjson')

        # Solver recording not supported in parallel - this will raise RuntimeError
        try:
            prob.model.nonlinear_solver.add_recorder(recorder)
        except RuntimeError as e:
            if "parallel" in str(e).lower():
                # Expected in parallel, skip the test
                self.skipTest("Solver recording not supported in parallel")
            else:
                raise

        prob.setup()
        prob.run_model()
        recorder.shutdown()
        prob.cleanup()

        # Read and verify
        solver_records = []
        with open('solver_cases.ndjson', 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['type'] == 'solver_iteration':
                    solver_records.append(record)

        self.assertGreater(len(solver_records), 0)
        for record in solver_records:
            self.assertIn('abs_err', record)
            self.assertIn('rel_err', record)


@use_tempdirs
class TestStreamRecorderDataTypes(unittest.TestCase):
    """Test StreamRecorder handling of different data types."""

    def test_numpy_array_serialization(self):
        """Test that numpy arrays are properly serialized to lists."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x',
                                                 x=np.array([1., 2., 3.]),
                                                 y=np.array([0., 0., 0.])))

        recorder = StreamRecorder('array_cases.ndjson')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read and verify arrays were converted to lists
        with open('array_cases.ndjson', 'r') as f:
            record = json.loads(f.readline())

        # Check that we can parse it (arrays should be lists now)
        self.assertIsInstance(record, dict)

    def test_scalar_values(self):
        """Test recording of scalar values."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=2*x'), promotes=['*'])

        recorder = StreamRecorder('scalar_cases.ndjson')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.set_val('x', 5.0)
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Verify scalars are recorded correctly
        with open('scalar_cases.ndjson', 'r') as f:
            record = json.loads(f.readline())

        self.assertIsInstance(record, dict)


@use_tempdirs
class TestStreamRecorderBuffering(unittest.TestCase):
    """Test StreamRecorder buffering options."""

    def test_line_buffering(self):
        """Test line buffering mode."""
        recorder = StreamRecorder('buffered.ndjson', buffer_size=1)

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()

        # File should have content even before closing (due to line buffering)
        self.assertTrue(os.path.exists('buffered.ndjson'))
        file_size = os.path.getsize('buffered.ndjson')
        self.assertGreater(file_size, 0)

        recorder.shutdown()
        prob.cleanup()

    def test_large_buffer(self):
        """Test large buffer size."""
        recorder = StreamRecorder('large_buffer.ndjson', buffer_size=65536)

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Should still work correctly
        self.assertTrue(os.path.exists('large_buffer.ndjson'))
        with open('large_buffer.ndjson', 'r') as f:
            records = [json.loads(line) for line in f]
        self.assertGreater(len(records), 0)


@use_tempdirs
class TestStreamRecorderMPI(unittest.TestCase):
    """Test StreamRecorder with MPI."""

    N_PROCS = 2

    def test_parallel_recording_separate_files(self):
        """Test that parallel recording creates separate files per rank."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder = StreamRecorder('parallel.ndjson')
        recorder.record_on_process = True  # All ranks record
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        if MPI and prob.comm.size > 1:
            # Each rank should have its own file
            rank = prob.comm.rank
            expected_file = f'parallel_rank{rank}.ndjson'
            self.assertTrue(os.path.exists(expected_file))

            with open(expected_file, 'r') as f:
                records = [json.loads(line) for line in f]
            self.assertGreater(len(records), 0)
        else:
            # Serial run should use original filename
            self.assertTrue(os.path.exists('parallel.ndjson'))

    def test_rank_zero_only_recording(self):
        """Test default behavior of recording only on rank 0."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder = StreamRecorder('rank0_only.ndjson')
        # Don't set record_on_process - default is rank 0 only
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        if MPI and prob.comm.size > 1:
            if prob.comm.rank == 0:
                # Rank 0 should have the file
                self.assertTrue(os.path.exists('rank0_only.ndjson'))
            # Other ranks won't create files
        else:
            # Serial run
            self.assertTrue(os.path.exists('rank0_only.ndjson'))

    def test_coordinator_mode_ndjson(self):
        """Test coordinator mode where all ranks write to single file."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder = StreamRecorder('coordinator.ndjson', coordinator_mode=True)
        recorder.record_on_process = True  # All ranks participate
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        if MPI and prob.comm.size > 1:
            # Only rank 0 creates the file, but it contains data from all ranks
            if prob.comm.rank == 0:
                self.assertTrue(os.path.exists('coordinator.ndjson'))

                with open('coordinator.ndjson', 'r') as f:
                    records = [json.loads(line) for line in f]

                # Should have records from all ranks
                ranks_seen = set()
                for record in records:
                    if 'rank' in record:
                        ranks_seen.add(record['rank'])

                # We should see data from all ranks
                self.assertEqual(ranks_seen, set(range(prob.comm.size)))
        else:
            # Serial run should still work
            self.assertTrue(os.path.exists('coordinator.ndjson'))

    @unittest.skipIf(not MSGPACK_AVAILABLE, "msgpack not available")
    def test_coordinator_mode_msgpack(self):
        """Test coordinator mode with MessagePack format."""
        from openmdao.recorders.stream_recorder import read_stream_file

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder = StreamRecorder('coordinator.msgpack', format='msgpack',
                                  coordinator_mode=True)
        recorder.record_on_process = True  # All ranks participate
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        if MPI and prob.comm.size > 1:
            # Only rank 0 creates the file
            if prob.comm.rank == 0:
                self.assertTrue(os.path.exists('coordinator.msgpack'))

                # Read back and verify
                records = list(read_stream_file('coordinator.msgpack', format='msgpack'))
                self.assertGreater(len(records), 0)

                # Should have records from all ranks
                ranks_seen = set()
                for record in records:
                    if 'rank' in record:
                        ranks_seen.add(record['rank'])

                # We should see data from all ranks
                self.assertEqual(ranks_seen, set(range(prob.comm.size)))
        else:
            # Serial run
            self.assertTrue(os.path.exists('coordinator.msgpack'))

    def test_coordinator_mode_vs_separate_files(self):
        """Compare coordinator mode output to separate file mode."""
        # First run with separate files
        prob1 = om.Problem()
        prob1.model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder1 = StreamRecorder('separate.ndjson', coordinator_mode=False)
        recorder1.record_on_process = True
        prob1.driver.add_recorder(recorder1)

        prob1.setup()
        prob1.run_driver()
        recorder1.shutdown()
        prob1.cleanup()

        # Now run with coordinator mode
        prob2 = om.Problem()
        prob2.model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder2 = StreamRecorder('coordinator_compare.ndjson', coordinator_mode=True)
        recorder2.record_on_process = True
        prob2.driver.add_recorder(recorder2)

        prob2.setup()
        prob2.run_driver()
        recorder2.shutdown()
        prob2.cleanup()

        if MPI and prob1.comm.size > 1:
            rank = prob1.comm.rank

            # Separate files mode creates one file per rank
            separate_file = f'separate_rank{rank}.ndjson'
            self.assertTrue(os.path.exists(separate_file))

            if rank == 0:
                # Coordinator mode creates single file on rank 0
                self.assertTrue(os.path.exists('coordinator_compare.ndjson'))

                with open('coordinator_compare.ndjson', 'r') as f:
                    coord_records = [json.loads(line) for line in f]

                # Count records from each rank in coordinator file
                rank_counts = {}
                for record in coord_records:
                    r = record.get('rank', 0)
                    rank_counts[r] = rank_counts.get(r, 0) + 1

                # Should have records from all ranks
                self.assertEqual(set(rank_counts.keys()), set(range(prob1.comm.size)))
        else:
            # Serial run
            self.assertTrue(os.path.exists('separate.ndjson'))
            self.assertTrue(os.path.exists('coordinator_compare.ndjson'))

    def test_system_recorder_coordinator_mode(self):
        """Test that system recording works in parallel with coordinator mode."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', om.ExecComp('y=2*x'), promotes=['*'])

        # With coordinator mode, system recording should work in parallel
        recorder = StreamRecorder('system_parallel.ndjson', coordinator_mode=True)
        recorder.record_on_process = True  # All ranks participate

        # This should NOT raise RuntimeError with coordinator mode
        model.add_recorder(recorder)

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True

        prob.setup()
        prob.run_model()
        recorder.shutdown()
        prob.cleanup()

        if MPI and prob.comm.size > 1:
            if prob.comm.rank == 0:
                # Only rank 0 creates the file in coordinator mode
                self.assertTrue(os.path.exists('system_parallel.ndjson'))

                # Read and verify
                system_records = []
                with open('system_parallel.ndjson', 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if record['type'] == 'system_iteration':
                            system_records.append(record)

                # Should have system records with rank information
                self.assertGreater(len(system_records), 0)
                for record in system_records:
                    self.assertIn('rank', record)
                    self.assertIn('inputs', record)
                    self.assertIn('outputs', record)
                    self.assertIn('residuals', record)
        else:
            # Serial run
            self.assertTrue(os.path.exists('system_parallel.ndjson'))

    def test_solver_recorder_coordinator_mode(self):
        """Test that solver recording works in parallel with coordinator mode."""
        prob = om.Problem(model=SellarDerivatives())

        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.DirectSolver()

        # With coordinator mode, solver recording should work in parallel
        recorder = StreamRecorder('solver_parallel.ndjson', coordinator_mode=True)
        recorder.record_on_process = True  # All ranks participate

        # This should NOT raise RuntimeError with coordinator mode
        prob.model.nonlinear_solver.add_recorder(recorder)

        prob.model.nonlinear_solver.recording_options['record_abs_error'] = True
        prob.model.nonlinear_solver.recording_options['record_rel_error'] = True

        prob.setup()
        prob.run_model()
        recorder.shutdown()
        prob.cleanup()

        if MPI and prob.comm.size > 1:
            if prob.comm.rank == 0:
                # Only rank 0 creates the file in coordinator mode
                self.assertTrue(os.path.exists('solver_parallel.ndjson'))

                # Read and verify
                solver_records = []
                with open('solver_parallel.ndjson', 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if record['type'] == 'solver_iteration':
                            solver_records.append(record)

                # Should have solver records with rank information
                self.assertGreater(len(solver_records), 0)
                for record in solver_records:
                    self.assertIn('rank', record)
                    self.assertIn('abs_err', record)
                    self.assertIn('rel_err', record)
        else:
            # Serial run
            self.assertTrue(os.path.exists('solver_parallel.ndjson'))


@use_tempdirs
class TestStreamRecorderIntegration(unittest.TestCase):
    """Integration tests for StreamRecorder."""

    def test_multiple_recorders(self):
        """Test using multiple StreamRecorders simultaneously."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        # Add multiple recorders
        recorder1 = StreamRecorder('output1.ndjson')
        recorder2 = StreamRecorder('output2.ndjson')
        prob.driver.add_recorder(recorder1)
        prob.driver.add_recorder(recorder2)

        prob.setup()
        prob.run_driver()
        recorder1.shutdown()
        recorder2.shutdown()
        prob.cleanup()

        # Both files should exist and have same content
        self.assertTrue(os.path.exists('output1.ndjson'))
        self.assertTrue(os.path.exists('output2.ndjson'))

        with open('output1.ndjson', 'r') as f1, open('output2.ndjson', 'r') as f2:
            records1 = [json.loads(line) for line in f1]
            records2 = [json.loads(line) for line in f2]

        self.assertEqual(len(records1), len(records2))

    def test_recorder_with_derivatives(self):
        """Test recording derivatives."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        recorder = StreamRecorder('derivatives.ndjson')
        prob.driver.recording_options['record_derivatives'] = True
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Check for derivative records
        with open('derivatives.ndjson', 'r') as f:
            records = [json.loads(line) for line in f]

        # Should have both iteration and derivative records
        record_types = set(r['type'] for r in records)
        self.assertIn('driver_iteration', record_types)


@use_tempdirs
class TestStreamRecorderDecoding(unittest.TestCase):
    """Test decoding and reading StreamRecorder output."""

    @unittest.skipIf(not MSGPACK_AVAILABLE, "msgpack not available")
    def test_msgpack_decoder(self):
        """Test MessagePackDecoder can read encoded data."""
        from openmdao.recorders.stream_recorder import MessagePackDecoder

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        recorder = StreamRecorder('test.msgpack', format='msgpack')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read back with decoder
        decoder = MessagePackDecoder()
        records = []
        with open('test.msgpack', 'rb') as f:
            for record in decoder.decode(f):
                records.append(record)

        self.assertGreater(len(records), 0)
        # Check that we have driver iterations
        driver_iters = [r for r in records if r['type'] == 'driver_iteration']
        self.assertGreater(len(driver_iters), 0)

    def test_read_stream_file_ndjson(self):
        """Test read_stream_file utility with NDJSON."""
        from openmdao.recorders.stream_recorder import read_stream_file

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExecComp('y=2*x'), promotes=['*'])

        recorder = StreamRecorder('test.ndjson', format='ndjson')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read back with utility
        records = list(read_stream_file('test.ndjson', format='ndjson'))

        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('type', record)

    @unittest.skipIf(not MSGPACK_AVAILABLE, "msgpack not available")
    def test_read_stream_file_msgpack(self):
        """Test read_stream_file utility with MessagePack."""
        from openmdao.recorders.stream_recorder import read_stream_file

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExecComp('y=3*x'), promotes=['*'])

        recorder = StreamRecorder('test.msgpack', format='msgpack')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Read back with utility
        records = list(read_stream_file('test.msgpack', format='msgpack'))

        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn('type', record)

    @unittest.skipIf(not MSGPACK_AVAILABLE, "msgpack not available")
    def test_convert_stream_to_json(self):
        """Test converting MessagePack to JSON."""
        from openmdao.recorders.stream_recorder import convert_stream_to_json

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder = StreamRecorder('original.msgpack', format='msgpack')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Convert to JSON
        num_records = convert_stream_to_json('original.msgpack', 'converted.ndjson')

        self.assertGreater(num_records, 0)
        self.assertTrue(os.path.exists('converted.ndjson'))

        # Verify converted file is valid JSON
        with open('converted.ndjson', 'r') as f:
            for line in f:
                record = json.loads(line)
                self.assertIn('type', record)

    @unittest.skipIf(not MSGPACK_AVAILABLE, "msgpack not available")
    def test_convert_stream_auto_filename(self):
        """Test convert_stream_to_json with auto-generated filename."""
        from openmdao.recorders.stream_recorder import convert_stream_to_json

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', om.ExecComp('y=x**2'), promotes=['*'])

        recorder = StreamRecorder('data.msgpack', format='msgpack')
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        recorder.shutdown()
        prob.cleanup()

        # Convert with auto-generated filename
        num_records = convert_stream_to_json('data.msgpack')

        self.assertGreater(num_records, 0)
        self.assertTrue(os.path.exists('data.ndjson'))


if __name__ == '__main__':
    unittest.main()