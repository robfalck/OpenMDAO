"""
Stream-based recorder that writes cases to a file-like stream in real-time.

Supports NDJSON (newline-delimited JSON) or MessagePack binary formats for streaming
case data to files, stdout, or custom streams. Designed for real-time monitoring,
live visualization, and debugging.
"""
import sys
import json
from pathlib import Path
from io import IOBase
import numpy as np

from openmdao.recorders.case_recorder import CaseRecorder


class StreamRecorder(CaseRecorder):
    """
    Recorder that writes cases to a stream in real-time using NDJSON or MessagePack.

    This recorder provides a lightweight alternative to SqliteRecorder for scenarios
    requiring real-time monitoring, live visualization, or simple logging. Each record
    is written immediately to the stream without buffering in memory.

    Supports:
    - Writing to files, stdout, stderr, or any file-like object
    - NDJSON format (human-readable, one JSON object per line)
    - MessagePack format (binary, more compact and faster)
    - MPI parallel execution with two modes:
        - Separate files per rank (default, coordinator_mode=False)
        - Single stream with coordinator pattern (coordinator_mode=True)

    Parameters
    ----------
    stream : str, Path, file-like, or None, optional
        Output destination. Can be:
        - str or Path: filename to create/open for writing
        - file-like object: sys.stdout, sys.stderr, or custom stream
        - None: defaults to sys.stdout
    format : str, optional
        Output format, either 'ndjson' (default) or 'msgpack'.
    buffer_size : int, optional
        Buffer size in bytes for file streams. Default is 8192.
        Set to 1 for line buffering (useful for real-time tailing).
        Set to 0 for unbuffered output (not recommended for performance).
    record_viewer_data : bool, optional
        If True, record data needed for visualization. Default is True.

    Attributes
    ----------
    _stream : file-like object or None
        The stream to which records are written.
    _owns_stream : bool
        True if this recorder opened the stream and is responsible for closing it.
    _encoder : NDJSONEncoder or MessagePackEncoder
        The encoder used to serialize records.
    format : str
        The output format being used.

    Examples
    --------
    Write driver iterations to a file in NDJSON format:

    >>> prob = om.Problem()
    >>> # ... setup problem ...
    >>> prob.driver.add_recorder(StreamRecorder('driver_history.ndjson'))
    >>> prob.run_driver()

    Write to stdout for real-time monitoring:

    >>> prob.driver.add_recorder(StreamRecorder())  # Defaults to stdout

    Use MessagePack for better performance:

    >>> prob.driver.add_recorder(StreamRecorder('data.msgpack', format='msgpack'))

    Real-time visualization by piping to another process:

    >>> import subprocess
    >>> viz = subprocess.Popen(['python', 'viz_tool.py'],
    ...                        stdin=subprocess.PIPE, text=True)
    >>> prob.driver.add_recorder(StreamRecorder(viz.stdin))

    MPI coordinator mode for streaming to a single destination:

    >>> # All MPI ranks will send data to rank 0, which writes to one file
    >>> prob.driver.add_recorder(StreamRecorder('output.ndjson', coordinator_mode=True))
    >>> prob.driver.recording_options['record_outputs'] = True
    >>> prob.driver.recording_options['includes'] = ['*']

    Notes
    -----
    - NDJSON format is human-readable and can be tailed with `tail -f filename.ndjson`
    - MessagePack format is binary and more efficient but requires special tools to read
    - For MPI runs, by default only rank 0 records (matching SqliteRecorder behavior)
    - If multiple ranks record with coordinator_mode=False, each gets a separate file
    - With coordinator_mode=True, all ranks send data to rank 0 for single-stream output
    - Coordinator mode is useful for real-time streaming to web apps or visualization tools
    - In coordinator mode, each record includes a 'rank' field indicating its source
    - Large numpy arrays are converted to lists for JSON serialization
    """

    def __init__(self, stream=None, format='ndjson', buffer_size=8192,
                 record_viewer_data=True, coordinator_mode=False):
        """
        Initialize the StreamRecorder.

        Parameters
        ----------
        stream : str, Path, file-like, or None, optional
            Output destination for the stream.
        format : str, optional
            Output format, either 'ndjson' or 'msgpack'.
        buffer_size : int, optional
            Buffer size in bytes for file streams.
        record_viewer_data : bool, optional
            If True, record data needed for visualization.
        coordinator_mode : bool, optional
            If True, all MPI ranks send records to rank 0 for writing to a single stream.
            This is useful for real-time streaming to destinations like web apps where
            multiple files cannot be merged post-hoc. Default is False, which creates
            separate files per rank.
        """
        super().__init__(record_viewer_data=record_viewer_data)

        self.stream_target = stream
        self.format = format.lower()
        self.buffer_size = buffer_size
        self.coordinator_mode = coordinator_mode
        self._stream = None
        self._owns_stream = False
        self._encoder = None
        self._comm = None

        # Validate format
        if self.format not in ('ndjson', 'msgpack'):
            raise ValueError(f"Format must be 'ndjson' or 'msgpack', got '{format}'")

        # For msgpack, check if it's available
        if self.format == 'msgpack':
            try:
                import msgpack  # noqa: F401
            except ImportError:
                raise ImportError(
                    "MessagePack format requires the 'msgpack' package. "
                    "Install it with: pip install msgpack"
                )

    def _setup_stream(self, stream, buffer_size):
        """
        Set up the output stream with appropriate buffering.

        Parameters
        ----------
        stream : str, Path, file-like, or None
            The stream target to set up.
        buffer_size : int
            Buffer size in bytes.
        """
        if stream is None:
            # Default to stdout
            self._stream = sys.stdout
            self._owns_stream = False
        elif isinstance(stream, (str, Path)):
            # Open file for writing
            mode = 'wb' if self.format == 'msgpack' else 'w'
            self._stream = open(stream, mode, buffering=buffer_size)
            self._owns_stream = True
        elif isinstance(stream, IOBase):
            # Use provided stream object
            self._stream = stream
            self._owns_stream = False
        else:
            raise TypeError(
                f"stream must be str, Path, file-like object, or None, "
                f"got {type(stream)}"
            )

    def _get_encoder(self):
        """
        Get the appropriate encoder for the configured format.

        Returns
        -------
        NDJSONEncoder or MessagePackEncoder
            The encoder instance.
        """
        if self.format == 'ndjson':
            return NDJSONEncoder()
        elif self.format == 'msgpack':
            return MessagePackEncoder()

    @property
    def supports_parallel_recording(self):
        """
        Return True if this recorder can record system/solver iterations in parallel.

        StreamRecorder supports parallel recording when coordinator_mode=True, which
        uses MPI gather to safely collect records from all ranks and write them to
        a single stream without corruption.

        Returns
        -------
        bool
            True if coordinator_mode is enabled.
        """
        return self.coordinator_mode

    def _add_rank_suffix(self, stream, rank):
        """
        Add rank number suffix to stream filename for parallel recording.

        Parameters
        ----------
        stream : str, Path, file-like, or None
            The original stream target.
        rank : int
            The MPI rank number.

        Returns
        -------
        str, Path, or file-like
            Stream target with rank suffix added (if applicable).
        """
        if isinstance(stream, (str, Path)):
            path = Path(stream)
            return f"{path.stem}_rank{rank}{path.suffix}"
        return stream  # Can't add suffix to non-file streams

    def startup(self, recording_requester, comm=None):
        """
        Prepare for a new run and set up MPI configuration.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        comm : MPI.Comm or None
            The MPI communicator for the recorder.
        """
        super().startup(recording_requester, comm)

        # Store comm for use in coordinator mode
        self._comm = comm

        # For serial runs, record_on_process will be None, so default to True
        should_record = self.record_on_process
        if should_record is None:
            should_record = True

        # Set up encoder for all ranks (needed for serialization in coordinator mode)
        self._encoder = self._get_encoder()

        # Open stream based on mode
        if self._parallel and self.coordinator_mode:
            # Coordinator mode: only rank 0 opens the stream
            if comm and comm.rank == 0:
                self._setup_stream(self.stream_target, self.buffer_size)
                if isinstance(self.stream_target, (str, Path)):
                    print(f"Note: StreamRecorder in coordinator mode - rank 0 writing to {self.stream_target}")
        elif should_record:
            # Non-coordinator mode: open streams based on record_on_process
            stream_to_open = self.stream_target

            if self._parallel:
                # Multiple ranks recording - create separate files per rank
                rank = comm.rank if comm else 0
                stream_to_open = self._add_rank_suffix(self.stream_target, rank)
                if isinstance(self.stream_target, (str, Path)):
                    print(f"Note: StreamRecorder rank {rank} writing to {stream_to_open}")

            self._setup_stream(stream_to_open, self.buffer_size)

    def record_viewer_data(self, model_viewer_data):
        """
        Record model viewer data for visualization.

        Parameters
        ----------
        model_viewer_data : dict
            Data required to visualize the model (tree structure, connections, etc.).
        """
        # In serial runs, record_on_process will be None, which we treat as True
        should_record = self.record_on_process if self.record_on_process is not None else True

        if not should_record or not self._record_viewer_data:
            return

        record = {
            'type': 'viewer_data',
            'viewer_data': model_viewer_data
        }
        self._write_record(record)

    def shutdown(self):
        """
        Shut down the recorder and close the stream if owned.
        """
        if self._stream and self._owns_stream:
            self._stream.close()
            self._stream = None

    def record_metadata_system(self, system, run_number=None):
        """
        Record system metadata.

        Parameters
        ----------
        system : System
            The System for which to record metadata.
        run_number : int or None
            Number indicating which run the metadata is associated with.
        """
        # Get basic metadata about the system
        path = system.pathname if system.pathname else 'root'
        record = {
            'type': 'system_metadata',
            'name': path,
            'run_number': run_number,
            'class': type(system).__name__,
        }
        self._write_record(record)

    def record_metadata_solver(self, solver, run_number=None):
        """
        Record solver metadata.

        Parameters
        ----------
        solver : Solver
            The Solver for which to record metadata.
        run_number : int or None
            Number indicating which run the metadata is associated with.
        """
        # Get basic metadata about the solver
        path = solver._system().pathname if solver._system().pathname else 'root'
        solver_class = type(solver).__name__
        record = {
            'type': 'solver_metadata',
            'name': f"{path}.{solver_class}",
            'run_number': run_number,
            'solver_class': solver_class,
            'system_path': path,
        }
        self._write_record(record)

    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record a driver iteration.

        Parameters
        ----------
        recording_requester : Driver
            Driver in need of recording.
        data : dict
            Dictionary containing outputs, inputs, and residuals.
        metadata : dict
            Dictionary containing execution metadata.
        """
        record = {
            'type': 'driver_iteration',
            'counter': self._counter,
            'iteration_coordinate': self._iteration_coordinate,
            'timestamp': metadata.get('timestamp'),
            'success': metadata.get('success'),
            'msg': metadata.get('msg'),
        }

        # Only include non-None/non-empty data
        outputs = self._serialize_vars(data.get('output'))
        if outputs:
            record['outputs'] = outputs

        inputs = self._serialize_vars(data.get('input'))
        if inputs:
            record['inputs'] = inputs

        residuals = self._serialize_vars(data.get('residual'))
        if residuals:
            record['residuals'] = residuals

        self._write_record(record)

    def record_iteration_system(self, recording_requester, data, metadata):
        """
        Record a system iteration.

        Parameters
        ----------
        recording_requester : System
            System in need of recording.
        data : dict
            Dictionary containing inputs, outputs, and residuals.
        metadata : dict
            Dictionary containing execution metadata.
        """
        record = {
            'type': 'system_iteration',
            'counter': self._counter,
            'iteration_coordinate': self._iteration_coordinate,
            'timestamp': metadata.get('timestamp'),
            'success': metadata.get('success'),
            'msg': metadata.get('msg'),
            'inputs': self._serialize_vars(data.get('i')),
            'outputs': self._serialize_vars(data.get('o')),
            'residuals': self._serialize_vars(data.get('r')),
        }
        self._write_record(record)

    def record_iteration_solver(self, recording_requester, data, metadata):
        """
        Record a solver iteration.

        Parameters
        ----------
        recording_requester : Solver
            Solver in need of recording.
        data : dict
            Dictionary containing outputs, residuals, and errors.
        metadata : dict
            Dictionary containing execution metadata.
        """
        record = {
            'type': 'solver_iteration',
            'counter': self._counter,
            'iteration_coordinate': self._iteration_coordinate,
            'timestamp': metadata.get('timestamp'),
            'success': metadata.get('success'),
            'msg': metadata.get('msg'),
            'abs_err': data.get('abs'),
            'rel_err': data.get('rel'),
            'outputs': self._serialize_vars(data.get('o')),
            'residuals': self._serialize_vars(data.get('r')),
        }
        self._write_record(record)

    def record_iteration_problem(self, recording_requester, data, metadata):
        """
        Record a problem iteration.

        Parameters
        ----------
        recording_requester : Problem
            Problem in need of recording.
        data : dict
            Dictionary containing desvars, objectives, and constraints.
        metadata : dict
            Dictionary containing execution metadata.
        """
        record = {
            'type': 'problem_iteration',
            'counter': self._counter,
            'iteration_coordinate': self._iteration_coordinate,
            'timestamp': metadata.get('timestamp'),
            'success': metadata.get('success'),
            'msg': metadata.get('msg'),
            'outputs': self._serialize_vars(data.get('out')),
            'inputs': self._serialize_vars(data.get('in')),
            'residuals': self._serialize_vars(data.get('res')),
        }
        self._write_record(record)

    def record_derivatives_driver(self, recording_requester, data, metadata):
        """
        Record derivatives from a driver.

        Parameters
        ----------
        recording_requester : Driver
            Driver in need of recording.
        data : dict
            Dictionary containing derivatives keyed by 'of!wrt'.
        metadata : dict
            Dictionary containing execution metadata.
        """
        record = {
            'type': 'driver_derivatives',
            'counter': self._counter,
            'iteration_coordinate': self._iteration_coordinate,
            'timestamp': metadata.get('timestamp'),
            'success': metadata.get('success'),
            'msg': metadata.get('msg'),
            'derivatives': self._serialize_vars(data),
        }
        self._write_record(record)

    def _serialize_vars(self, vars_dict):
        """
        Convert variable dictionary to serializable format.

        Converts numpy arrays to lists and handles special types for JSON serialization.

        Parameters
        ----------
        vars_dict : dict or None
            Dictionary of variable names to values.

        Returns
        -------
        dict or None
            Serializable dictionary.
        """
        if vars_dict is None:
            return None

        result = {}
        for name, val in vars_dict.items():
            result[name] = self._serialize_value(val)
        return result

    def _serialize_value(self, val):
        """
        Convert a value to a serializable format.

        Parameters
        ----------
        val : any
            Value to serialize.

        Returns
        -------
        any
            Serializable version of the value.
        """
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif val is None or isinstance(val, (int, float, str, bool, list, dict)):
            return val
        else:
            # Try to convert to string as fallback
            return str(val)

    def _write_record(self, record):
        """
        Write a record to the stream using the configured encoder.

        In coordinator mode with MPI, all ranks serialize their records and send
        them to rank 0, which writes them to a single stream. This ensures that
        multiple ranks can contribute to a single stream destination (like a web app)
        without file corruption.

        Parameters
        ----------
        record : dict
            The record to write.
        """
        if not self._encoder:
            return

        # Check if we're in parallel coordinator mode
        if self._parallel and self.coordinator_mode and self._comm:
            # Coordinator mode: serialize locally, gather to rank 0, and write
            self._write_record_coordinator(record)
        elif self._stream:
            # Normal mode: direct write to stream
            self._encoder.encode(record, self._stream)

    def _write_record_coordinator(self, record):
        """
        Write a record in coordinator mode where rank 0 collects all records.

        Parameters
        ----------
        record : dict
            The record to write.
        """
        # Add rank information to the record
        record['rank'] = self._comm.rank

        # Serialize the record to bytes/string
        if self.format == 'ndjson':
            # Serialize to JSON string
            serialized = json.dumps(record, separators=(',', ':'), cls=NumpyJSONEncoder)
        else:
            # Serialize to MessagePack bytes
            serialized = self._encoder.msgpack.packb(
                record,
                default=self._encoder._default_encoder,
                use_bin_type=True
            )

        # Gather all serialized records to rank 0
        all_serialized = self._comm.gather(serialized, root=0)

        # Rank 0 writes all records to the stream
        if self._comm.rank == 0 and self._stream:
            for data in all_serialized:
                if data is not None:
                    if self.format == 'ndjson':
                        # Write JSON string with newline
                        self._stream.write(data)
                        self._stream.write('\n')
                    else:
                        # Write MessagePack with length prefix
                        length_bytes = len(data).to_bytes(4, byteorder='little')
                        self._stream.write(length_bytes)
                        self._stream.write(data)

            # Flush to ensure immediate availability
            self._stream.flush()


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy types.
    """

    def default(self, obj):
        """
        Convert numpy types and other non-serializable types to Python native types.

        Parameters
        ----------
        obj : any
            Object to convert.

        Returns
        -------
        any
            JSON-serializable version of obj.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, type):
            # Convert class types to their string representation
            return f"{obj.__module__}.{obj.__qualname__}"
        elif hasattr(obj, '__dict__'):
            # Try to serialize objects with __dict__
            return str(obj)
        return super().default(obj)


class NDJSONEncoder:
    """
    Encoder for newline-delimited JSON (NDJSON) format.

    Each record is written as a single line of compact JSON, making the output
    easy to parse line-by-line and suitable for streaming applications.
    """

    def encode(self, record, stream):
        """
        Write record as a single line of JSON.

        Parameters
        ----------
        record : dict
            The record to encode.
        stream : file-like
            The stream to write to.
        """
        # Use compact JSON representation (no whitespace) with custom encoder
        json.dump(record, stream, separators=(',', ':'), cls=NumpyJSONEncoder)
        stream.write('\n')
        # Flush to ensure data is immediately available for real-time monitoring
        stream.flush()


class MessagePackEncoder:
    """
    Encoder for MessagePack binary format.

    Records are written as length-prefixed MessagePack data, providing compact
    binary serialization with better performance than JSON.
    """

    def __init__(self):
        """Initialize the encoder and import msgpack."""
        import msgpack
        self.msgpack = msgpack

    def _default_encoder(self, obj):
        """
        Custom encoder for non-serializable types.

        Parameters
        ----------
        obj : any
            Object to encode.

        Returns
        -------
        any
            Serializable version of obj.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, type):
            return f"{obj.__module__}.{obj.__qualname__}"
        return obj

    def encode(self, record, stream):
        """
        Write record as length-prefixed MessagePack binary data.

        Uses a 4-byte little-endian length prefix to allow reading variable-length
        records from the stream.

        Parameters
        ----------
        record : dict
            The record to encode.
        stream : file-like
            The binary stream to write to.
        """
        # Pack the record with custom default encoder
        data = self.msgpack.packb(record, default=self._default_encoder, use_bin_type=True)

        # Write 4-byte length prefix (little-endian)
        length_bytes = len(data).to_bytes(4, byteorder='little')
        stream.write(length_bytes)

        # Write the packed data
        stream.write(data)

        # Flush to ensure data is written
        stream.flush()


class MessagePackDecoder:
    """
    Decoder for MessagePack binary format.

    Reads length-prefixed MessagePack records from a binary stream.
    """

    def __init__(self):
        """Initialize the decoder and import msgpack."""
        import msgpack
        self.msgpack = msgpack

    def decode(self, stream):
        """
        Read and decode records from a MessagePack stream.

        Yields records one at a time. Use in a loop to read all records.

        Parameters
        ----------
        stream : file-like
            The binary stream to read from.

        Yields
        ------
        dict
            The decoded record.

        Examples
        --------
        Read all records from a MessagePack file:

        >>> decoder = MessagePackDecoder()
        >>> with open('data.msgpack', 'rb') as f:
        ...     for record in decoder.decode(f):
        ...         print(record['type'], record['counter'])
        """
        while True:
            # Read 4-byte length prefix
            length_bytes = stream.read(4)
            if not length_bytes:
                # End of stream
                break

            if len(length_bytes) < 4:
                raise ValueError("Incomplete length prefix at end of stream")

            # Decode length
            length = int.from_bytes(length_bytes, byteorder='little')

            # Read record data
            data = stream.read(length)
            if len(data) < length:
                raise ValueError(f"Incomplete record: expected {length} bytes, got {len(data)}")

            # Unpack and yield the record
            record = self.msgpack.unpackb(data, raw=False)
            yield record


def read_stream_file(filename, format='ndjson'):
    """
    Read records from a StreamRecorder output file.

    This is a convenience function for reading back StreamRecorder data.

    Parameters
    ----------
    filename : str or Path
        Path to the stream recorder file.
    format : str, optional
        Format of the file, either 'ndjson' or 'msgpack'. Default is 'ndjson'.

    Yields
    ------
    dict
        Each record from the file.

    Examples
    --------
    Read an NDJSON file:

    >>> for record in read_stream_file('driver_history.ndjson'):
    ...     if record['type'] == 'driver_iteration':
    ...         print(f"Iteration {record['counter']}")

    Read a MessagePack file:

    >>> for record in read_stream_file('driver_history.msgpack', format='msgpack'):
    ...     print(record['type'])

    Filter for specific record types:

    >>> iterations = [r for r in read_stream_file('output.ndjson')
    ...               if r['type'] == 'driver_iteration']
    """
    if format == 'ndjson':
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield json.loads(line)
    elif format == 'msgpack':
        decoder = MessagePackDecoder()
        with open(filename, 'rb') as f:
            yield from decoder.decode(f)
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'ndjson' or 'msgpack'.")


def convert_stream_to_json(input_file, output_file=None, input_format='msgpack'):
    """
    Convert a StreamRecorder file to NDJSON format.

    Useful for converting MessagePack files to human-readable JSON for debugging
    or analysis.

    Parameters
    ----------
    input_file : str or Path
        Path to the input stream recorder file.
    output_file : str, Path, or None, optional
        Path to the output JSON file. If None, defaults to input_file with
        .ndjson extension.
    input_format : str, optional
        Format of the input file, either 'msgpack' or 'ndjson'. Default is 'msgpack'.

    Examples
    --------
    Convert MessagePack to NDJSON:

    >>> convert_stream_to_json('data.msgpack', 'data.ndjson')

    Auto-generate output filename:

    >>> convert_stream_to_json('data.msgpack')  # Creates data.ndjson
    """
    if output_file is None:
        # Auto-generate output filename
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.ndjson')

    records_written = 0
    with open(output_file, 'w') as outf:
        for record in read_stream_file(input_file, format=input_format):
            json.dump(record, outf, separators=(',', ':'), cls=NumpyJSONEncoder)
            outf.write('\n')
            records_written += 1

    print(f"Converted {records_written} records from {input_file} to {output_file}")
    return records_written