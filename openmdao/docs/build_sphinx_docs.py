#!/usr/bin/env python
"""
Build the Sphinx documentation for OpenMDAO.

This script provides a convenient interface for building the Sphinx documentation
with configurable parallelization and other options. It automatically manages
the ipyparallel cluster needed for parallel notebook execution.
"""
import argparse
import os
import subprocess
import sys
import pathlib
import signal
import atexit
import shutil


def start_ipyparallel_cluster(n_engines=4, verbose=False):
    """
    Start an ipyparallel cluster for notebook execution using run_mpi_cluster.py.

    Parameters
    ----------
    n_engines : int
        Number of engines to start.
    verbose : bool
        Whether to show verbose output.

    Returns
    -------
    bool
        True if cluster started successfully, False otherwise.
    """
    print(f"Starting ipyparallel cluster with {n_engines} engines...")

    # Get the path to run_mpi_cluster.py
    _this_file = pathlib.Path(__file__).resolve()
    docs_root = _this_file.parent
    run_mpi_cluster_path = docs_root / 'run_mpi_cluster.py'

    if not run_mpi_cluster_path.exists():
        print(f"Error: run_mpi_cluster.py not found at {run_mpi_cluster_path}")
        return False

    # Build the command
    cmd = [sys.executable, str(run_mpi_cluster_path), 'start', '-n', str(n_engines)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("ipyparallel cluster started successfully")
            return True
        else:
            print(f"Failed to start cluster (return code: {result.returncode})")
            if not verbose and result.stdout:
                print(result.stdout)
            if not verbose and result.stderr:
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("Error: Cluster startup timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"Failed to start ipyparallel cluster: {e}")
        return False


def stop_ipyparallel_cluster(verbose=False):
    """
    Stop the ipyparallel cluster using run_mpi_cluster.py.

    Parameters
    ----------
    verbose : bool
        Whether to show verbose output.
    """
    print("Stopping ipyparallel cluster...")

    # Get the path to run_mpi_cluster.py
    _this_file = pathlib.Path(__file__).resolve()
    docs_root = _this_file.parent
    run_mpi_cluster_path = docs_root / 'run_mpi_cluster.py'

    if not run_mpi_cluster_path.exists():
        print(f"Error: run_mpi_cluster.py not found at {run_mpi_cluster_path}")
        return

    # Build the command
    cmd = [sys.executable, str(run_mpi_cluster_path), 'stop']

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("ipyparallel cluster stopped successfully")
        else:
            print(f"Warning: Failed to stop cluster (return code: {result.returncode})")
            if not verbose and result.stdout:
                print(result.stdout)
            if not verbose and result.stderr:
                print(result.stderr)

    except subprocess.TimeoutExpired:
        print("Warning: Cluster stop timed out after 30 seconds")
    except Exception as e:
        if verbose:
            print(f"Error stopping cluster: {e}")


def build_sphinx_docs(sphinx_dir='sphinx_docs', jobs=4, clean=False, builder='html',
                      verbose=False, warnings_as_errors=False, keep_going=False,
                      manage_cluster=True, cluster_engines=4, nb_execution_mode='auto'):
    """
    Build the Sphinx documentation.

    Parameters
    ----------
    sphinx_dir : str
        Directory containing the Sphinx documentation source files.
    jobs : int
        Number of parallel jobs to use for the build.
    clean : bool
        Whether to clean the build directory before building.
    builder : str
        The Sphinx builder to use (e.g., 'html', 'latex', 'linkcheck').
    verbose : bool
        Whether to show verbose output.
    warnings_as_errors : bool
        Whether to treat warnings as errors.
    keep_going : bool
        Whether to continue building after errors.
    manage_cluster : bool
        Whether to automatically start/stop the ipyparallel cluster.
    cluster_engines : int
        Number of engines for the ipyparallel cluster.
    nb_execution_mode : str
        Behavior of nb_execution_mode for myst-nb. One of "clean", "force", "auto", or "off"

    Returns
    -------
    int
        Return code from the build process (0 for success).
    """
    # Get absolute path to sphinx_docs directory
    _this_file = pathlib.Path(__file__).resolve()
    docs_root = _this_file.parent
    sphinx_path = pathlib.Path(docs_root, sphinx_dir)

    if not sphinx_path.exists():
        print(f"Error: Sphinx directory not found: {sphinx_path}")
        return 1

    # Check if _srcdocs directory is populated
    srcdocs_path = sphinx_path / "_srcdocs"
    srcdocs_packages_path = srcdocs_path / "packages"

    needs_source_docs = False
    if not srcdocs_path.exists():
        needs_source_docs = True
        print("Source documentation directory does not exist.")
    elif not srcdocs_packages_path.exists():
        needs_source_docs = True
        print("Source documentation packages directory does not exist.")
    elif not any(srcdocs_packages_path.iterdir()):
        needs_source_docs = True
        print("Source documentation packages directory is empty.")

    if needs_source_docs:
        print("Building source documentation...")
        # Import and run build_source_docs
        build_source_docs_path = docs_root / 'build_source_docs.py'
        if not build_source_docs_path.exists():
            print(f"Error: build_source_docs.py not found at {build_source_docs_path}")
            return 1

        # Import the module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_source_docs", build_source_docs_path)
        build_source_docs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_source_docs_module)

        # Call build_src_docs with the correct paths
        # sphinx_dir is relative to docs_root (e.g., 'sphinx_docs')
        # src_dir should be the openmdao package directory
        src_dir = docs_root.parent  # Go up from docs to openmdao package
        try:
            build_source_docs_module.build_src_docs(
                top=str(sphinx_path),
                src_dir=str(src_dir),
                project_name='openmdao'
            )
            print("Source documentation built successfully.")
        except Exception as e:
            print(f"Error building source documentation: {e}")
            return 1

    # Start ipyparallel cluster if requested
    cluster_started = False
    cluster_stopped = False  # Track if we've already stopped the cluster

    if manage_cluster:
        cluster_started = start_ipyparallel_cluster(
            n_engines=cluster_engines,
            verbose=verbose
        )

        # Register cleanup function to stop cluster on exit
        if cluster_started:
            def cleanup():
                nonlocal cluster_stopped
                if not cluster_stopped:
                    stop_ipyparallel_cluster(verbose=verbose)
                    cluster_stopped = True

            atexit.register(cleanup)

            # Handle Ctrl+C gracefully
            def signal_handler(sig, frame):
                print("\nInterrupted! Cleaning up...")
                cleanup()
                sys.exit(1)

            signal.signal(signal.SIGINT, signal_handler)

    # Save current directory and change to sphinx_docs
    save_cwd = os.getcwd()
    os.chdir(sphinx_path)

    try:
        # Clean if requested
        if clean:
            print("Cleaning build directory...")
            result = subprocess.run(['make', 'clean'], capture_output=not verbose)
            if result.returncode != 0:
                print("Warning: Clean command failed")

            # Also clean the jupyter cache
            jupyter_cache_path = sphinx_path / '_build' / '.jupyter_cache'
            if jupyter_cache_path.exists():
                print("Cleaning jupyter cache...")
                try:
                    shutil.rmtree(jupyter_cache_path)
                    print("Jupyter cache cleared successfully")
                except Exception as e:
                    print(f"Warning: Failed to clear jupyter cache: {e}")

        # Build the make command
        make_cmd = ['make', builder]

        # Add parallel jobs flag
        if jobs > 1:
            make_cmd.append(f'-j{jobs}')

        # Set SPHINXOPTS environment variable for additional options
        env = os.environ.copy()
        sphinx_opts = []

        if warnings_as_errors:
            sphinx_opts.append('-W')

        if keep_going:
            sphinx_opts.append('--keep-going')

        if verbose:
            sphinx_opts.append('-v')

        if sphinx_opts:
            env['SPHINXOPTS'] = ' '.join(sphinx_opts)

        env['SPHINX_NB_EXECUTION_MODE'] = nb_execution_mode
        # env['OPENMDAO_REPORTS'] = '0'

        # Run the build
        print(f"Building Sphinx documentation with {jobs} parallel job(s)...")
        print(f"Command: {' '.join(make_cmd)}")
        if sphinx_opts:
            print(f"SPHINXOPTS: {' '.join(sphinx_opts)}")

        result = subprocess.run(make_cmd, env=env)

        if result.returncode == 0:
            build_dir = sphinx_path / '_build' / builder
            print("\nBuild completed successfully!")
            print(f"Documentation available at: {build_dir}")
            if builder == 'html':
                print(f"Open in browser: file://{build_dir}/index.html")
        else:
            print(f"\nBuild failed with return code {result.returncode}")

        return result.returncode

    finally:
        # Restore original directory
        os.chdir(save_cwd)

        # Stop the cluster if we started it and haven't stopped it yet
        if cluster_started and manage_cluster and not cluster_stopped:
            stop_ipyparallel_cluster(verbose=verbose)
            cluster_stopped = True


def main():
    """Parse command line arguments and build the documentation."""
    parser = argparse.ArgumentParser(
        description='Build OpenMDAO Sphinx documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default settings (4 parallel jobs, auto-start cluster)
  %(prog)s

  # Build with 8 parallel jobs and 8 cluster engines
  %(prog)s -j 8 -n 8

  # Clean and rebuild with verbose output
  %(prog)s --clean --verbose

  # Build without starting the ipyparallel cluster (already running)
  %(prog)s --no-cluster

  # Build and treat warnings as errors (useful for CI)
  %(prog)s -W

  # Check all external links
  %(prog)s --builder linkcheck

  # Build PDF documentation
  %(prog)s --builder latexpdf
        """
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=4,
        metavar='N',
        help='Number of parallel jobs to use for building (default: 4)'
    )

    parser.add_argument(
        '-c', '--clean',
        action='store_true',
        help='Clean the build directory and jupyter cache before building'
    )

    parser.add_argument(
        '-b', '--builder',
        default='html',
        choices=['html', 'latex', 'latexpdf', 'linkcheck', 'doctest', 'coverage'],
        help='Sphinx builder to use (default: html)'
    )

    parser.add_argument(
        '-d', '--dir',
        default='sphinx_docs',
        dest='sphinx_dir',
        metavar='DIR',
        help='Sphinx documentation directory (default: sphinx_docs)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output during build'
    )

    parser.add_argument(
        '-W', '--warnings-as-errors',
        action='store_true',
        help='Treat warnings as errors (build will fail on warnings)'
    )

    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='Continue building after errors (useful with -W)'
    )

    parser.add_argument(
        '--no-cluster',
        action='store_true',
        help='Do not automatically start/stop the ipyparallel cluster'
    )

    parser.add_argument(
        '-n', '--cluster-engines',
        type=int,
        default=4,
        metavar='N',
        help='Number of engines for the ipyparallel cluster (default: 4)'
    )

    parser.add_argument(
        '-m', '--nb-mode',
        type=str,
        default='auto',
        choices=['auto', 'cache', 'force', 'off'],
        metavar='MODE',
        help='Notebook execution mode. One of "auto", "cache", "force", or "off".'
    )

    args = parser.parse_args()

    # Build the documentation
    return_code = build_sphinx_docs(
        sphinx_dir=args.sphinx_dir,
        jobs=args.jobs,
        clean=args.clean,
        builder=args.builder,
        verbose=args.verbose,
        warnings_as_errors=args.warnings_as_errors,
        keep_going=args.keep_going,
        manage_cluster=not args.no_cluster,
        cluster_engines=args.cluster_engines,
        nb_execution_mode=args.nb_mode
    )

    sys.exit(return_code)


if __name__ == '__main__':
    main()