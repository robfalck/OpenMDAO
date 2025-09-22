"""
Start an ipyparlallel MPI cluster with the specified number of engines.

This script should be run in the background on CI before the documentation build
so that cells which require an ipyparallel cluster can execute.
"""
import argparse


def main(n):
    """
    Start an ipyparallel MPI cluster with the given number of engines.

    Parameters
    ----------
    n : int
        The number of engines to use.
    """

    # # Set MPI environment variables
    # export OMPI_MCA_rmaps_base_oversubscribe=1
    # export OMPI_MCA_btl=^openib

    # # Create a Python script to start the modern cluster
    # cat > start_mpi_cluster.py << 'EOF'
    import ipyparallel as ipp
    import os

    # Set MPI environment variables in Python as well
    os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    os.environ["OMPI_MCA_btl"] = "^openib"

    # Create and start MPI cluster using modern API
    print("Starting MPI cluster with 4 engines...")
    cluster = ipp.Cluster(engines="mpi", n=4, cluster_id='docs-mpi-cluster')

    try:
        # Start the cluster (synchronous version)
        cluster.start_cluster_sync()
        print("Cluster started successfully!")

        # Connect a client to verify it works
        rc = cluster.connect_client_sync()

        # IMPORTANT: Wait for engines to register before using them
        print("Waiting for engines to register...")
        rc.wait_for_engines(4, timeout=60)  # Wait up to 60 seconds
        print(f"Connected to cluster with {len(rc)} engines")

        # Test MPI functionality
        def test_mpi():
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                return f"Hello from rank {comm.Get_rank()} of {comm.Get_size()}"
            except ImportError:
                return "MPI not available"

        # Run test on all engines
        results = rc[:].apply_sync(test_mpi)
        print("MPI test results:")
        for i, result in enumerate(results):
            print(f"  Engine {i}: {result}")

        print("\nCluster is ready for use!")
        print("You can now connect to it from Jupyter notebooks or other Python scripts.")
        print("To connect: rc = ipp.Client()")

        # Keep the cluster running
        print("\nPress Ctrl+C to stop the cluster...")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down cluster...")

    except Exception as e:
        print(f"Error starting cluster: {e}")
    finally:
        try:
            cluster.stop_cluster_sync()
            print("Cluster stopped.")
        except Exception:
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='start_mpi_cluster',
                        description='Start an MPI cluster using ipyparallel.')

    parser.add_argument('-n', '--num_engines', default=4, help='Number of MPI engines')
    args = parser.parse_args()
    main(n=args.num_engines)
