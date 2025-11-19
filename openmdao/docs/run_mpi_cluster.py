"""
Start or stop an ipyparallel MPI cluster with the specified number of engines.

This script can be used to start a cluster in the background on CI before the
documentation build so that cells which require an ipyparallel cluster can execute.
Later, the same script can be used to stop the cluster.
"""
import argparse
import sys


def start_ipyparallel_cluster(n=4, cluster_id='docs-mpi-cluster'):
    """
    Start an ipyparallel MPI cluster with the given number of engines.

    The cluster will run in the background after this script exits.

    Parameters
    ----------
    n : int
        The number of engines to use.
    cluster_id : str
        The cluster ID to use for this cluster.
    """
    import ipyparallel as ipp
    import os

    # Set MPI environment variables in Python as well
    os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    os.environ["OMPI_MCA_btl"] = "^openib"

    # Create and start MPI cluster using modern API
    print(f"Starting MPI cluster with {n} engines (cluster_id: {cluster_id})...")
    cluster = ipp.Cluster(engines="mpi", n=n, cluster_id=cluster_id)

    try:
        # Start the cluster (synchronous version)
        cluster.start_cluster_sync()
        print("Cluster started successfully!")

        # Connect a client to verify it works
        rc = cluster.connect_client_sync()

        # IMPORTANT: Wait for engines to register before using them
        print("Waiting for engines to register...")
        rc.wait_for_engines(n, timeout=60)  # Wait up to 60 seconds
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
        print(f"Cluster will continue running in the background with ID: {cluster_id}")
        print("To connect from other scripts: rc = ipp.Client(cluster_id='docs-mpi-cluster')")
        print(f"To stop the cluster later: python {sys.argv[0]} stop")

    except Exception as e:
        print(f"Error starting cluster: {e}")
        # Try to clean up on error
        try:
            cluster.stop_cluster_sync()
        except Exception:
            pass
        sys.exit(1)


def stop_ipyparallel_cluster(cluster_id='docs-mpi-cluster'):
    """
    Stop a running ipyparallel MPI cluster.

    Parameters
    ----------
    cluster_id : str
        The cluster ID of the cluster to stop.
    """
    import ipyparallel as ipp

    print(f"Stopping MPI cluster with ID: {cluster_id}...")

    try:
        # Connect to the existing cluster
        cluster = ipp.Cluster(cluster_id=cluster_id)

        # Stop the cluster
        cluster.stop_cluster_sync()
        print("Cluster stopped successfully!")

    except FileNotFoundError:
        print(f"Error: No cluster found with ID '{cluster_id}'")
        print("The cluster may not be running or may have been started with a different ID.")
        sys.exit(1)
    except Exception as e:
        print(f"Error stopping cluster: {e}")
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='run_mpi_cluster',
        description='Start or stop an MPI cluster using ipyparallel.',
        epilog="""
Examples:
  # Start a cluster with 4 engines (default)
  python run_mpi_cluster.py start

  # Start a cluster with 8 engines
  python run_mpi_cluster.py start -n 8

  # Stop the cluster
  python run_mpi_cluster.py stop
        """
    )

    parser.add_argument(
        'action',
        choices=['start', 'stop'],
        help='Action to perform: start or stop the cluster'
    )
    parser.add_argument(
        '-n', '--num_engines',
        type=int,
        default=4,
        help='Number of MPI engines (only used with start action, default: 4)'
    )
    parser.add_argument(
        '--cluster-id',
        default='docs-mpi-cluster',
        help='Cluster ID to use (default: docs-mpi-cluster)'
    )

    args = parser.parse_args()

    if args.action == 'start':
        start_ipyparallel_cluster(n=args.num_engines, cluster_id=args.cluster_id)
    elif args.action == 'stop':
        stop_ipyparallel_cluster(cluster_id=args.cluster_id)