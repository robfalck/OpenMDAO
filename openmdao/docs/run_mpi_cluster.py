"""
Start or stop an ipyparallel MPI cluster with the specified number of engines.

This script can be used to start a cluster in the background on CI before the
documentation build so that cells which require an ipyparallel cluster can execute.
Later, the same script can be used to stop the cluster.
"""
import argparse
import sys


def start_ipyparallel_cluster(n=4, cluster_id='docs-mpi-cluster', profile='mpi'):
    """
    Start an ipyparallel MPI cluster with the given number of engines.

    The cluster will run in the background after this script exits.

    Parameters
    ----------
    n : int
        The number of engines to use.
    cluster_id : str
        The cluster ID to use for this cluster (not used with ipcluster command).
    profile : str
        The profile name to use for this cluster (for compatibility with notebooks).
    """
    import subprocess
    import os
    from pathlib import Path
    import time

    # Set MPI environment variables
    os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    os.environ["OMPI_MCA_btl"] = "^openib"

    # Ensure the profile exists - use ipython to create it properly
    print(f"Ensuring profile '{profile}' exists...")
    profile_dir = Path.home() / ".ipython" / f"profile_{profile}"
    cfg_file = profile_dir / "ipcluster_config.py"

    if not cfg_file.exists():
        print(f"Creating ipyparallel profile: {profile}")
        result = subprocess.run(
            ['ipython', 'profile', 'create', '--parallel', f'--profile={profile}'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error creating profile: {result.stderr}")
            sys.exit(1)
        print(f"Profile created at: {profile_dir}")
    else:
        print(f"Using existing profile at: {profile_dir}")

    # Start the cluster using ipcluster command (old API, but it works!)
    print(f"Starting MPI cluster with {n} engines using profile '{profile}'...")
    cmd = [
        'ipcluster', 'start',
        '-n', str(n),
        f'--profile={profile}',
        "--engines=ipyparallel.cluster.launcher.MPIEngineSetLauncher",
        '--daemonize'
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)

    if result.returncode != 0:
        print(f"Error starting cluster: {result.stderr}")
        sys.exit(1)

    print("Cluster start command issued successfully!")
    print("Waiting for cluster to initialize...")

    # Wait a bit for the cluster to start and write connection files
    time.sleep(5)

    # Try to connect to verify it's working
    try:
        from ipyparallel import Client
        print("Attempting to connect to cluster...")
        rc = Client(profile=profile, timeout=30)
        rc.wait_for_engines(n, timeout=30)
        print(f"Successfully connected! {len(rc)} engines available.")

        # Test MPI functionality
        def test_mpi():
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                return f"Hello from rank {comm.Get_rank()} of {comm.Get_size()}"
            except ImportError:
                return "MPI not available"

        results = rc[:].apply_sync(test_mpi)
        print("\nMPI test results:")
        for i, result in enumerate(results):
            print(f"  Engine {i}: {result}")

        print("\nCluster is ready for use!")
        print(f"To connect from notebooks: cluster = Client(profile='{profile}')")
        print(f"To stop the cluster: ipcluster stop --profile={profile}")
        print(f"Or use: python {sys.argv[0]} stop --profile={profile}")

    except Exception as e:
        print(f"Warning: Could not verify cluster connection: {e}")
        print("The cluster may still be starting up. Give it a few more seconds.")
        print(f"To check status: ipcluster engines --profile={profile}")


def stop_ipyparallel_cluster(cluster_id='docs-mpi-cluster', profile='mpi'):
    """
    Stop a running ipyparallel MPI cluster.

    Parameters
    ----------
    cluster_id : str
        The cluster ID of the cluster to stop (not used with ipcluster command).
    profile : str
        The profile name of the cluster to stop.
    """
    import subprocess

    print(f"Stopping MPI cluster with profile: {profile}...")

    result = subprocess.run(
        ['ipcluster', 'stop', f'--profile={profile}'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Cluster stopped successfully!")
    else:
        # ipcluster stop returns non-zero even when successful sometimes
        if "CRITICAL" in result.stderr or "Error" in result.stderr:
            print(f"Error stopping cluster: {result.stderr}")
            sys.exit(1)
        else:
            print("Cluster stop command completed.")
            if result.stdout:
                print(result.stdout)


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
    parser.add_argument(
        '--profile',
        default='mpi',
        help='Profile name to use (default: mpi)'
    )

    args = parser.parse_args()

    if args.action == 'start':
        start_ipyparallel_cluster(n=args.num_engines, cluster_id=args.cluster_id, profile=args.profile)
    elif args.action == 'stop':
        stop_ipyparallel_cluster(cluster_id=args.cluster_id, profile=args.profile)