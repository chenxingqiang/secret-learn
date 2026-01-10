#!/usr/bin/env python3
"""
Test SS mode with local SPU simulation

Uses SecretFlow's built-in simulation mode for single-machine testing.
"""

import numpy as np
import sys

try:
    import secretflow as sf
    from secretflow.utils.simulation.datasets import load_iris
except ImportError:
    print("❌ SecretFlow not installed")
    sys.exit(1)


def main():
    print("="*70)
    print(" SS Local SPU Test")
    print("="*70)
    
    # Use simulation mode for local testing
    print("\n[1/4] Initializing SecretFlow (simulation)...")
    sf.shutdown()  # Clean up any previous session
    
    # Initialize with simulation cluster
    sf.init(['alice', 'bob', 'carol'], address='local')
    
    # Create devices
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')
    
    # Create SPU with simulation
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))
    print("  ✓ SPU created")
    
    # Step 2: Create sample data
    print("\n[2/4] Creating sample data...")
    from secretflow.data import FedNdarray, PartitionWay
    
    np.random.seed(42)
    n_samples = 100
    n_features = 6
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    
    # Split data
    X_alice = X[:, :2]
    X_bob = X[:, 2:4]  
    X_carol = X[:, 4:6]
    
    fed_X = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(X_alice),
            bob: bob(lambda x: x)(X_bob),
            carol: carol(lambda x: x)(X_carol),
        },
        partition_way=PartitionWay.VERTICAL
    )
    fed_y = FedNdarray(
        partitions={alice: alice(lambda x: x)(y)},
        partition_way=PartitionWay.HORIZONTAL
    )
    print(f"  ✓ Data: {n_samples} samples × {n_features} features")
    
    # Step 3: Test SS module
    print("\n[3/4] Testing SSLinearRegression...")
    sys.path.insert(0, '/Users/xingqiangchen/secret-learn')
    from secretlearn.SS.linear_models.linear_regression import SSLinearRegression
    
    model = SSLinearRegression(spu)
    
    try:
        model.fit(fed_X, fed_y)
        print("  ✓ SSLinearRegression fit completed")
    except Exception as e:
        print(f"  ✗ SSLinearRegression fit failed: {e}")
    
    # Step 4: Cleanup
    print("\n[4/4] Cleanup...")
    sf.shutdown()
    print("  ✓ Done")
    
    print("\n" + "="*70)
    print(" ✅ SS Local Test Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
