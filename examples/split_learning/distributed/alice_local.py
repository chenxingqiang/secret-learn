#!/usr/bin/env python3
"""
Alice (local) for distributed SL testing
Via SSH tunnel: remote:9497 -> local:9497, remote:9498 -> remote:9498
"""

import sys
import time
import numpy as np

try:
    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.distributed.const import DISTRIBUTION_MODE
except ImportError:
    print("❌ SecretFlow not installed")
    sys.exit(1)


def main():
    REMOTE_IP = "47.80.8.23"
    
    print("="*60)
    print(" Alice (Local) - Distributed SL Test")
    print("="*60)
    
    # Via SSH tunnel: remote sees localhost:9497
    cluster_config = {
        'parties': {
            'alice': {'address': 'localhost:9497', 'listen_addr': '0.0.0.0:9497'},
            'bob': {'address': f'{REMOTE_IP}:9498', 'listen_addr': f'0.0.0.0:9498'},
        },
        'self_party': 'alice'
    }
    
    print(f"\n[1/4] Initializing SecretFlow...")
    print(f"  Alice: localhost:9497 (via SSH tunnel)")
    print(f"  Bob: {REMOTE_IP}:9498")
    
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    print("  ✓ SecretFlow initialized")
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    
    print(f"\n[2/4] Creating sample data...")
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X_alice = np.random.randn(n_samples, n_features // 2).astype(np.float32)
    X_bob = np.random.randn(n_samples, n_features // 2).astype(np.float32)
    y = (np.random.randn(n_samples) > 0).astype(np.float32)
    
    print(f"  Alice data: {X_alice.shape}")
    print(f"  Bob data: {X_bob.shape}")
    
    print(f"\n[3/4] Creating federated data...")
    fed_X = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(X_alice),
            bob: bob(lambda x: x)(X_bob),
        },
        partition_way=PartitionWay.VERTICAL
    )
    fed_y = FedNdarray(
        partitions={alice: alice(lambda x: x)(y)},
        partition_way=PartitionWay.HORIZONTAL
    )
    print("  ✓ Federated data created")
    
    print(f"\n[4/4] Testing SL model (LinearRegression)...")
    from secretlearn.split_learning.linear_models.linear_regression import SLLinearRegression
    
    devices = {'alice': alice, 'bob': bob}
    model = SLLinearRegression(devices, aggregation='mean')
    
    start = time.time()
    model.fit(fed_X, fed_y)
    duration = (time.time() - start) * 1000
    
    print(f"  ✓ Training completed in {duration:.1f}ms")
    
    print("\n" + "="*60)
    print(" ✅ Distributed SL Test PASSED!")
    print("="*60)
    
    sf.shutdown()


if __name__ == "__main__":
    main()
