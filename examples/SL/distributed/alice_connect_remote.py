#!/usr/bin/env python3
"""
Alice (local) connects to Bob on remote server (47.80.8.23)
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
    print(" Alice (Local) -> Bob (Remote) Distributed SL Test")
    print("="*60)
    
    cluster_config = {
        'parties': {
            'alice': {'address': '0.0.0.0:19497', 'listen_addr': '0.0.0.0:19497'},
            'bob': {'address': f'{REMOTE_IP}:19498', 'listen_addr': f'0.0.0.0:19498'},
        },
        'self_party': 'alice'
    }
    
    print(f"\n[1/4] Initializing SecretFlow...")
    print(f"  Alice: 0.0.0.0:19497 (local)")
    print(f"  Bob: {REMOTE_IP}:19498 (remote Docker)")
    
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    print("  ✓ SecretFlow initialized, connected to Bob!")
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    
    print(f"\n[2/4] Creating sample data...")
    np.random.seed(42)
    n_samples = 100
    
    X_alice = np.random.randn(n_samples, 5).astype(np.float32)
    X_bob = np.random.randn(n_samples, 5).astype(np.float32)
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
    
    print(f"\n[4/4] Testing distributed SL LinearRegression...")
    sys.path.insert(0, '/Users/xingqiangchen/secret-learn')
    from secretlearn.SL.linear_models.linear_regression import SLLinearRegression
    
    devices = {'alice': alice, 'bob': bob}
    model = SLLinearRegression(devices, aggregation='mean')
    
    start = time.time()
    model.fit(fed_X, fed_y)
    duration = (time.time() - start) * 1000
    
    print(f"  ✓ Training completed in {duration:.1f}ms")
    
    print("\n" + "="*60)
    print(" ✅ Distributed SL Test PASSED!")
    print("    Alice (local) <-> Bob (47.80.8.23)")  
    print("="*60)
    
    sf.shutdown()


if __name__ == "__main__":
    main()
