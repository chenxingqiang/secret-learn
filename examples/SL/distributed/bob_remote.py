#!/usr/bin/env python3
"""
Bob (remote) for distributed SL testing
Run this on remote server (47.80.8.23)

Via SSH tunnel: localhost:9497 connects to Alice's local machine
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
    print("="*60)
    print(" Bob (Remote) - Distributed SL Test")
    print("="*60)
    
    # Via SSH tunnel: localhost:9497 reaches Alice
    cluster_config = {
        'parties': {
            'alice': {'address': 'localhost:9497', 'listen_addr': '0.0.0.0:9497'},
            'bob': {'address': '0.0.0.0:9498', 'listen_addr': '0.0.0.0:9498'},
        },
        'self_party': 'bob'
    }
    
    print(f"\n[1/3] Initializing SecretFlow...")
    print(f"  Alice: localhost:9497 (via SSH tunnel)")
    print(f"  Bob: 0.0.0.0:9498")
    
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    print("  ✓ SecretFlow initialized")
    
    bob = sf.PYU('bob')
    
    print(f"\n[2/3] Creating Bob's data...")
    np.random.seed(43)
    n_samples = 100
    n_features = 5
    
    X_bob = np.random.randn(n_samples, n_features).astype(np.float32)
    print(f"  Bob data: {X_bob.shape}")
    
    print(f"\n[3/3] Waiting for Alice's training request...")
    print("  (Press Ctrl+C to stop)")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        sf.shutdown()
        print("✓ Bob stopped")


if __name__ == "__main__":
    main()
