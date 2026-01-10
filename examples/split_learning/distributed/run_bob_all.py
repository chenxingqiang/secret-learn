#!/usr/bin/env python3
"""
Bob server for testing SL examples (run on remote server 47.80.8.23)

Usage:
    python run_bob_all.py --alice_addr <ALICE_IP>:9497
"""

import argparse
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


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alice_addr', required=True, help='Alice address (e.g., 192.168.1.100:9497)')
    parser.add_argument('--bob_port', type=int, default=9498)
    args = parser.parse_args()
    
    local_ip = get_local_ip()
    print(f"Bob Server Starting...")
    print(f"  Local IP: {local_ip}:{args.bob_port}")
    print(f"  Alice: {args.alice_addr}")
    
    cluster_config = {
        'parties': {
            'alice': {'address': args.alice_addr, 'listen_addr': f'0.0.0.0:{args.alice_addr.split(":")[-1]}'},
            'bob': {'address': f'{local_ip}:{args.bob_port}', 'listen_addr': f'0.0.0.0:{args.bob_port}'},
        },
        'self_party': 'bob'
    }
    
    print(f"\n⏳ Initializing SecretFlow (waiting for Alice)...")
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    print("✓ Connected!")
    
    bob = sf.PYU('bob')
    
    # Create Bob's data partition
    np.random.seed(43)
    X_bob = np.random.randn(500, 7).astype(np.float32)
    
    fed_X = FedNdarray(
        partitions={bob: bob(lambda x: x)(X_bob)},
        partition_way=PartitionWay.VERTICAL
    )
    
    print("\n✓ Bob ready and participating in SL training")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sf.shutdown()
        print("✓ Bob stopped")


if __name__ == "__main__":
    main()
