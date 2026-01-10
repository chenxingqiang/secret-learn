# Split Learning Distributed Setup

This directory contains scripts for running Split Learning across multiple machines.

## Architecture

```
┌─────────────────────────┐         ┌─────────────────────────┐
│    Local Machine        │         │   Remote Server         │
│    (Alice)              │◄───────►│   (Bob)                 │
│                         │  gRPC   │   47.80.8.23            │
│  - Features: X_alice    │         │  - Features: X_bob      │
│  - Labels: y            │         │                         │
│  - Model part: W_alice  │         │  - Model part: W_bob    │
└─────────────────────────┘         └─────────────────────────┘
```

## Prerequisites

Both machines need:
```bash
pip install secret-learn secretflow
```

## Quick Start

### Step 1: Start Bob on Remote Server

SSH into the remote server and run:
```bash
ssh root@47.80.8.23
cd /path/to/secret-learn
python examples/SL/distributed/bob_remote.py --alice_addr <YOUR_LOCAL_IP>:9497
```

### Step 2: Start Alice on Local Machine

On your local machine:
```bash
python examples/SL/distributed/alice_local.py --bob_addr 47.80.8.23:9498
```

## Network Requirements

1. **Firewall**: Ensure ports 9497 and 9498 are open on both machines
2. **Connectivity**: Both machines must be able to reach each other

### Test connectivity:
```bash
# From local machine
nc -zv 47.80.8.23 9498

# From remote server
nc -zv <YOUR_LOCAL_IP> 9497
```

## Configuration Options

### Alice (Local)
```bash
python alice_local.py --help

Options:
  --bob_addr      Bob server address (default: 47.80.8.23:9498)
  --alice_port    Alice listen port (default: 9497)
```

### Bob (Remote)
```bash
python bob_remote.py --help

Options:
  --alice_addr    Alice server address (required)
  --bob_port      Bob listen port (default: 9498)
```

## Troubleshooting

### Connection refused
- Check firewall rules
- Ensure correct IP addresses
- Verify ports are not in use

### Timeout
- Check network connectivity
- Ensure both parties use matching cluster configuration
- Start Bob before Alice

### SecretFlow errors
- Ensure same SecretFlow version on both machines
- Use Python 3.10 or 3.11
