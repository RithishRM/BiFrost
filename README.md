# BiFrost

**A privacy-preserving federated learning engine in Rust**, combining differentially-private local training, gradient compression, error-feedback correction, and Byzantine-robust aggregation over a real gRPC-based network transport.

BiFrost simulates a federated learning system where edge nodes train a small recurrent model locally on network traffic data, compress and privatize their gradients before transmission, and a central hub aggregates updates from multiple nodes while filtering out adversarial/corrupted contributions — without ever seeing raw client data.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Validated Results](#validated-results)
- [Design Notes](#design-notes)
- [Roadmap](#roadmap)

---

## Overview

BiFrost is built around a simple idea: many devices (edge nodes) hold sensitive local data (here, network traffic flow records) and want to collaboratively train a shared model **without sending raw data anywhere**. Each node instead:

1. Trains a small local model on its own data
2. Computes gradients via backpropagation
3. Clips and privatizes those gradients with differential privacy (DP-SGD)
4. Compresses them via Top-k sparsification, banking the untransmitted remainder as *residue* for future rounds (error-feedback)
5. Streams the compressed update to a central hub over gRPC

The hub then:

1. Buffers incoming updates and anonymizes their origin via a MixNet-style shuffle
2. Runs Byzantine-robust aggregation (Krum) to exclude outlier/adversarial contributions
3. Returns consensus "master weights" back to participating nodes

## Architecture

```
                     ┌──────────────────────────────┐
                     │        Edge Node(s)          │
                     │                              │
  Raw traffic CSV →  │  DP-SGD Local Training       │
                     │  → L2-Norm Gradient Clipping │
                     │  → Gaussian Noise Injection  │
                     │  → Top-k Sparsification      │
                     │  → Error-Feedback Residue    │
                     │    Bank                      │  
                     └───────────────┬──────────────┘
                                     │ gRPC (streamed)
                                     ▼
                     ┌─────────────────────────────────┐
                     │      Operational Hub (Server)   │
                     │                                 │
                     │  MixNet Shuffle (anonymization) │
                     │  → Parallel Krum Aggregation    │
                     │     (Byzantine-fault tolerant)  │
                     │  → Master Weight Consensus      │
                     └───────────────┬─────────────────┘
                                     │
                                     ▼
                     Nodes overwrite local weights with consensus
```

## Features

| Component | Description |
|---|---|
| **DP-SGD Training** | Small hand-rolled recurrent model trained per-node with differentially private gradients (L2 clipping + calibrated Gaussian noise) |
| **Gradient Compression** | Top-k sparsification transmits only the most significant ~10% of parameters per round |
| **Error Feedback** | Untransmitted gradient mass is banked as residue and re-injected in the following round, preventing information loss from compression |
| **Byzantine-Robust Aggregation** | Krum-based node scoring excludes adversarial/corrupted updates from the aggregate |
| **Anonymization** | A MixNet-style shuffler decouples update order from node identity before aggregation |
| **Real Network Transport** | Bidirectional gRPC streaming (Tonic/Prost) between client nodes and the hub |
| **P2P-ready** | `libp2p` included as a dependency for future peer discovery / gossip-based extensions |

## Project Structure

```
bifrost_core/
├── Cargo.toml
├── build.rs                     # Compiles bifrost.proto via tonic-build
├── proto/
│   └── bifrost.proto             # gRPC service & message definitions
├── src/
│   ├── lib.rs                    # Exposes core modules as a library
│   ├── main.rs                   # Full networked system run (training + gRPC)
│   ├── engine.rs                 # Model, DP-SGD, clipping, top-k, residue logic
│   ├── protocol.rs                # GradientUpdate wire format
│   ├── network.rs                 # gRPC server/client implementation
│   ├── aggregation.rs              # Krum Byzantine-robust aggregation
│   ├── shuffler.rs                  # MixNet-style anonymizing shuffle
│   └── bin/
│       ├── mock_node.rs              # Local-only pipeline test (no networking)
│       ├── byzantine_test.rs          # Multi-node gRPC Krum validation
│       └── krum_direct_test.rs         # Deterministic Krum proof (no shuffle)
├── data/
│   └── MachineLearningCVE/            # CICIDS2017 dataset (gitignored, see below)
└── payloads/
    └── sample_gradient_update.json     # Example serialized payload
```

## Getting Started

### Prerequisites

- Rust (stable toolchain, edition 2024)
- `cargo`

### Build

```bash
cargo build
```

## Dataset Setup

BiFrost trains its local model on real network traffic data from the **[CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)**.

1. Download `MachineLearningCSV.zip` from the CICIDS2017 dataset page (under `MachineLearningCVE`)
2. Unzip its contents into `data/MachineLearningCVE/` at the project root, so you have files like:

```
data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv
data/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv
data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv
...
```

> **Note:** The dataset is not committed to this repository (see `.gitignore`) due to its size. You must download it separately before running any binary that trains on real data.

## Running the Project

### 1. Full networked system (training + gRPC + Krum, single node)

```bash
cargo run --bin bifrost_core
```

Boots a local gRPC hub, trains on real traffic data for 3 rounds, compresses and transmits gradients, and applies consensus weight updates.

### 2. Local pipeline only (no networking)

```bash
cargo run --bin mock_node
```

Runs data ingestion → DP-SGD training → error-feedback → Top-k packing → serialized output, entirely offline. Writes a sample payload to `payloads/sample_gradient_update.json`.

### 3. Byzantine-robustness validation (multi-node, real gRPC)

```bash
cargo run --bin byzantine_test
```

Spins up 5 concurrent simulated nodes (4 honest, 1 adversarial) against a real gRPC server with `byzantine_bounds = 1`, triggering genuine Krum filtering instead of the insufficient-node fallback.

### 4. Deterministic Krum proof (no networking, no shuffle)

```bash
cargo run --bin krum_direct_test
```

Calls the Krum aggregation function directly on a known-order set of vectors (bypassing the anonymizing shuffle), then matches the selected output back to its known label — providing unambiguous proof of correct adversarial exclusion.

## Testing

Run the full unit test suite:

```bash
cargo test -- --nocapture
```

Covers:
- Gradient clipping under adversarial/exploding input
- Exact Top-k compression sizing
- Multi-round error-feedback residue accumulation
- Data ingestion, preprocessing, and full local training pipeline
- Gradient update serialization/deserialization
- Krum distance-scoring correctness

## Validated Results

These are real, reproducible results from running this project against the actual CICIDS2017 dataset (Monday-WorkingHours):

- **8 / 8 unit tests passing**
- **529,916 real sequence samples** ingested and trained on from real network flow data
- **~90% gradient compression** achieved consistently (11 / 113 parameters transmitted per round)
- **Error-feedback residue accumulates correctly** across rounds rather than discarding untransmitted gradient mass
- **Byzantine-robust aggregation validated deterministically**: in a 5-node test with 1 adversarial node (gradients scaled to be extreme outliers), Krum correctly excluded the adversarial vector — pairwise distances showed the adversarial node was **~10 orders of magnitude** farther from the honest cluster than honest nodes were from each other, and Krum's selection was confirmed by exact vector matching (not just score inference)

## Design Notes

- The local model is a small hand-rolled recurrent network (113 total parameters: input/hidden/output weight matrices + biases) trained on 3-step sequence windows of 4 traffic features (destination port, flow duration, forward/backward packet counts).
- DP-SGD uses L2-norm gradient clipping followed by calibrated Gaussian noise injection, parameterized by an epsilon/delta privacy budget.
- Krum's aggregation requires `n > 2 * byzantine_bounds + 2` participating nodes to actually run its filtering logic; with fewer nodes, it safely falls back to returning the raw batch unfiltered (see `byzantine_test.rs` and `krum_direct_test.rs` for validation of the real filtering path).
- The MixNet shuffler intentionally decouples submission order from Krum's node index in its output — this is by design (anonymization), which is why `krum_direct_test.rs` exists as a separate, shuffle-free proof.

## Roadmap

- [ ] Wire `libp2p` for genuine peer-to-peer node discovery instead of a fixed gRPC hub
- [ ] Add configurable row-limiting for faster iteration on large dataset files
- [ ] Extend Krum validation to sweep multiple `byzantine_bounds` / adversary-count combinations
- [ ] Add throughput/latency benchmarks (e.g. via `criterion`) alongside existing correctness tests
- [ ] Support multiple CICIDS2017 daily files (attack-specific days) for broader local training coverage
