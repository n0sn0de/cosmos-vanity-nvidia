# cosmos-vanity-amd ⚡

Generate vanity wallet addresses for the Cosmos ecosystem using AMD GPU acceleration via ROCm/OpenCL.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (clap)                           │
│              Progress display, structured output            │
├───────────┬───────────┬───────────┬──────────┬──────────────┤
│  keyderiv │  address  │    gpu    │  verify  │    bench     │
│           │           │           │          │              │
│ BIP-39    │ SHA-256   │ OpenCL    │ CPU re-  │ Criterion    │
│ BIP-32/44 │ RIPEMD160 │ kernel    │ derive   │ benchmarks   │
│ secp256k1 │ Bech32    │ mgmt      │ & check  │              │
│           │           │           │          │              │
│ Mnemonic  │ Cosmos    │ ROCm/AMD  │ Every    │ Throughput   │
│ generation│ address   │ GPU batch │ GPU hit  │ measurement  │
│ & HD key  │ encoding  │ hashing   │ verified │              │
│ derivation│           │           │ on CPU   │              │
└───────────┴───────────┴───────────┴──────────┴──────────────┘
```

### Data Flow

```
1. CPU: Generate random BIP-39 mnemonic (24 words)
2. CPU: Derive secp256k1 keypair via BIP-44 (m/44'/118'/0'/0/0)
3. GPU: Batch SHA-256 + RIPEMD-160 hashing of public keys
4. GPU: Prefix/pattern matching on raw address bytes
5. CPU: Bech32 encode matched candidates
6. CPU: Full re-derivation verification of every match
7. Output: Address + mnemonic + derivation path
```

## Features

- **Vanity address generation** — find addresses matching custom prefix/suffix/contains/regex patterns
- **AMD GPU acceleration** — OpenCL kernels for parallel SHA-256 + RIPEMD-160 on ROCm
- **CPU fallback** — works without GPU, just slower
- **Multi-chain support** — configurable HRPs (cosmos, osmo, juno, stars, akash, etc.)
- **BIP-44 compliant** — standard Cosmos SDK derivation (coin type 118)
- **Verification** — every GPU match independently verified on CPU before reporting
- **Resumable** — save/restore search state across sessions
- **Structured output** — text or JSON output formats
- **Secure** — sensitive memory zeroized on drop

## Build Instructions (Ubuntu 24.04)

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build essentials
sudo apt install -y build-essential pkg-config libssl-dev
```

### CPU-only build (no GPU required)

```bash
cargo build --release
```

### GPU build (AMD ROCm/OpenCL)

#### ROCm Setup

```bash
# Add ROCm repository (Ubuntu 24.04)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
sudo dpkg -i amdgpu-install_6.4.60400-1_all.deb
sudo amdgpu-install --usecase=rocm,opencl

# Install OpenCL development headers
sudo apt install -y rocm-opencl-dev ocl-icd-opencl-dev

# Add user to render/video groups
sudo usermod -aG render,video $USER

# Verify
clinfo  # Should show your AMD GPU
```

#### Build with OpenCL

```bash
cargo build --release --features opencl
```

## Usage

### Search for a vanity address

```bash
# Find an address starting with "cosmos1aaa"
cosmos-vanity search --pattern aaa

# Different chain
cosmos-vanity search --pattern dead --hrp osmo

# Suffix matching
cosmos-vanity search --pattern 420 --match-type suffix

# Contains matching
cosmos-vanity search --pattern cafe --match-type contains

# Multiple matches with JSON output
cosmos-vanity search --pattern aa --max-matches 5 --format json

# Custom derivation path
cosmos-vanity search --pattern abc --path "m/44'/118'/1'/0/0"

# GPU accelerated (requires --features opencl build)
cosmos-vanity search --pattern aaaa --gpu

# Resumable search
cosmos-vanity search --pattern aaaaa --state-file search.json

# Control thread count
cosmos-vanity search --pattern aaa -j 8
```

### Generate a random address

```bash
cosmos-vanity generate
cosmos-vanity generate --hrp osmo
cosmos-vanity generate --format json
```

### Verify a mnemonic

```bash
cosmos-vanity verify \
  --mnemonic "word1 word2 ... word24" \
  --address "cosmos1abc..."
```

### Run benchmarks

```bash
cosmos-vanity bench --iterations 10000
```

## Pattern Difficulty

The Bech32 character set has 32 characters. Expected attempts for prefix matches:

| Prefix Length | Expected Attempts | ~Time (1000 addr/s) | ~Time (1M addr/s GPU) |
|---------------|-------------------|---------------------|-----------------------|
| 1 char        | 32                | <1s                 | <1ms                  |
| 2 chars       | 1,024             | ~1s                 | ~1ms                  |
| 3 chars       | 32,768            | ~33s                | ~33ms                 |
| 4 chars       | 1,048,576         | ~17min              | ~1s                   |
| 5 chars       | 33,554,432        | ~9h                 | ~34s                  |
| 6 chars       | 1,073,741,824     | ~12 days            | ~18min                |
| 7 chars       | 34,359,738,368    | ~1 year             | ~10h                  |

## Security Considerations

### ⚠️ CRITICAL

1. **Mnemonic safety** — The mnemonic phrase gives full control of funds. Never share it, log it, or store it unencrypted.

2. **Memory handling** — This tool uses [`zeroize`](https://crates.io/crates/zeroize) to clear sensitive data from memory when no longer needed. However:
   - The OS may swap memory to disk
   - Core dumps could contain secrets
   - Compiler optimizations could copy data

3. **Recommended practices:**
   - Run on an air-gapped machine
   - Disable swap: `sudo swapoff -a`
   - Disable core dumps: `ulimit -c 0`
   - Clear terminal history after use
   - Use encrypted disk
   - Verify the generated address with an independent tool before funding

4. **GPU memory** — GPU VRAM may retain data after kernel execution. Power cycle the GPU after generating production keys.

5. **Entropy** — We use the system's CSPRNG via the `rand` crate's `OsRng`. Ensure your system has adequate entropy (most modern Linux systems do via `/dev/urandom`).

6. **Verification** — Every match is independently re-derived and verified on CPU. Never trust a GPU result without verification.

## Development

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p cosmos-vanity-keyderiv
cargo test -p cosmos-vanity-address

# Run benchmarks
cargo bench -p cosmos-vanity-bench

# Check without building (no GPU needed)
cargo check --workspace

# Check with GPU features
cargo check --workspace --features cosmos-vanity-gpu/opencl

# Lint
cargo clippy --workspace
```

## License

MIT

## Contributing

PRs welcome. Security-sensitive changes require extra review.
