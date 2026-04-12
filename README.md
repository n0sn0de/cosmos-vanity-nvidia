# cosmos-vanity-amd ⚡

Generate Cosmos vanity addresses with CPU, OpenCL, or CUDA backends.

This repo contains:
- CPU search and verification paths
- OpenCL kernels for AMD and other OpenCL-capable GPUs
- CUDA/NVRTC support for NVIDIA GPUs, reusing the same kernel bodies through a compatibility preamble
- CI and release automation for Linux and Windows builds

## Honest support status

| Platform | Backend | Status | Evidence |
|---|---|---|---|
| Linux | CPU | validated | `cargo test --workspace` passes locally |
| Linux | OpenCL | validated on AMD | `cargo test --workspace --features "opencl cuda"` passes on an RX 9070 XT |
| Linux | CUDA | compile-validated, runtime tests auto-skip without driver | `cargo test --workspace --features cuda` passes locally; CUDA parity tests run automatically when a CUDA driver is present |
| Windows (MSVC) | CPU | build-validated | `cargo xwin build --release --target x86_64-pc-windows-msvc -p cosmos-vanity-cli` with `clang-cl`, `lld-link`, and `llvm-ar` available on `PATH` |
| Windows (MSVC) | CUDA | build-validated | `cargo xwin build --release --target x86_64-pc-windows-msvc -p cosmos-vanity-cli --features cuda` with `clang-cl`, `lld-link`, and `llvm-ar` available on `PATH` |

Notes:
- I only claim runtime validation where it was actually exercised.
- Remote SSH access to the available NVIDIA/Windows hosts was not usable from this environment during this pass, so those hosts were not counted as validation evidence.
- GitHub Actions now covers Linux and Windows build/test automation so future regressions are visible quickly.

## Performance

Benchmarked on AMD Radeon RX 9070 XT (gfx1201, 32 CUs, ROCm/OpenCL):

| Mode | Throughput | Description |
|---|---:|---|
| GPU raw | ~1,000,000/s | Private key → secp256k1 → hash on GPU |
| GPU mnemonic | ~21,000/s | Full BIP-39 + BIP-32 + secp256k1 pipeline on GPU |
| CPU mnemonic | ~12,000/s | CPU-only search |

## Backend model

### OpenCL
- primary validated runtime path on Linux/AMD
- supports raw key search and mnemonic pipeline
- runtime tests now skip cleanly on machines without an OpenCL GPU

### CUDA
- NVIDIA path via CUDA driver + NVRTC
- compiles the same OpenCL kernel bodies through a CUDA compatibility preamble
- parity tests exist for hashing, secp256k1, and mnemonic pipeline
- runtime tests skip cleanly when no CUDA driver/device is available

## Build

### Linux CPU-only

```bash
cargo build --release -p cosmos-vanity-cli
```

### Linux OpenCL

Ubuntu/Debian packages:

```bash
sudo apt update
sudo apt install -y ocl-icd-opencl-dev clinfo
cargo build --release -p cosmos-vanity-cli --features opencl
```

### Linux CUDA

You need an NVIDIA driver plus NVRTC available at runtime.

```bash
cargo build --release -p cosmos-vanity-cli --features cuda
```

### Linux multi-backend build

```bash
cargo build --release -p cosmos-vanity-cli --features "opencl cuda"
```

### Native Windows (MSVC)

Install:
- Rust with the `x86_64-pc-windows-msvc` target
- Visual Studio Build Tools with the C++ workload
- NVIDIA driver / CUDA toolkit if you want the CUDA backend at runtime

CPU build:

```powershell
cargo build --release -p cosmos-vanity-cli
```

CUDA build:

```powershell
cargo build --release -p cosmos-vanity-cli --features cuda
```

### Cross-building Windows from Linux

`cargo xwin` still needs an LLVM toolchain that provides `clang-cl`, `lld-link`, and an archiver such as `llvm-ar` on `PATH`.
On this machine that came from `/opt/rocm-7.2.0/lib/llvm/bin`.

```bash
cargo install cargo-xwin --locked
export PATH="/opt/rocm-7.2.0/lib/llvm/bin:$PATH"
export CC_x86_64_pc_windows_msvc=clang-cl
export CXX_x86_64_pc_windows_msvc=clang-cl
export AR_x86_64_pc_windows_msvc=llvm-ar
export TARGET_AR=llvm-ar

env 'AR_x86_64-pc-windows-msvc=llvm-ar'     'CC_x86_64-pc-windows-msvc=clang-cl'     'CXX_x86_64-pc-windows-msvc=clang-cl'     cargo xwin build --release --target x86_64-pc-windows-msvc -p cosmos-vanity-cli

env 'AR_x86_64-pc-windows-msvc=llvm-ar'     'CC_x86_64-pc-windows-msvc=clang-cl'     'CXX_x86_64-pc-windows-msvc=clang-cl'     cargo xwin build --release --target x86_64-pc-windows-msvc -p cosmos-vanity-cli --features cuda
```

## Usage

```bash
# default pure GPU mode when a compiled backend is available, otherwise CPU fallback
cosmos-vanity search -p abc

# force CUDA on NVIDIA
cosmos-vanity search -p abc --gpu-api cuda

# force OpenCL
cosmos-vanity search -p abc --gpu-api opencl

# mnemonic output instead of raw private key
cosmos-vanity search -p abc -k mnemonic

# CPU only
cosmos-vanity search -p abc -m cpu -k mnemonic
```

## Validation commands

```bash
# baseline
cargo test --workspace

# CUDA compile surface, runtime tests skip when no CUDA driver is present
cargo test --workspace --features cuda

# full Linux feature surface, runtime GPU tests skip when hardware is absent
cargo test --workspace --features "opencl cuda"

# lint and formatting
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features "opencl cuda" -- -D warnings
```

## CI and releases

This repo now includes:
- `.github/workflows/ci.yml`
  - Ubuntu: fmt, clippy, default tests, CUDA feature tests, OpenCL+CUDA feature tests
  - Windows: default tests and CUDA feature tests
- `.github/workflows/release.yml`
  - builds tagged Linux and Windows release artifacts
  - publishes zipped/tarred binaries to GitHub Releases

Release artifacts are intended to be:
- `cosmos-vanity-linux-x86_64-cpu.tar.gz`
- `cosmos-vanity-linux-x86_64-gpu.tar.gz`
- `cosmos-vanity-windows-x86_64-cpu.zip`
- `cosmos-vanity-windows-x86_64-cuda.zip`

## Security notes

- mnemonic or private key output gives full wallet control
- every reported match is verified again on CPU before output
- `zeroize` is used for sensitive material where practical
- GPU VRAM can retain sensitive data after execution, so production key generation should be treated accordingly

## License

MIT
