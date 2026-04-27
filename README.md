# cosmos-vanity-nvidia ⚡

NVIDIA/CUDA-focused Cosmos vanity address generation.

This repo is the split-out NVIDIA sibling of [`cosmos-vanity-amd`](https://github.com/n0sn0de/cosmos-vanity-amd). It keeps the shared CPU path and some inherited OpenCL code from the split, but the runtime path this repo actually targets and claims is Linux + NVIDIA + CUDA. If you want the AMD/OpenCL-focused repo, use `cosmos-vanity-amd`.

To keep the split pragmatic, the crate and binary names stay `cosmos-vanity-*` / `cosmos-vanity`.

## Honest support status

| Platform | Backend | Status | Evidence |
|---|---|---|---|
| Linux | CPU | build/test validated | `cargo test --workspace --features cuda` passed locally |
| Linux | CUDA | runtime-validated | `nosnode` (RTX 2070, driver `580.126.09`) found real `n0s` matches in raw and mnemonic modes with `"verified": true` |
| Linux | OpenCL | inherited, not claimed here | retained from the split, but this repo is not the AMD/OpenCL support target |
| Windows | CUDA | build-validated on native Windows | `win11` native `cargo build --release -p cosmos-vanity-cli --features cuda` passed on the RTX 3070 host, and `cosmos-vanity.exe --version` / `--help` succeeded |

Notes:
- I only claim runtime validation that was actually re-run for this repo split.
- Windows runtime wallet-generation correctness was not re-validated in this pass, so this README still makes no Windows runtime safety claim.
- The raw and mnemonic validation artifacts contained live secrets, so they were cleaned off the remote host after verification.

## Validation proof

Validated on `nosnode` with an NVIDIA GeForce RTX 2070.

Commands used:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

./cosmos-vanity search -p n0s --gpu-api cuda -m gpu --format json --log-level info
./cosmos-vanity search -p n0s --gpu-api cuda -m gpu -k mnemonic -w 12 --format json --log-level info
```

Observed proof snippets, secrets redacted:

```text
INFO CUDA device: NVIDIA GeForce RTX 2070 (SMs: 36, max threads/block: 768)
INFO 🎯 GPU Raw Match! Address: cosmos1n0saehgjx6f798ws2hwqpvt96x2kvxzjwqu9w4 (candidate #16085)
"verified": true

INFO CUDA mnemonic pipeline kernel compiled successfully
INFO 🎯 GPU Mnemonic Match! Address: cosmos1n0smv3p5lwqzw7fmnga9yfkd7505uzv66flyfm (candidate #3731)
"verified": true
```

## Build

### Linux CPU-only

```bash
cargo build --release -p cosmos-vanity-cli
```

### Linux CUDA

You need an NVIDIA driver plus NVRTC available at runtime.

```bash
cargo build --release -p cosmos-vanity-cli --features cuda
```

## Usage

```bash
# force CUDA on NVIDIA
cosmos-vanity search -p abc --gpu-api cuda

# full mnemonic pipeline on CUDA
cosmos-vanity search -p abc --gpu-api cuda -k mnemonic -w 12

# CPU only
cosmos-vanity search -p abc -m cpu -k mnemonic
```

## Validation commands

```bash
cargo test --workspace --features cuda
cargo build --release -p cosmos-vanity-cli --features cuda
```

## Security notes

- mnemonic or private key output gives full wallet control
- every reported match is verified again on CPU before output
- `zeroize` is used for sensitive material where practical
- GPU VRAM can retain sensitive data after execution, so production key generation should be treated accordingly

## License

MIT
