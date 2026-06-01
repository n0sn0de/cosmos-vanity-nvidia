# cosmos-vanity-nvidia ⚡

NVIDIA/CUDA-focused Cosmos vanity address generation.

This repo is the split-out NVIDIA sibling of [`cosmos-vanity-amd`](https://github.com/n0sn0de/cosmos-vanity-amd). It keeps the shared CPU path and some inherited OpenCL code from the split, but the runtime paths this repo actually targets and claims are NVIDIA + CUDA on Linux and Windows. If you want the AMD/OpenCL-focused repo, use `cosmos-vanity-amd`.

To keep the split pragmatic, the crate and binary names stay `cosmos-vanity-*` / `cosmos-vanity`.

## Honest support status

| Platform | Backend | Status | Evidence |
|---|---|---|---|
| Linux | CPU | build/test validated | `cargo test --workspace --features cuda` |
| Linux | CUDA | runtime-validated | raw and mnemonic CUDA searches produced verified matches on NVIDIA hardware |
| Linux | OpenCL | inherited, not claimed here | retained from the split, but this repo is not the AMD/OpenCL support target |
| Windows | CUDA | native build + runtime validated | native CUDA build, raw search, and targeted ignored CUDA correctness tests passed on NVIDIA hardware |

Notes:
- I only claim runtime validation that was actually re-run for this repo split.
- The Windows runtime proof succeeded through PTX fallback on `compute_80` after the host NVRTC path rejected the device-specific target and did not expose the newer CUBIN APIs. That is still a real native Windows CUDA validation, but it is not a claim that every Windows toolkit/NVRTC combination will take the same compile path.
- On Windows, run the ignored CUDA tests one-by-one with `--test-threads=1`. Letting Cargo fire multiple GPU correctness tests at once is a great way to create fake hangs.

## Validation status

### Linux

CPU and CUDA validation:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cargo test --workspace --features cuda
cargo build --release -p cosmos-vanity-cli --features cuda
./cosmos-vanity search -p n0s --gpu-api cuda -m gpu --format json --log-level info
./cosmos-vanity search -p n0s --gpu-api cuda -m gpu -k mnemonic -w 12 --format json --log-level info
```

### Windows

Run from a Developer Command Prompt or from `cmd.exe` after `vcvars64.bat`:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

cargo build --release -p cosmos-vanity-cli --features cuda
cargo test -p cosmos-vanity-gpu cuda::tests::test_cuda_secp256k1_known_vector --features cuda -- --ignored --exact --test-threads=1
cargo test -p cosmos-vanity-gpu cuda::tests::test_cuda_secp256k1_matches_cpu --features cuda -- --ignored --exact --test-threads=1
cargo test -p cosmos-vanity-gpu cuda::tests::test_cuda_mnemonic_pipeline --features cuda -- --ignored --exact --test-threads=1

target\release\cosmos-vanity.exe search -p n0s --gpu-api cuda -m gpu --format json --log-level info
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

### Windows CUDA

Use a shell primed by MSVC first, then put the CUDA toolkit `bin` directory on `PATH`.

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

cargo build --release -p cosmos-vanity-cli --features cuda
```

If your local NVRTC does not expose the newer CUBIN APIs, the runtime can still succeed through PTX fallback.

If the binary logs `nvcc fatal   : Cannot find compiler 'cl.exe' in PATH` while probing the external CUBIN path, that probe may fail and fall through to PTX anyway. Treat the successful path as the one the logs actually show, not the one you hoped it took.

## Usage

```bash
# list visible CUDA GPUs and exit
cosmos-vanity search --list-gpus --format json

# force CUDA raw mode on the default single GPU (backward-compatible behavior)
cosmos-vanity search -p abc --gpu-api cuda -m gpu -k raw

# pin CUDA to one device explicitly
cosmos-vanity search -p abc --gpu-api cuda -m gpu -k raw --gpu-devices 0

# fan one raw search across every visible NVIDIA GPU
cosmos-vanity search -p abc --gpu-api cuda -m gpu -k raw --gpu-devices all

# select a mixed subset of GPUs explicitly
cosmos-vanity search -p abc --gpu-api cuda -m hybrid -k mnemonic -w 12 --gpu-devices 0,1,3

# full mnemonic pipeline on CUDA
cosmos-vanity search -p abc --gpu-api cuda -m gpu -k mnemonic -w 12

# CPU only mnemonic mode
cosmos-vanity search -p abc -m cpu -k mnemonic

# write secrets to a new JSON Lines file instead of stdout
cosmos-vanity search -p abc -m cpu -k mnemonic --secret-output-file ./vanity-secrets.jsonl
cosmos-vanity generate --secret-output-file ./generated-secret.jsonl

# verify without putting the mnemonic in argv/shell history
cosmos-vanity verify --mnemonic-file ./mnemonic.txt --address cosmos1...

# opt into printing wallet secrets to stdout (unsafe)
cosmos-vanity generate --unsafe-print-secrets
```

Notes:
- `--gpu-devices` is CUDA-only and requires `--gpu-api cuda` plus `--mode gpu` or `--mode hybrid`
- omitting `--gpu-devices` keeps the old single-GPU CUDA default

## Validation commands

### Linux

```bash
cargo test --workspace --features cuda
cargo build --release -p cosmos-vanity-cli --features cuda
```

### Windows

```bat
cargo build --release -p cosmos-vanity-cli --features cuda
cargo test -p cosmos-vanity-gpu cuda::tests::test_cuda_secp256k1_known_vector --features cuda -- --ignored --exact --test-threads=1
cargo test -p cosmos-vanity-gpu cuda::tests::test_cuda_secp256k1_matches_cpu --features cuda -- --ignored --exact --test-threads=1
cargo test -p cosmos-vanity-gpu cuda::tests::test_cuda_mnemonic_pipeline --features cuda -- --ignored --exact --test-threads=1
target\release\cosmos-vanity.exe search -p n0s --gpu-api cuda -m gpu --format json --log-level info
```

## Security notes

- mnemonic/private-key output is redacted by default; use `--secret-output-file <path>` to write secrets to a new restrictive-permission file, or `--unsafe-print-secrets` to print them to stdout
- `verify --mnemonic-file <path>` avoids putting mnemonics in argv/shell history; legacy `--mnemonic` still works but is hidden and warns
- raw key mode is only allowed on the pure GPU raw path; CPU, hybrid, and GPU fallback paths must use mnemonic mode
- every reported match is verified again on CPU before output; failed verification results are skipped, not counted
- mnemonic word count is restricted to 12 or 24 words
- `zeroize` is used for sensitive material where practical, including additional cleanup for GPU mnemonic host buffers
- GPU VRAM can retain sensitive data after execution, so production key generation should be treated accordingly

## License

MIT
