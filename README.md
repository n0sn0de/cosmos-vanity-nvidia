# cosmos-vanity-nvidia ⚡

NVIDIA/CUDA-focused Cosmos vanity address generation.

This repo is the split-out NVIDIA sibling of [`cosmos-vanity-amd`](https://github.com/n0sn0de/cosmos-vanity-amd). It keeps the shared CPU path and some inherited OpenCL code from the split, but the runtime paths this repo actually targets and claims are NVIDIA + CUDA on Linux and Windows. If you want the AMD/OpenCL-focused repo, use `cosmos-vanity-amd`.

To keep the split pragmatic, the crate and binary names stay `cosmos-vanity-*` / `cosmos-vanity`.

## Honest support status

| Platform | Backend | Status | Evidence |
|---|---|---|---|
| Linux | CPU | build/test validated | `cargo test --workspace --features cuda` passed locally |
| Linux | CUDA | runtime-validated | `nosnode` (RTX 2070, driver `580.126.09`) found real `n0s` matches in raw and mnemonic modes with `"verified": true` |
| Linux | OpenCL | inherited, not claimed here | retained from the split, but this repo is not the AMD/OpenCL support target |
| Windows | CUDA | native build + runtime validated | `win11` (RTX 3070) passed `cargo build --release -p cosmos-vanity-cli --features cuda`, returned a real raw `n0s` match with `"verified": true`, and passed the targeted ignored CUDA correctness tests listed below |

Notes:
- I only claim runtime validation that was actually re-run for this repo split.
- The Windows runtime proof on `win11` succeeded through PTX fallback on `compute_80` after the host NVRTC path rejected `compute_86` and did not expose the newer CUBIN APIs. That is still a real native Windows CUDA validation, but it is not a claim that every Windows toolkit/NVRTC combination will take the same compile path.
- On Windows, run the ignored CUDA tests one-by-one with `--test-threads=1`. Letting Cargo fire multiple GPU correctness tests at once is a great way to create fake hangs.
- The raw and mnemonic validation artifacts contained live secrets, so they were cleaned off the remote host after verification.

## Validation proof

### Linux (`nosnode`, RTX 2070)

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

### Windows (`win11`, RTX 3070)

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

Observed proof snippets, secrets redacted:

```text
INFO Loaded CUDA module vanity_search.cu via PTX (compute_80) in 0.36s
INFO CUDA device: NVIDIA GeForce RTX 3070 (SMs: 46, max threads/block: 1024)
INFO Loaded CUDA module secp256k1.cu via PTX (compute_80) in 6.21s
INFO 🎯 GPU Raw Match! Address: cosmos1n0see5yjfc56cf2w2vqn4ne0hpy7h3wnw2gj2p (candidate #10023)
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

### Windows CUDA

Use a shell primed by MSVC first, then put the CUDA toolkit `bin` directory on `PATH`.

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

cargo build --release -p cosmos-vanity-cli --features cuda
```

If your local NVRTC does not expose the newer CUBIN APIs, the runtime can still succeed through PTX fallback. That is what the native `win11` validation used here.

If the binary logs `nvcc fatal   : Cannot find compiler 'cl.exe' in PATH` while probing the external CUBIN path, that probe may fail and fall through to PTX anyway. Treat the successful path as the one the logs actually show, not the one you hoped it took.

## Usage

```bash
# force CUDA raw mode on NVIDIA
cosmos-vanity search -p abc --gpu-api cuda -m gpu -k raw

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
