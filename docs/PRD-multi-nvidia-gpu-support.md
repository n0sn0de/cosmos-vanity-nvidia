# PRD: Multi-NVIDIA GPU Search Support

## Summary

Add real single-process multi-GPU support to `cosmos-vanity-nvidia` so one search can use multiple visible NVIDIA GPUs at once, including mixed-card systems.

## Problem

The current CUDA runtime initializes only one GPU context and runs one GPU driver loop, so hosts with multiple NVIDIA GPUs leave performance on the table.

## Goals

- Use multiple visible NVIDIA GPUs in one search run.
- Support mixed GPU models and generations.
- Keep correctness identical to current single-GPU behavior.
- Preserve existing CPU / single-GPU flows unless multi-GPU is explicitly selected or auto-enabled.

## Non-Goals

- Cross-vendor orchestration.
- Distributed multi-host searching.
- Perfect dynamic auto-tuning in the first iteration.

## User Experience

### New CLI behavior

Add device selection controls for CUDA searches:

- `--gpu-devices all` → use all visible CUDA devices
- `--gpu-devices 0,1,3` → use only selected device indices
- `--gpu-devices 0` → explicit single-device selection
- `--list-gpus` → print visible CUDA GPUs and exit

If `--gpu-api cuda` is selected and `--gpu-devices` is omitted:
- default to current single-device behavior for backward compatibility, **or**
- clearly log the chosen default if implementation opts for `all`

Prefer backward-compatible default behavior unless there is a strong reason to change it.

## Functional Requirements

1. Enumerate visible CUDA devices and expose stable per-device metadata:
   - index
   - name
   - compute capability when available
   - SM / compute unit count

2. Allow creation of one independent CUDA context per selected device.

3. Run one GPU worker/driver per selected device for:
   - pure GPU raw mode
   - pure GPU mnemonic mode
   - hybrid mode

4. Aggregate global progress/results across all active GPUs:
   - shared stop flag
   - shared candidate counter
   - shared match limit handling
   - shared result stream

5. Keep result verification behavior unchanged.

6. If one GPU fails during startup:
   - fail clearly if the user explicitly requested only that GPU
   - otherwise return a clear error describing which requested devices failed

7. If one GPU fails during execution:
   - stop the run with a clear error in the first implementation
   - do not silently continue with partial coverage unless explicitly designed and tested

## Implementation Notes

### Architecture

Refactor the current single `ActiveGpuContext` flow into a selected-device collection for CUDA mode.

Likely shape:
- add CUDA device enumeration helpers
- add selected-device parsing/validation in CLI
- build `Vec<GpuContext>` (or equivalent device wrapper)
- spawn one GPU worker thread per device
- reuse shared atomics/channels for aggregation

### Work Distribution

Use independent per-device batch loops rather than static equal partitioning.

Each GPU should:
- use its own suggested batch size
- increment the shared candidate counter by its own completed batch size
- report matches through the shared result channel

This should naturally let faster GPUs contribute more work.

### Mode-specific guidance

#### Raw GPU mode
- easiest first target
- each GPU can self-feed private key batches and search independently

#### Mnemonic GPU mode
- each GPU can run its own mnemonic driver loop
- CPU mnemonic generation may be shared or per-device, but keep ownership/simple shutdown clear

#### Hybrid mode
- keep current CPU workers
- fan GPU-oriented work to multiple device workers
- ensure CPU+GPU accounting stays honest

## Testing Requirements

Implementation is not done until all of the following are covered.

### Automated

- unit tests for device selection parsing
- tests for duplicate / invalid device index handling
- tests for config defaults and CLI validation

### Runtime validation

Validate at least:
- single GPU CUDA regression still works
- multi-GPU same-model system works
- multi-GPU mixed-model system works
- raw mode works across multiple GPUs
- mnemonic mode works across multiple GPUs
- hybrid mode works across multiple GPUs

### Correctness expectations

- no duplicate or corrupted output due to concurrency bugs
- `max_matches` stops the run correctly across all workers
- candidate counters remain monotonic and believable
- graceful shutdown / Ctrl-C works without hanging workers

## Acceptance Criteria

- A user can list CUDA GPUs from the CLI.
- A user can target one or more CUDA devices explicitly.
- A user can run one search that uses multiple NVIDIA GPUs concurrently.
- Mixed NVIDIA GPUs contribute simultaneously without breaking correctness.
- Existing single-GPU workflows remain working.
- README usage/examples are updated after implementation lands.

## Suggested Delivery Order

1. CLI and config surface (`--list-gpus`, `--gpu-devices`)
2. CUDA device enumeration
3. raw multi-GPU execution
4. mnemonic multi-GPU execution
5. hybrid multi-GPU execution
6. tests, docs, and runtime proof
