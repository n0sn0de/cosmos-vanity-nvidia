//! CUDA GPU acceleration for Cosmos vanity address hashing.
//!
//! This backend reuses the existing OpenCL kernel sources and compiles them for
//! CUDA/NVRTC using a thin compatibility preamble. That keeps the CUDA path
//! behavior-aligned with the existing OpenCL implementation instead of growing
//! a second, drifting kernel codebase.

use std::{
    ffi::{CStr, CString},
    sync::{Arc, Mutex},
    time::Instant,
};

use cudarc::driver::{
    result::DriverError, CudaContext as DriverContext, CudaFunction, CudaModule, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{
    compile_ptx_with_opts, result as nvrtc_result, sys as nvrtc_sys, CompileError, CompileOptions,
    Ptx,
};
use thiserror::Error;
use tracing::{debug, info};

/// Size of a compressed secp256k1 public key.
const PUBKEY_SIZE: usize = 33;
/// Size of a RIPEMD-160 hash (Cosmos address hash).
const HASH_SIZE: usize = 20;
/// Size of a raw private key.
const PRIVKEY_SIZE: usize = 32;

type GpuBatchResult = (Vec<u8>, Vec<u8>, Vec<u32>);

const CUDA_COMPAT_PREAMBLE: &str = r#"
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long long ulong;

#define __kernel extern "C" __global__
#define __global
#define __constant __constant__
#define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
"#;

const HASH_KERNEL_SOURCE: &str = include_str!("kernels/vanity_search.cl");
const SECP256K1_KERNEL_SOURCE: &str = include_str!("kernels/secp256k1.cl");
const MNEMONIC_KERNEL_SOURCE: &str = include_str!("kernels/mnemonic_pipeline.cl");

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No CUDA device found")]
    NoDevice,
    #[error("CUDA runtime unavailable: {0}")]
    RuntimeUnavailable(String),
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("CUDA compilation error: {0}")]
    Nvrtc(String),
    #[error("GPU batch size must be > 0")]
    InvalidBatchSize,
}

impl From<CompileError> for GpuError {
    fn from(value: CompileError) -> Self {
        Self::Nvrtc(value.to_string())
    }
}

/// Check if CUDA acceleration is available.
pub fn is_available() -> bool {
    std::panic::catch_unwind(|| match DriverContext::device_count() {
        Ok(count) if count > 0 => DriverContext::new(0).is_ok(),
        _ => false,
    })
    .unwrap_or(false)
}

/// CUDA context holding the stream, compiled modules, and device info.
pub struct GpuContext {
    ctx: Arc<DriverContext>,
    stream: Arc<CudaStream>,
    hash_function: CudaFunction,
    secp256k1_function: Mutex<Option<CudaFunction>>,
    #[cfg(test)]
    mnemonic_module: Mutex<Option<Arc<CudaModule>>>,
    mnemonic_function: Mutex<Option<CudaFunction>>,
    device_name: String,
    max_threads_per_block: u32,
    max_compute_units: u32,
}

impl GpuContext {
    /// Initialize CUDA context — finds an NVIDIA GPU and compiles the kernels.
    pub fn new() -> Result<Self, GpuError> {
        let device_count =
            std::panic::catch_unwind(DriverContext::device_count).map_err(|_| {
                GpuError::RuntimeUnavailable(
                    "CUDA driver library could not be loaded on this machine".to_string(),
                )
            })??;
        if device_count <= 0 {
            return Err(GpuError::NoDevice);
        }

        let ctx = std::panic::catch_unwind(|| DriverContext::new(0)).map_err(|_| {
            GpuError::RuntimeUnavailable(
                "CUDA context creation panicked while loading the driver".to_string(),
            )
        })??;
        let stream = ctx.default_stream();

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "Unknown NVIDIA GPU".to_string());
        let max_compute_units = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?
            .max(1) as u32;

        let hash_module = compile_module(
            &ctx,
            &cuda_source(&[HASH_KERNEL_SOURCE]),
            "vanity_search.cu",
        )?;
        let hash_function = hash_module.load_function("compute_address_hashes")?;
        let max_threads_per_block = hash_function.max_threads_per_block()?.max(1) as u32;
        info!(
            "CUDA device: {} (SMs: {}, max threads/block: {})",
            device_name, max_compute_units, max_threads_per_block
        );
        info!("CUDA hash kernel compiled successfully");

        Ok(Self {
            ctx,
            stream,
            hash_function,
            secp256k1_function: Mutex::new(None),
            #[cfg(test)]
            mnemonic_module: Mutex::new(None),
            mnemonic_function: Mutex::new(None),
            device_name,
            max_threads_per_block,
            max_compute_units,
        })
    }

    /// Device name string.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Max compute units / SM count.
    pub fn max_compute_units(&self) -> u32 {
        self.max_compute_units
    }

    /// Check if the secp256k1 kernel is available.
    pub fn has_secp256k1_kernel(&self) -> bool {
        self.ensure_secp256k1_function().is_ok()
    }

    /// Check if the mnemonic pipeline kernel is available.
    pub fn has_mnemonic_kernel(&self) -> bool {
        self.ensure_mnemonic_function().is_ok()
    }

    /// Compute SHA-256 → RIPEMD-160 hashes for a batch of compressed public keys on CUDA.
    pub fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> Result<Vec<u8>, GpuError> {
        let n = pubkeys.len() / PUBKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("CUDA hashing batch of {} pubkeys", n);

        let pubkey_buf = self.stream.clone_htod(pubkeys)?;
        let mut hash_buf = self.stream.alloc_zeros::<u8>(n * HASH_SIZE)?;
        let prefix_buf = self.stream.alloc_zeros::<u8>(1)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;

        let prefix_len = 0u32;
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(&self.hash_function)
                .arg(&pubkey_buf)
                .arg(&mut hash_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        let hashes = self.stream.clone_dtoh(&hash_buf)?;
        debug!("CUDA batch complete: {} hashes computed", n);
        Ok(hashes)
    }

    /// Compute hashes and check for prefix matches on CUDA.
    pub fn hash_and_match_batch(
        &self,
        pubkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u32>), GpuError> {
        let n = pubkeys.len() / PUBKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }

        let pubkey_buf = self.stream.clone_htod(pubkeys)?;
        let mut hash_buf = self.stream.alloc_zeros::<u8>(n * HASH_SIZE)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;
        let prefix_len = prefix_bytes.len() as u32;
        let prefix_buf = if prefix_bytes.is_empty() {
            self.stream.alloc_zeros::<u8>(1)?
        } else {
            self.stream.clone_htod(prefix_bytes)?
        };
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(&self.hash_function)
                .arg(&pubkey_buf)
                .arg(&mut hash_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        Ok((
            self.stream.clone_dtoh(&hash_buf)?,
            self.stream.clone_dtoh(&matches_buf)?,
        ))
    }

    /// Generate public keys and address hashes from raw private keys entirely on CUDA.
    pub fn generate_and_hash_batch(
        &self,
        privkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> Result<GpuBatchResult, GpuError> {
        let function = self.ensure_secp256k1_function()?;

        let n = privkeys.len() / PRIVKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("CUDA secp256k1 batch: {} private keys", n);

        let privkey_buf = self.stream.clone_htod(privkeys)?;
        let mut pubkey_buf = self.stream.alloc_zeros::<u8>(n * PUBKEY_SIZE)?;
        let mut hash_buf = self.stream.alloc_zeros::<u8>(n * HASH_SIZE)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;
        let prefix_len = prefix_bytes.len() as u32;
        let prefix_buf = if prefix_bytes.is_empty() {
            self.stream.alloc_zeros::<u8>(1)?
        } else {
            self.stream.clone_htod(prefix_bytes)?
        };
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&privkey_buf)
                .arg(&mut pubkey_buf)
                .arg(&mut hash_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        Ok((
            self.stream.clone_dtoh(&pubkey_buf)?,
            self.stream.clone_dtoh(&hash_buf)?,
            self.stream.clone_dtoh(&matches_buf)?,
        ))
    }

    /// Suggested batch size for hybrid mode.
    pub fn suggested_batch_size(&self) -> usize {
        let warps_per_sm = 16;
        let warp_size = 32;
        let base = self.max_compute_units as usize * warps_per_sm * warp_size;
        base.max(32_768).next_power_of_two()
    }

    /// Batch size for pure GPU mode.
    pub fn pure_gpu_batch_size(&self) -> usize {
        let warps_per_sm = 32;
        let warp_size = 32;
        let base = self.max_compute_units as usize * warps_per_sm * warp_size;
        base.clamp(65_536, 131_072).next_power_of_two()
    }

    /// Batch size for mnemonic GPU mode.
    pub fn mnemonic_batch_size(&self) -> usize {
        let warps_per_sm = 4;
        let warp_size = 32;
        let base = self.max_compute_units as usize * warps_per_sm * warp_size;
        base.clamp(2_048, 8_192)
    }

    /// Run the full mnemonic pipeline on CUDA.
    pub fn mnemonic_batch(
        &self,
        mnemonics_flat: &[u8],
        mnemonic_lens: &[u32],
    ) -> Result<GpuBatchResult, GpuError> {
        let function = self.ensure_mnemonic_function()?;

        let n = mnemonic_lens.len();
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("CUDA mnemonic batch: {} candidates", n);

        let mnemonics_buf = self.stream.clone_htod(mnemonics_flat)?;
        let lens_buf = self.stream.clone_htod(mnemonic_lens)?;
        let mut privkeys_buf = self.stream.alloc_zeros::<u8>(n * 32)?;
        let mut hashes_buf = self.stream.alloc_zeros::<u8>(n * 20)?;
        let prefix_buf = self.stream.alloc_zeros::<u8>(1)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;
        let prefix_len = 0u32;
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&mnemonics_buf)
                .arg(&lens_buf)
                .arg(&mut privkeys_buf)
                .arg(&mut hashes_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        Ok((
            self.stream.clone_dtoh(&privkeys_buf)?,
            self.stream.clone_dtoh(&hashes_buf)?,
            self.stream.clone_dtoh(&matches_buf)?,
        ))
    }

    fn launch_config(&self, count: u32) -> LaunchConfig {
        let block = self.max_threads_per_block.clamp(1, 256);
        let grid = count.div_ceil(block);
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    #[cfg(test)]
    fn mnemonic_module(&self) -> Option<Arc<CudaModule>> {
        if self.ensure_mnemonic_function().is_err() {
            return None;
        }

        self.mnemonic_module.lock().ok()?.clone()
    }

    fn ensure_secp256k1_function(&self) -> Result<CudaFunction, GpuError> {
        if let Some(function) = self
            .secp256k1_function
            .lock()
            .map_err(|_| GpuError::RuntimeUnavailable("CUDA secp256k1 mutex poisoned".into()))?
            .clone()
        {
            return Ok(function);
        }

        let module = compile_module(
            &self.ctx,
            &cuda_source(&[SECP256K1_KERNEL_SOURCE]),
            "secp256k1.cu",
        )?;
        let function = module.load_function("generate_addresses")?;
        info!("CUDA secp256k1 kernel compiled successfully");

        let mut slot = self
            .secp256k1_function
            .lock()
            .map_err(|_| GpuError::RuntimeUnavailable("CUDA secp256k1 mutex poisoned".into()))?;
        if let Some(existing) = slot.clone() {
            return Ok(existing);
        }
        *slot = Some(function.clone());
        Ok(function)
    }

    fn ensure_mnemonic_function(&self) -> Result<CudaFunction, GpuError> {
        if let Some(function) = self
            .mnemonic_function
            .lock()
            .map_err(|_| GpuError::RuntimeUnavailable("CUDA mnemonic mutex poisoned".into()))?
            .clone()
        {
            return Ok(function);
        }

        let module = compile_module(
            &self.ctx,
            &cuda_source(&[SECP256K1_KERNEL_SOURCE, MNEMONIC_KERNEL_SOURCE]),
            "mnemonic_pipeline.cu",
        )?;
        let function = module.load_function("mnemonic_to_address")?;
        info!("CUDA mnemonic pipeline kernel compiled successfully");

        #[cfg(test)]
        {
            let mut module_slot = self.mnemonic_module.lock().map_err(|_| {
                GpuError::RuntimeUnavailable("CUDA mnemonic module mutex poisoned".into())
            })?;
            if module_slot.is_none() {
                *module_slot = Some(module.clone());
            }
        }

        let mut function_slot = self
            .mnemonic_function
            .lock()
            .map_err(|_| GpuError::RuntimeUnavailable("CUDA mnemonic mutex poisoned".into()))?;
        if let Some(existing) = function_slot.clone() {
            return Ok(existing);
        }
        *function_slot = Some(function.clone());
        Ok(function)
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device_name)
            .field("compute_units", &self.max_compute_units)
            .field("max_threads_per_block", &self.max_threads_per_block)
            .finish()
    }
}

fn cuda_source(parts: &[&str]) -> String {
    let mut source = String::from(CUDA_COMPAT_PREAMBLE);
    for part in parts {
        source.push('\n');
        source.push_str(part);
        source.push('\n');
    }
    source
}

fn compile_module(
    ctx: &Arc<DriverContext>,
    src: &str,
    name: &str,
) -> Result<Arc<CudaModule>, GpuError> {
    let capability = ctx.compute_capability().ok();
    let compile_started = Instant::now();

    if let Some((major, minor)) = capability {
        let sm_arch = format!("sm_{major}{minor}");
        tracing::info!("Compiling CUDA module {name} for {sm_arch} via CUBIN");

        match compile_cubin_module(ctx, src, name, &sm_arch) {
            Ok(module) => {
                tracing::info!(
                    "Loaded CUDA module {name} via CUBIN in {:.2}s",
                    compile_started.elapsed().as_secs_f64()
                );
                return Ok(module);
            }
            Err(err) => {
                tracing::warn!(
                    "CUBIN compile/load failed for {name} on {sm_arch}: {err}. Falling back to PTX."
                );
            }
        }
    }

    let arch = capability.map(|(major, minor)| {
        Box::leak(format!("compute_{major}{minor}").into_boxed_str()) as &'static str
    });

    tracing::info!("Compiling CUDA module {name} via PTX fallback");
    let ptx = compile_ptx_with_opts(
        src,
        CompileOptions {
            arch,
            name: Some(name.to_string()),
            options: vec![
                "--std=c++14".to_string(),
                "--device-as-default-execution-space".to_string(),
            ],
            ..Default::default()
        },
    )?;

    let module = ctx.load_module(ptx)?;
    tracing::info!(
        "Loaded CUDA module {name} via PTX in {:.2}s",
        compile_started.elapsed().as_secs_f64()
    );
    Ok(module)
}

fn compile_cubin_module(
    ctx: &Arc<DriverContext>,
    src: &str,
    name: &str,
    arch: &str,
) -> Result<Arc<CudaModule>, GpuError> {
    let src_c = CString::new(src).expect("CUDA source must not contain NUL bytes");
    let name_c = CString::new(name).expect("CUDA module name must not contain NUL bytes");
    let program = nvrtc_result::create_program(src_c.as_c_str(), Some(name_c.as_c_str()))
        .map_err(|err| GpuError::Nvrtc(format!("NVRTC create_program failed for {name}: {err}")))?;

    let options = vec![
        "--std=c++14".to_string(),
        "--device-as-default-execution-space".to_string(),
        format!("--gpu-architecture={arch}"),
    ];

    let compile_result = unsafe { nvrtc_result::compile_program(program, &options) };
    if let Err(err) = compile_result {
        let log = nvrtc_program_log(program);
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        let details = if log.is_empty() {
            format!("NVRTC CUBIN compile failed for {name} ({arch}): {err}")
        } else {
            format!("NVRTC CUBIN compile failed for {name} ({arch}): {err}\n{log}")
        };
        return Err(GpuError::Nvrtc(details));
    }

    let cubin = unsafe { nvrtc_get_cubin(program) }.map_err(|err| {
        let log = nvrtc_program_log(program);
        if log.is_empty() {
            GpuError::Nvrtc(format!("NVRTC get CUBIN failed for {name} ({arch}): {err}"))
        } else {
            GpuError::Nvrtc(format!(
                "NVRTC get CUBIN failed for {name} ({arch}): {err}\n{log}"
            ))
        }
    })?;

    let destroy_result = unsafe { nvrtc_result::destroy_program(program) };
    if let Err(err) = destroy_result {
        return Err(GpuError::Nvrtc(format!(
            "NVRTC destroy_program failed for {name}: {err}"
        )));
    }

    Ok(ctx.load_module(Ptx::from_binary(cubin))?)
}

fn nvrtc_program_log(program: nvrtc_sys::nvrtcProgram) -> String {
    unsafe { nvrtc_result::get_program_log(program) }
        .ok()
        .map(|raw| {
            unsafe { CStr::from_ptr(raw.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_default()
}

unsafe fn nvrtc_get_cubin(
    program: nvrtc_sys::nvrtcProgram,
) -> Result<Vec<u8>, nvrtc_result::NvrtcError> {
    let mut size = 0usize;
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut size as *mut _) }.result()?;

    let mut cubin = vec![0u8; size];
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) }.result()?;
    Ok(cubin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_tests::{self, BackendHarness};

    fn runtime_cuda_ctx(test_name: &str) -> Option<GpuContext> {
        if !is_available() {
            eprintln!("CUDA runtime validation skipped for {test_name}: no CUDA device available");
            return None;
        }

        Some(
            GpuContext::new().unwrap_or_else(|err| {
                panic!(
                    "CUDA device was detected for {test_name}, but backend initialization failed: {err}"
                )
            }),
        )
    }

    impl BackendHarness for GpuContext {
        fn label(&self) -> &'static str {
            "cuda"
        }

        fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> anyhow::Result<Vec<u8>> {
            Ok(GpuContext::hash_pubkeys_batch(self, pubkeys)?)
        }

        fn has_secp256k1_kernel(&self) -> bool {
            GpuContext::has_secp256k1_kernel(self)
        }

        fn generate_and_hash_batch(
            &self,
            privkeys: &[u8],
            prefix_bytes: &[u8],
        ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
            Ok(GpuContext::generate_and_hash_batch(
                self,
                privkeys,
                prefix_bytes,
            )?)
        }

        fn has_mnemonic_kernel(&self) -> bool {
            GpuContext::has_mnemonic_kernel(self)
        }

        fn mnemonic_batch(
            &self,
            mnemonics_flat: &[u8],
            mnemonic_lens: &[u32],
        ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
            Ok(GpuContext::mnemonic_batch(
                self,
                mnemonics_flat,
                mnemonic_lens,
            )?)
        }
    }

    #[test]
    fn test_is_available() {
        let _ = is_available();
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_hash_matches_cpu() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_hash_matches_cpu") else {
            return;
        };

        backend_tests::assert_hash_matches_cpu(&ctx);
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_secp256k1_known_vector() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_secp256k1_known_vector") else {
            return;
        };

        backend_tests::assert_secp256k1_known_vector(&ctx);
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_secp256k1_matches_cpu() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_secp256k1_matches_cpu") else {
            return;
        };

        backend_tests::assert_secp256k1_matches_cpu(&ctx);
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_mnemonic_pipeline() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_mnemonic_pipeline") else {
            return;
        };

        backend_tests::assert_mnemonic_pipeline(&ctx);
    }

    #[test]
    fn test_cuda_kernel_source_contains_expected_entrypoints() {
        let hash_src = cuda_source(&[HASH_KERNEL_SOURCE]);
        assert!(hash_src.contains("compute_address_hashes"));

        let secp_src = cuda_source(&[SECP256K1_KERNEL_SOURCE]);
        assert!(secp_src.contains("generate_addresses"));

        let mnemonic_src = cuda_source(&[SECP256K1_KERNEL_SOURCE, MNEMONIC_KERNEL_SOURCE]);
        assert!(mnemonic_src.contains("mnemonic_to_address"));
        assert!(mnemonic_src.contains("test_sha512_kernel"));
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_mnemonic_module_keeps_diagnostic_kernels() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_mnemonic_module_keeps_diagnostic_kernels")
        else {
            return;
        };

        let module = ctx.mnemonic_module().expect(
            "mnemonic diagnostic module should be available during CUDA runtime validation",
        );
        module
            .load_function("test_sha512_kernel")
            .expect("missing diagnostic kernel");
        module
            .load_function("test_hmac_sha512_kernel")
            .expect("missing diagnostic kernel");
        module
            .load_function("test_pbkdf2_kernel")
            .expect("missing diagnostic kernel");
        module
            .load_function("test_bip32_kernel")
            .expect("missing diagnostic kernel");
    }
}
