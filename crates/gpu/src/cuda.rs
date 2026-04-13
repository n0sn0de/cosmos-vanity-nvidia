//! CUDA GPU acceleration for Cosmos vanity address hashing.
//!
//! This backend reuses the existing OpenCL kernel sources and compiles them for
//! CUDA/NVRTC using a thin compatibility preamble. That keeps the CUDA path
//! behavior-aligned with the existing OpenCL implementation instead of growing
//! a second, drifting kernel codebase.

use std::{
    ffi::{c_char, c_int, CStr, CString},
    fs,
    mem::MaybeUninit,
    process::Command,
    ptr,
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use cudarc::driver::{
    result::DriverError, CudaContext as DriverContext, CudaFunction, CudaModule, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{sys as nvrtc_sys, Ptx};
use libloading::Library;
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
const KNOWN_COMPUTE_ARCHES: &[u32] = &[
    120, 110, 100, 90, 89, 87, 86, 80, 75, 72, 70, 62, 61, 60, 53, 52, 50, 37, 35, 30,
];

type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut nvrtc_sys::nvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> nvrtc_sys::nvrtcResult;
type NvrtcCompileProgram = unsafe extern "C" fn(
    nvrtc_sys::nvrtcProgram,
    c_int,
    *const *const c_char,
) -> nvrtc_sys::nvrtcResult;
type NvrtcDestroyProgram =
    unsafe extern "C" fn(*mut nvrtc_sys::nvrtcProgram) -> nvrtc_sys::nvrtcResult;
type NvrtcGetPtx =
    unsafe extern "C" fn(nvrtc_sys::nvrtcProgram, *mut c_char) -> nvrtc_sys::nvrtcResult;
type NvrtcGetPtxSize =
    unsafe extern "C" fn(nvrtc_sys::nvrtcProgram, *mut usize) -> nvrtc_sys::nvrtcResult;
type NvrtcGetProgramLog =
    unsafe extern "C" fn(nvrtc_sys::nvrtcProgram, *mut c_char) -> nvrtc_sys::nvrtcResult;
type NvrtcGetProgramLogSize =
    unsafe extern "C" fn(nvrtc_sys::nvrtcProgram, *mut usize) -> nvrtc_sys::nvrtcResult;
type NvrtcGetCubin =
    unsafe extern "C" fn(nvrtc_sys::nvrtcProgram, *mut c_char) -> nvrtc_sys::nvrtcResult;
type NvrtcGetCubinSize =
    unsafe extern "C" fn(nvrtc_sys::nvrtcProgram, *mut usize) -> nvrtc_sys::nvrtcResult;
type NvrtcGetErrorString = unsafe extern "C" fn(nvrtc_sys::nvrtcResult) -> *const c_char;

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

fn arch_candidates(capability: Option<(i32, i32)>, prefix: &str) -> Vec<String> {
    let override_var = match prefix {
        "sm" => "COSMOS_VANITY_CUDA_FORCE_SM_ARCH",
        "compute" => "COSMOS_VANITY_CUDA_FORCE_COMPUTE_ARCH",
        _ => "",
    };
    if !override_var.is_empty() {
        if let Ok(forced) = std::env::var(override_var) {
            let forced = forced.trim();
            if !forced.is_empty() {
                let arch = if forced.contains('_') {
                    forced.to_string()
                } else {
                    format!("{prefix}_{forced}")
                };
                return vec![arch];
            }
        }
    }

    let Some((major, minor)) = capability else {
        return Vec::new();
    };

    let target = (major as u32) * 10 + minor as u32;
    let mut arches = vec![target];
    for &candidate in KNOWN_COMPUTE_ARCHES {
        if candidate <= target && !arches.contains(&candidate) {
            arches.push(candidate);
        }
    }

    arches
        .into_iter()
        .map(|arch| format!("{prefix}_{arch:02}"))
        .collect()
}

fn compile_module(
    ctx: &Arc<DriverContext>,
    src: &str,
    name: &str,
) -> Result<Arc<CudaModule>, GpuError> {
    let capability = ctx.compute_capability().ok();
    let compile_started = Instant::now();
    let nvrtc = NvrtcLibrary::load()?;

    if capability.is_some() && nvrtc.supports_cubin() {
        for sm_arch in arch_candidates(capability, "sm") {
            tracing::info!("Compiling CUDA module {name} for {sm_arch} via CUBIN");
            match compile_cubin_module(ctx, &nvrtc, src, name, &sm_arch) {
                Ok(module) => {
                    tracing::info!(
                        "Loaded CUDA module {name} via CUBIN ({sm_arch}) in {:.2}s",
                        compile_started.elapsed().as_secs_f64()
                    );
                    return Ok(module);
                }
                Err(err) => {
                    tracing::warn!(
                        "CUBIN compile/load failed for {name} on {sm_arch}: {err}. Trying next fallback."
                    );
                }
            }
        }
    } else if capability.is_some() {
        tracing::warn!(
            "NVRTC library does not expose CUBIN APIs on this machine, trying external nvcc before PTX fallback"
        );
    }

    if capability.is_some() {
        for sm_arch in arch_candidates(capability, "sm") {
            tracing::info!("Compiling CUDA module {name} for {sm_arch} via external nvcc CUBIN");
            match compile_nvcc_cubin_module(ctx, src, name, &sm_arch) {
                Ok(module) => {
                    tracing::info!(
                        "Loaded CUDA module {name} via external nvcc CUBIN ({sm_arch}) in {:.2}s",
                        compile_started.elapsed().as_secs_f64()
                    );
                    return Ok(module);
                }
                Err(err) => {
                    tracing::warn!(
                        "external nvcc CUBIN compile/load failed for {name} on {sm_arch}: {err}. Trying next fallback."
                    );
                }
            }
        }
    }

    let ptx_arches = arch_candidates(capability, "compute");
    if ptx_arches.is_empty() {
        tracing::info!("Compiling CUDA module {name} via PTX fallback");
        let module = compile_ptx_module(ctx, &nvrtc, src, name, None)?;
        tracing::info!(
            "Loaded CUDA module {name} via PTX in {:.2}s",
            compile_started.elapsed().as_secs_f64()
        );
        return Ok(module);
    }

    let mut last_error = None;
    for compute_arch in ptx_arches {
        tracing::info!("Compiling CUDA module {name} for {compute_arch} via PTX fallback");
        match compile_ptx_module(ctx, &nvrtc, src, name, Some(&compute_arch)) {
            Ok(module) => {
                tracing::info!(
                    "Loaded CUDA module {name} via PTX ({compute_arch}) in {:.2}s",
                    compile_started.elapsed().as_secs_f64()
                );
                return Ok(module);
            }
            Err(err) => {
                tracing::warn!(
                    "PTX compile/load failed for {name} on {compute_arch}: {err}. Trying next fallback."
                );
                last_error = Some(err);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        GpuError::Nvrtc(format!("all CUDA compilation strategies failed for {name}"))
    }))
}

fn compile_cubin_module(
    ctx: &Arc<DriverContext>,
    nvrtc: &NvrtcLibrary,
    src: &str,
    name: &str,
    arch: &str,
) -> Result<Arc<CudaModule>, GpuError> {
    let cubin = nvrtc.compile_cubin(src, name, arch)?;
    Ok(ctx.load_module(Ptx::from_binary(cubin))?)
}

fn compile_nvcc_cubin_module(
    ctx: &Arc<DriverContext>,
    src: &str,
    name: &str,
    arch: &str,
) -> Result<Arc<CudaModule>, GpuError> {
    let cubin = compile_cubin_with_nvcc(src, name, arch)?;
    Ok(ctx.load_module(Ptx::from_binary(cubin))?)
}

fn compile_cubin_with_nvcc(src: &str, name: &str, arch: &str) -> Result<Vec<u8>, GpuError> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    let stem = name.trim_end_matches(".cu").replace(['/', '\\'], "_");
    let temp_dir = std::env::temp_dir().join(format!(
        "cosmos-vanity-nvcc-{stem}-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir_all(&temp_dir).map_err(|err| {
        GpuError::RuntimeUnavailable(format!(
            "failed to create temporary nvcc directory for {name}: {err}"
        ))
    })?;

    let src_path = temp_dir.join(format!("{stem}.cu"));
    let cubin_path = temp_dir.join(format!("{stem}.cubin"));
    fs::write(&src_path, src).map_err(|err| {
        GpuError::RuntimeUnavailable(format!(
            "failed to write temporary CUDA source for {name}: {err}"
        ))
    })?;

    let output = Command::new("nvcc")
        .arg("--std=c++14")
        .arg(format!("--gpu-architecture={arch}"))
        .arg("--cubin")
        .arg(&src_path)
        .arg("-o")
        .arg(&cubin_path)
        .output();

    let output = match output {
        Ok(output) => output,
        Err(err) => {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(GpuError::RuntimeUnavailable(format!(
                "failed to start nvcc for {name} ({arch}): {err}"
            )));
        }
    };

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(GpuError::Nvrtc(format!(
            "external nvcc CUBIN compile failed for {name} ({arch}) with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout,
            stderr,
        )));
    }

    let cubin = fs::read(&cubin_path).map_err(|err| {
        GpuError::RuntimeUnavailable(format!(
            "failed to read nvcc-generated CUBIN for {name} ({arch}): {err}"
        ))
    })?;
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(cubin)
}

fn compile_ptx_module(
    ctx: &Arc<DriverContext>,
    nvrtc: &NvrtcLibrary,
    src: &str,
    name: &str,
    arch: Option<&str>,
) -> Result<Arc<CudaModule>, GpuError> {
    let ptx = nvrtc.compile_ptx(src, name, arch)?;
    Ok(ctx.load_module(Ptx::from_src(ptx))?)
}

fn nvrtc_compile_options(arch: Option<&str>) -> Vec<String> {
    let mut options = vec![
        "--std=c++14".to_string(),
        "--device-as-default-execution-space".to_string(),
    ];
    if let Some(arch) = arch {
        options.push(format!("--gpu-architecture={arch}"));
    }
    options
}

struct NvrtcLibrary {
    _library: Library,
    create_program: NvrtcCreateProgram,
    compile_program: NvrtcCompileProgram,
    destroy_program: NvrtcDestroyProgram,
    get_ptx: NvrtcGetPtx,
    get_ptx_size: NvrtcGetPtxSize,
    get_program_log: NvrtcGetProgramLog,
    get_program_log_size: NvrtcGetProgramLogSize,
    get_cubin: Option<NvrtcGetCubin>,
    get_cubin_size: Option<NvrtcGetCubinSize>,
    get_error_string: NvrtcGetErrorString,
}

impl NvrtcLibrary {
    fn load() -> Result<Self, GpuError> {
        let candidates = cudarc::get_lib_name_candidates("nvrtc");
        let mut failures = Vec::new();

        for candidate in &candidates {
            let library = unsafe { Library::new(candidate) };
            let library = match library {
                Ok(library) => library,
                Err(err) => {
                    failures.push(format!("{candidate}: {err}"));
                    continue;
                }
            };

            match unsafe { Self::from_library(library) } {
                Ok(nvrtc) => return Ok(nvrtc),
                Err(err) => failures.push(format!("{candidate}: {err}")),
            }
        }

        Err(GpuError::RuntimeUnavailable(format!(
            "unable to load NVRTC dynamically from {:?}: {}",
            candidates,
            failures.join("; ")
        )))
    }

    unsafe fn from_library(library: Library) -> Result<Self, String> {
        Ok(Self {
            create_program: unsafe { load_required_symbol(&library, b"nvrtcCreateProgram\0") }?,
            compile_program: unsafe { load_required_symbol(&library, b"nvrtcCompileProgram\0") }?,
            destroy_program: unsafe { load_required_symbol(&library, b"nvrtcDestroyProgram\0") }?,
            get_ptx: unsafe { load_required_symbol(&library, b"nvrtcGetPTX\0") }?,
            get_ptx_size: unsafe { load_required_symbol(&library, b"nvrtcGetPTXSize\0") }?,
            get_program_log: unsafe { load_required_symbol(&library, b"nvrtcGetProgramLog\0") }?,
            get_program_log_size: unsafe {
                load_required_symbol(&library, b"nvrtcGetProgramLogSize\0")
            }?,
            get_cubin: unsafe { load_optional_symbol(&library, b"nvrtcGetCUBIN\0") },
            get_cubin_size: unsafe { load_optional_symbol(&library, b"nvrtcGetCUBINSize\0") },
            get_error_string: unsafe { load_required_symbol(&library, b"nvrtcGetErrorString\0") }?,
            _library: library,
        })
    }

    fn supports_cubin(&self) -> bool {
        self.get_cubin.is_some() && self.get_cubin_size.is_some()
    }

    fn compile_cubin(&self, src: &str, name: &str, arch: &str) -> Result<Vec<u8>, GpuError> {
        let program = NvrtcProgram::create(self, src, name)?;
        program.compile(
            name,
            Some(arch),
            "CUBIN",
            &nvrtc_compile_options(Some(arch)),
        )?;
        program.cubin(name, arch)
    }

    fn compile_ptx(&self, src: &str, name: &str, arch: Option<&str>) -> Result<String, GpuError> {
        let program = NvrtcProgram::create(self, src, name)?;
        program.compile(name, arch, "PTX", &nvrtc_compile_options(arch))?;
        program.ptx(name, arch)
    }

    fn error_string(&self, result: nvrtc_sys::nvrtcResult) -> String {
        let raw = unsafe { (self.get_error_string)(result) };
        if raw.is_null() {
            format!("{result:?}")
        } else {
            unsafe { CStr::from_ptr(raw) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

struct NvrtcProgram<'a> {
    library: &'a NvrtcLibrary,
    handle: nvrtc_sys::nvrtcProgram,
    _src: CString,
    _name: CString,
}

impl<'a> NvrtcProgram<'a> {
    fn create(library: &'a NvrtcLibrary, src: &str, name: &str) -> Result<Self, GpuError> {
        let src_c = CString::new(src).expect("CUDA source must not contain NUL bytes");
        let name_c = CString::new(name).expect("CUDA module name must not contain NUL bytes");
        let mut handle = MaybeUninit::uninit();

        let result = unsafe {
            (library.create_program)(
                handle.as_mut_ptr(),
                src_c.as_ptr(),
                name_c.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };
        if result.result().is_err() {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC create_program failed for {name}: {}",
                library.error_string(result)
            )));
        }

        Ok(Self {
            library,
            handle: unsafe { handle.assume_init() },
            _src: src_c,
            _name: name_c,
        })
    }

    fn compile(
        &self,
        name: &str,
        arch: Option<&str>,
        mode: &str,
        options: &[String],
    ) -> Result<(), GpuError> {
        let c_strings: Vec<CString> = options
            .iter()
            .map(|option| CString::new(option.as_str()).expect("NVRTC option must not contain NUL"))
            .collect();
        let option_ptrs: Vec<*const c_char> = c_strings.iter().map(|opt| opt.as_ptr()).collect();
        let arch_label = arch.unwrap_or("default");

        let result = unsafe {
            (self.library.compile_program)(
                self.handle,
                option_ptrs.len() as c_int,
                option_ptrs.as_ptr(),
            )
        };
        if result.result().is_err() {
            let log = self.program_log();
            let details = if log.is_empty() {
                format!(
                    "NVRTC {mode} compile failed for {name} ({arch_label}): {}",
                    self.library.error_string(result)
                )
            } else {
                format!(
                    "NVRTC {mode} compile failed for {name} ({arch_label}): {}\n{log}",
                    self.library.error_string(result)
                )
            };
            return Err(GpuError::Nvrtc(details));
        }

        Ok(())
    }

    fn ptx(&self, name: &str, arch: Option<&str>) -> Result<String, GpuError> {
        let arch_label = arch.unwrap_or("default");
        let mut size = 0usize;
        let size_result = unsafe { (self.library.get_ptx_size)(self.handle, &mut size as *mut _) };
        if size_result.result().is_err() {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC get PTX size failed for {name} ({arch_label}): {}",
                self.library.error_string(size_result)
            )));
        }

        let mut ptx = vec![0 as c_char; size];
        let ptx_result = unsafe { (self.library.get_ptx)(self.handle, ptx.as_mut_ptr()) };
        if ptx_result.result().is_err() {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC get PTX failed for {name} ({arch_label}): {}",
                self.library.error_string(ptx_result)
            )));
        }

        Ok(unsafe { CStr::from_ptr(ptx.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    fn cubin(&self, name: &str, arch: &str) -> Result<Vec<u8>, GpuError> {
        let Some(get_cubin_size) = self.library.get_cubin_size else {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC library does not expose CUBIN APIs needed for {name} ({arch})"
            )));
        };
        let Some(get_cubin) = self.library.get_cubin else {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC library does not expose CUBIN APIs needed for {name} ({arch})"
            )));
        };

        let mut size = 0usize;
        let size_result = unsafe { get_cubin_size(self.handle, &mut size as *mut _) };
        if size_result.result().is_err() {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC get CUBIN size failed for {name} ({arch}): {}",
                self.library.error_string(size_result)
            )));
        }

        let mut cubin = vec![0u8; size];
        let cubin_result = unsafe { get_cubin(self.handle, cubin.as_mut_ptr().cast()) };
        if cubin_result.result().is_err() {
            return Err(GpuError::Nvrtc(format!(
                "NVRTC get CUBIN failed for {name} ({arch}): {}",
                self.library.error_string(cubin_result)
            )));
        }

        Ok(cubin)
    }

    fn program_log(&self) -> String {
        let mut size = 0usize;
        let size_result =
            unsafe { (self.library.get_program_log_size)(self.handle, &mut size as *mut _) };
        if size_result.result().is_err() || size == 0 {
            return String::new();
        }

        let mut log = vec![0 as c_char; size];
        let log_result = unsafe { (self.library.get_program_log)(self.handle, log.as_mut_ptr()) };
        if log_result.result().is_err() {
            return String::new();
        }

        unsafe { CStr::from_ptr(log.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for NvrtcProgram<'_> {
    fn drop(&mut self) {
        let mut handle = self.handle;
        let _ = unsafe { (self.library.destroy_program)(&mut handle as *mut _) }.result();
    }
}

unsafe fn load_required_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|err| {
            format!(
                "missing symbol {}: {err}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            )
        })
}

unsafe fn load_optional_symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    unsafe { library.get::<T>(name) }.ok().map(|symbol| *symbol)
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

    fn single_thread_launch() -> LaunchConfig {
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn run_cuda_hmac_sha512_test(ctx: &GpuContext, key: &[u8], msg: &[u8]) -> Vec<u8> {
        let module = ctx
            .mnemonic_module()
            .expect("mnemonic diagnostic module should be available for HMAC test");
        let function = module
            .load_function("test_hmac_sha512_kernel")
            .expect("missing HMAC diagnostic kernel");

        let key_buf = ctx.stream.clone_htod(key).expect("htod key");
        let msg_buf = ctx.stream.clone_htod(msg).expect("htod msg");
        let mut output_buf = ctx.stream.alloc_zeros::<u8>(64).expect("alloc output");

        unsafe {
            ctx.stream
                .launch_builder(&function)
                .arg(&key_buf)
                .arg(&(key.len() as u32))
                .arg(&msg_buf)
                .arg(&(msg.len() as u32))
                .arg(&mut output_buf)
                .launch(single_thread_launch())
                .expect("launch HMAC diagnostic kernel");
        }
        ctx.stream.synchronize().expect("sync HMAC diagnostic kernel");
        ctx.stream
            .clone_dtoh(&output_buf)
            .expect("dtoh HMAC diagnostic kernel")
    }

    fn run_cuda_pbkdf2_test(
        ctx: &GpuContext,
        password: &[u8],
        salt: &[u8],
        iterations: u32,
    ) -> Vec<u8> {
        let module = ctx
            .mnemonic_module()
            .expect("mnemonic diagnostic module should be available for PBKDF2 test");
        let function = module
            .load_function("test_pbkdf2_kernel")
            .expect("missing PBKDF2 diagnostic kernel");

        let password_buf = ctx.stream.clone_htod(password).expect("htod password");
        let salt_buf = ctx.stream.clone_htod(salt).expect("htod salt");
        let mut output_buf = ctx.stream.alloc_zeros::<u8>(64).expect("alloc output");

        unsafe {
            ctx.stream
                .launch_builder(&function)
                .arg(&password_buf)
                .arg(&(password.len() as u32))
                .arg(&salt_buf)
                .arg(&(salt.len() as u32))
                .arg(&iterations)
                .arg(&mut output_buf)
                .launch(single_thread_launch())
                .expect("launch PBKDF2 diagnostic kernel");
        }
        ctx.stream.synchronize().expect("sync PBKDF2 diagnostic kernel");
        ctx.stream
            .clone_dtoh(&output_buf)
            .expect("dtoh PBKDF2 diagnostic kernel")
    }

    fn run_cuda_bip32_test(ctx: &GpuContext, seed: &[u8]) -> Vec<u8> {
        let module = ctx
            .mnemonic_module()
            .expect("mnemonic diagnostic module should be available for BIP32 test");
        let function = module
            .load_function("test_bip32_kernel")
            .expect("missing BIP32 diagnostic kernel");

        let seed_buf = ctx.stream.clone_htod(seed).expect("htod seed");
        let mut output_buf = ctx.stream.alloc_zeros::<u8>(32 * 6).expect("alloc output");

        unsafe {
            ctx.stream
                .launch_builder(&function)
                .arg(&seed_buf)
                .arg(&mut output_buf)
                .launch(single_thread_launch())
                .expect("launch BIP32 diagnostic kernel");
        }
        ctx.stream.synchronize().expect("sync BIP32 diagnostic kernel");
        ctx.stream
            .clone_dtoh(&output_buf)
            .expect("dtoh BIP32 diagnostic kernel")
    }

    fn run_cuda_bip32_hardened_debug(ctx: &GpuContext, seed: &[u8]) -> Vec<u8> {
        let module = ctx
            .mnemonic_module()
            .expect("mnemonic diagnostic module should be available for BIP32 debug");
        let function = module
            .load_function("test_bip32_hardened_debug_kernel")
            .expect("missing BIP32 hardened debug kernel");

        let seed_buf = ctx.stream.clone_htod(seed).expect("htod seed");
        let mut output_buf = ctx.stream.alloc_zeros::<u8>(32 * 8).expect("alloc output");

        unsafe {
            ctx.stream
                .launch_builder(&function)
                .arg(&seed_buf)
                .arg(&mut output_buf)
                .launch(single_thread_launch())
                .expect("launch BIP32 hardened debug kernel");
        }
        ctx.stream
            .synchronize()
            .expect("sync BIP32 hardened debug kernel");
        ctx.stream
            .clone_dtoh(&output_buf)
            .expect("dtoh BIP32 hardened debug kernel")
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
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_hmac_sha512_bip32_vectors() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_hmac_sha512_bip32_vectors") else {
            return;
        };

        let master_key = hex::decode(
            "4ce7e11819686786a7a01310c3b5e9c7257d724253525d4502840a7abac0e2c7",
        )
        .unwrap();
        let master_chain = hex::decode(
            "9930a92a22307366bc88347a037a203fe4ce27858c68cff91195c24dcd6aef14",
        )
        .unwrap();
        let child_44_key = hex::decode(
            "b47ebdd2866a6deabadb573fae5da2760fe6dd7ac5b71a8246e77cfdd375a7d1",
        )
        .unwrap();
        let child_44_chain = hex::decode(
            "9972f653ca638e6c053e04530d93d7edaf2ae5745dfc1613877649b3d5dee3dd",
        )
        .unwrap();

        let mut msg_44 = vec![0u8; 37];
        msg_44[1..33].copy_from_slice(&master_key);
        msg_44[33..].copy_from_slice(&0x8000002Cu32.to_be_bytes());
        let hmac_44 = run_cuda_hmac_sha512_test(&ctx, &master_chain, &msg_44);
        assert_eq!(
            hex::encode(&hmac_44[..32]),
            "6796dcba6d020664133b442eeaa7b8aeea696b387264bd3d4463728318b4c50a",
            "cuda HMAC IL mismatch for 44'"
        );
        assert_eq!(
            hex::encode(&hmac_44[32..]),
            "9972f653ca638e6c053e04530d93d7edaf2ae5745dfc1613877649b3d5dee3dd",
            "cuda HMAC IR mismatch for 44'"
        );

        let mut msg_118 = vec![0u8; 37];
        msg_118[1..33].copy_from_slice(&child_44_key);
        msg_118[33..].copy_from_slice(&0x80000076u32.to_be_bytes());
        let hmac_118 = run_cuda_hmac_sha512_test(&ctx, &child_44_chain, &msg_118);
        assert_eq!(
            hex::encode(&hmac_118[..32]),
            "7e8cd78308ebd2bd4a09c1e338bfd93166ee16075c9ba4d59f15be2d90263ca9",
            "cuda HMAC IL mismatch for 118'"
        );
        assert_eq!(
            hex::encode(&hmac_118[32..]),
            "cc6abe5c7245f24c79c93983793531b413b9acc90ae64789b233b83f0eb5ba8a",
            "cuda HMAC IR mismatch for 118'"
        );
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_pbkdf2_diagnostic_kernel() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_pbkdf2_diagnostic_kernel") else {
            return;
        };

        let mnemonic =
            b"monster asthma shaft average main office dial since rural guitar estate sight";
        let salt = b"mnemonic";
        let expected_seed =
            "e2fcb5bec6933a405806d6e8461a562036d584fe55958e13f4705014c936f716db0009b8558ad35cd7d023d043b0c51bbb9a4dc810addd4cfbb4604115224dd8";

        let result = run_cuda_pbkdf2_test(&ctx, mnemonic, salt, 2048);
        assert_eq!(
            hex::encode(result),
            expected_seed,
            "cuda PBKDF2 diagnostic mismatch"
        );
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_bip32_hardened_debug_kernel() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_bip32_hardened_debug_kernel") else {
            return;
        };

        let seed = hex::decode("e2fcb5bec6933a405806d6e8461a562036d584fe55958e13f4705014c936f716db0009b8558ad35cd7d023d043b0c51bbb9a4dc810addd4cfbb4604115224dd8").unwrap();
        let result = run_cuda_bip32_hardened_debug(&ctx, &seed);
        let expected = [
            "4ce7e11819686786a7a01310c3b5e9c7257d724253525d4502840a7abac0e2c7",
            "9930a92a22307366bc88347a037a203fe4ce27858c68cff91195c24dcd6aef14",
            "6796dcba6d020664133b442eeaa7b8aeea696b387264bd3d4463728318b4c50a",
            "9972f653ca638e6c053e04530d93d7edaf2ae5745dfc1613877649b3d5dee3dd",
            "b47ebdd2866a6deabadb573fae5da2760fe6dd7ac5b71a8246e77cfdd375a7d1",
            "7e8cd78308ebd2bd4a09c1e338bfd93166ee16075c9ba4d59f15be2d90263ca9",
            "cc6abe5c7245f24c79c93983793531b413b9acc90ae64789b233b83f0eb5ba8a",
            "330b95558f5640a804e51922e71d7ba8bc26169b730a1f1c262adc9e9365a339",
        ];

        for (i, expected_hex) in expected.iter().enumerate() {
            let got = hex::encode(&result[i * 32..(i + 1) * 32]);
            assert_eq!(got, *expected_hex, "cuda BIP32 hardened debug mismatch at chunk {i}");
        }
    }

    #[test]
    #[ignore = "requires explicit CUDA runtime validation"]
    fn test_cuda_bip32_diagnostic_kernel() {
        let Some(ctx) = runtime_cuda_ctx("test_cuda_bip32_diagnostic_kernel") else {
            return;
        };

        let seed = hex::decode("e2fcb5bec6933a405806d6e8461a562036d584fe55958e13f4705014c936f716db0009b8558ad35cd7d023d043b0c51bbb9a4dc810addd4cfbb4604115224dd8").unwrap();
        let expected_keys = [
            "4ce7e11819686786a7a01310c3b5e9c7257d724253525d4502840a7abac0e2c7",
            "b47ebdd2866a6deabadb573fae5da2760fe6dd7ac5b71a8246e77cfdd375a7d1",
            "330b95558f5640a804e51922e71d7ba8bc26169b730a1f1c262adc9e9365a339",
            "8eb2adf81551e45133ec870200c9b338efb5df52af20d4b40f60fd10695f553b",
            "2ec9f25625f0043b96b13e0436bb597f8e4cceef31b246c9c13717722193597b",
            "9094ec80ab633ffa6888a996ed97fb46e0880526cae8582461655377d20c08c1",
        ];

        let result = run_cuda_bip32_test(&ctx, &seed);
        for (i, expected) in expected_keys.iter().enumerate() {
            let got = hex::encode(&result[i * 32..(i + 1) * 32]);
            assert_eq!(got, *expected, "cuda BIP32 diagnostic mismatch at level {i}");
        }
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
