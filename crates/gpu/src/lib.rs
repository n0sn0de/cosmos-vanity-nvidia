//! # cosmos-vanity-gpu
//!
//! GPU acceleration for Cosmos vanity address generation.
//!
//! Supported backends:
//! - OpenCL (AMD/ROCm, and other OpenCL-capable devices)
//! - CUDA (NVIDIA, via the CUDA driver + NVRTC)
//!
//! When no GPU backend is compiled or available, the crate falls back to
//! CPU-based searching.

pub mod search;
pub mod state;

#[cfg(any(feature = "opencl", feature = "cuda"))]
pub(crate) mod backend;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(not(feature = "opencl"))]
pub mod opencl {
    //! Stub module when OpenCL is not available.

    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum GpuError {
        #[error("OpenCL support not compiled in. Rebuild with --features opencl")]
        NotAvailable,
    }

    /// Check if OpenCL GPU acceleration is available.
    pub fn is_available() -> bool {
        false
    }

    /// Placeholder OpenCL context.
    pub struct GpuContext;

    impl GpuContext {
        pub fn new() -> Result<Self, GpuError> {
            Err(GpuError::NotAvailable)
        }
    }
}

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(not(feature = "cuda"))]
pub mod cuda {
    //! Stub module when CUDA is not available.

    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum GpuError {
        #[error("CUDA support not compiled in. Rebuild with --features cuda")]
        NotAvailable,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CudaDeviceInfo {
        pub index: usize,
        pub name: String,
        pub compute_capability: Option<(u32, u32)>,
        pub max_compute_units: u32,
    }

    /// Check if CUDA GPU acceleration is available.
    pub fn is_available() -> bool {
        false
    }

    pub fn list_devices() -> Result<Vec<CudaDeviceInfo>, GpuError> {
        Err(GpuError::NotAvailable)
    }

    /// Placeholder CUDA context.
    pub struct GpuContext;

    impl GpuContext {
        pub fn new() -> Result<Self, GpuError> {
            Err(GpuError::NotAvailable)
        }

        pub fn new_for_device(_device_index: usize) -> Result<Self, GpuError> {
            Err(GpuError::NotAvailable)
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) mod backend_tests;

pub use search::{
    GpuApi, GpuDeviceSelection, KeyMode, SearchConfig, SearchMode, SearchResult, VanitySearcher,
};
pub use state::SearchState;

/// Returns true if the requested GPU API is available on this machine.
#[cfg(any(feature = "opencl", feature = "cuda"))]
pub fn is_gpu_api_available(api: GpuApi) -> bool {
    backend::ActiveGpuContext::is_available(api)
}

/// Returns false when GPU backends were not compiled in.
#[cfg(not(any(feature = "opencl", feature = "cuda")))]
pub fn is_gpu_api_available(_api: GpuApi) -> bool {
    false
}
