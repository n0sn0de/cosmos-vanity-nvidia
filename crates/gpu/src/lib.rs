//! # cosmos-vanity-gpu
//!
//! OpenCL/ROCm GPU acceleration for Cosmos vanity address generation.
//!
//! When compiled with the `opencl` feature, this crate uses AMD GPUs via ROCm
//! to accelerate the SHA-256 + RIPEMD-160 hashing pipeline. Falls back to
//! CPU-based searching when GPU is unavailable.
//!
//! ## Architecture
//!
//! The GPU handles the bulk hashing work:
//! 1. CPU generates random mnemonics and derives secp256k1 public keys
//! 2. Public keys are batched and sent to GPU
//! 3. GPU computes SHA-256(pubkey) → RIPEMD-160 for all keys in parallel
//! 4. GPU checks for prefix matches on raw address bytes
//! 5. Matches are returned to CPU for full Bech32 verification

pub mod search;
pub mod state;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(not(feature = "opencl"))]
pub mod opencl {
    //! Stub module when OpenCL is not available.
    //!
    //! All GPU functions return errors indicating GPU is unavailable.

    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum GpuError {
        #[error("OpenCL support not compiled in. Rebuild with --features opencl")]
        NotAvailable,
    }

    /// Check if GPU acceleration is available.
    pub fn is_available() -> bool {
        false
    }

    /// Placeholder GPU context.
    pub struct GpuContext;

    impl GpuContext {
        pub fn new() -> Result<Self, GpuError> {
            Err(GpuError::NotAvailable)
        }
    }
}

pub use search::{SearchConfig, SearchMode, SearchResult, VanitySearcher};
pub use state::SearchState;
