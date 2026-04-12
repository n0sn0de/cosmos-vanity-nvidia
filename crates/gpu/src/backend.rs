use crate::search::GpuApi;

/// Active GPU backend chosen at runtime.
pub(crate) enum ActiveGpuContext {
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::GpuContext),
    #[cfg(feature = "opencl")]
    OpenCl(crate::opencl::GpuContext),
}

impl ActiveGpuContext {
    pub(crate) fn new(preference: GpuApi) -> anyhow::Result<Self> {
        match preference {
            GpuApi::Auto => {
                #[cfg(feature = "cuda")]
                {
                    if crate::cuda::is_available() {
                        match crate::cuda::GpuContext::new() {
                            Ok(ctx) => return Ok(Self::Cuda(ctx)),
                            Err(err) => {
                                tracing::warn!(
                                    "CUDA backend detected but initialization failed: {err}"
                                );
                            }
                        }
                    }
                }

                #[cfg(feature = "opencl")]
                {
                    if crate::opencl::is_available() {
                        return crate::opencl::GpuContext::new()
                            .map(Self::OpenCl)
                            .map_err(|err| anyhow::anyhow!("OpenCL init failed: {err}"));
                    }
                }

                Err(anyhow::anyhow!("No compatible GPU backend available"))
            }
            GpuApi::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if !crate::cuda::is_available() {
                        return Err(anyhow::anyhow!(
                            "CUDA backend requested, but no CUDA device is available"
                        ));
                    }
                    crate::cuda::GpuContext::new()
                        .map(Self::Cuda)
                        .map_err(|err| anyhow::anyhow!("CUDA init failed: {err}"))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(anyhow::anyhow!(
                        "CUDA backend requested, but support was not compiled in"
                    ))
                }
            }
            GpuApi::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    if !crate::opencl::is_available() {
                        return Err(anyhow::anyhow!(
                            "OpenCL backend requested, but no OpenCL GPU is available"
                        ));
                    }
                    crate::opencl::GpuContext::new()
                        .map(Self::OpenCl)
                        .map_err(|err| anyhow::anyhow!("OpenCL init failed: {err}"))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(anyhow::anyhow!(
                        "OpenCL backend requested, but support was not compiled in"
                    ))
                }
            }
        }
    }

    pub(crate) fn is_available(preference: GpuApi) -> bool {
        match preference {
            GpuApi::Auto => {
                #[cfg(feature = "cuda")]
                if crate::cuda::is_available() {
                    return true;
                }

                #[cfg(feature = "opencl")]
                if crate::opencl::is_available() {
                    return true;
                }

                false
            }
            GpuApi::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::is_available()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            }
            GpuApi::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    crate::opencl::is_available()
                }
                #[cfg(not(feature = "opencl"))]
                {
                    false
                }
            }
        }
    }

    pub(crate) fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => "opencl",
        }
    }

    pub(crate) fn device_name(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.device_name(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.device_name(),
        }
    }

    pub(crate) fn max_compute_units(&self) -> u32 {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.max_compute_units(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.max_compute_units(),
        }
    }

    pub(crate) fn suggested_batch_size(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.suggested_batch_size(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.suggested_batch_size(),
        }
    }

    pub(crate) fn pure_gpu_batch_size(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.pure_gpu_batch_size(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.pure_gpu_batch_size(),
        }
    }

    pub(crate) fn mnemonic_batch_size(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.mnemonic_batch_size(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.mnemonic_batch_size(),
        }
    }

    pub(crate) fn has_secp256k1_kernel(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.has_secp256k1_kernel(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.has_secp256k1_kernel(),
        }
    }

    pub(crate) fn has_mnemonic_kernel(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.has_mnemonic_kernel(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx.has_mnemonic_kernel(),
        }
    }

    pub(crate) fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> anyhow::Result<Vec<u8>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx
                .hash_pubkeys_batch(pubkeys)
                .map_err(|err| anyhow::anyhow!("CUDA hash batch failed: {err}")),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx
                .hash_pubkeys_batch(pubkeys)
                .map_err(|err| anyhow::anyhow!("OpenCL hash batch failed: {err}")),
        }
    }

    pub(crate) fn generate_and_hash_batch(
        &self,
        privkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx
                .generate_and_hash_batch(privkeys, prefix_bytes)
                .map_err(|err| anyhow::anyhow!("CUDA secp256k1 batch failed: {err}")),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx
                .generate_and_hash_batch(privkeys, prefix_bytes)
                .map_err(|err| anyhow::anyhow!("OpenCL secp256k1 batch failed: {err}")),
        }
    }

    pub(crate) fn mnemonic_batch(
        &self,
        mnemonics_flat: &[u8],
        mnemonic_lens: &[u32],
    ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx
                .mnemonic_batch(mnemonics_flat, mnemonic_lens)
                .map_err(|err| anyhow::anyhow!("CUDA mnemonic batch failed: {err}")),
            #[cfg(feature = "opencl")]
            Self::OpenCl(ctx) => ctx
                .mnemonic_batch(mnemonics_flat, mnemonic_lens)
                .map_err(|err| anyhow::anyhow!("OpenCL mnemonic batch failed: {err}")),
        }
    }
}
