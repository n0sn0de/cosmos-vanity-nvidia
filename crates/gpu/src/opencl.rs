//! OpenCL GPU acceleration for Cosmos vanity address hashing.
//!
//! This module manages the GPU context, compiles the OpenCL kernel,
//! and provides batch hashing of compressed public keys on AMD GPUs via ROCm.

use ocl::enums::{DeviceInfo, DeviceInfoResult};
use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
use thiserror::Error;
use tracing::{debug, info};

/// Size of a compressed secp256k1 public key.
const PUBKEY_SIZE: usize = 33;
/// Size of a RIPEMD-160 hash (Cosmos address hash).
const HASH_SIZE: usize = 20;
/// Size of a raw private key.
const PRIVKEY_SIZE: usize = 32;

/// OpenCL kernel source (embedded at compile time).
const KERNEL_SOURCE: &str = include_str!("kernels/vanity_search.cl");

/// OpenCL kernel source for secp256k1 EC operations (embedded at compile time).
const SECP256K1_KERNEL_SOURCE: &str = include_str!("kernels/secp256k1.cl");

/// OpenCL kernel source for full mnemonic pipeline (embedded at compile time).
/// Concatenated with secp256k1.cl since it depends on EC + hash functions.
const MNEMONIC_KERNEL_SOURCE: &str = include_str!("kernels/mnemonic_pipeline.cl");

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No OpenCL platform found")]
    NoPlatform,
    #[error("No GPU device found on platform")]
    NoDevice,
    #[error("OpenCL error: {0}")]
    Ocl(#[from] ocl::Error),
    #[error("GPU batch size must be > 0")]
    InvalidBatchSize,
}

/// Check if GPU acceleration is available.
pub fn is_available() -> bool {
    match Platform::list() {
        platforms if platforms.is_empty() => false,
        platforms => platforms.iter().any(|p| {
            Device::list(p, Some(ocl::flags::DeviceType::GPU))
                .map(|devs| !devs.is_empty())
                .unwrap_or(false)
        }),
    }
}

/// GPU context holding the OpenCL queue, compiled programs, and device info.
pub struct GpuContext {
    queue: Queue,
    program: Program,
    secp256k1_program: Option<Program>,
    mnemonic_program: Option<Program>,
    device_name: String,
    max_work_group_size: usize,
    max_compute_units: u32,
}

impl GpuContext {
    /// Initialize GPU context — finds an AMD GPU, compiles the kernel.
    pub fn new() -> Result<Self, GpuError> {
        let platform = Platform::list()
            .into_iter()
            .find(|p| {
                Device::list(p, Some(ocl::flags::DeviceType::GPU))
                    .map(|devs| !devs.is_empty())
                    .unwrap_or(false)
            })
            .ok_or(GpuError::NoPlatform)?;

        let platform_name = platform
            .name()
            .unwrap_or_else(|_| "Unknown".to_string());
        info!("OpenCL platform: {}", platform_name);

        let device = Device::list(&platform, Some(ocl::flags::DeviceType::GPU))?
            .into_iter()
            .next()
            .ok_or(GpuError::NoDevice)?;

        let device_name = device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
        let max_work_group_size = device.max_wg_size()?;
        let max_compute_units = match device.info(DeviceInfo::MaxComputeUnits)? {
            DeviceInfoResult::MaxComputeUnits(n) => n,
            _ => 1,
        };

        info!(
            "GPU device: {} (CUs: {}, max workgroup: {})",
            device_name, max_compute_units, max_work_group_size
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let queue = Queue::new(&context, device, None)?;

        let program = Program::builder()
            .src(KERNEL_SOURCE)
            .devices(device)
            .build(&context)?;

        info!("OpenCL hash kernel compiled successfully");

        // Compile secp256k1 kernel
        let secp256k1_program = match Program::builder()
            .src(SECP256K1_KERNEL_SOURCE)
            .devices(device)
            .build(&context)
        {
            Ok(prog) => {
                info!("OpenCL secp256k1 kernel compiled successfully");
                Some(prog)
            }
            Err(e) => {
                tracing::warn!("Failed to compile secp256k1 kernel: {e}. Raw key mode will be unavailable.");
                None
            }
        };

        // Compile mnemonic pipeline kernel (secp256k1.cl + mnemonic_pipeline.cl concatenated)
        let mnemonic_program = match Program::builder()
            .src(format!("{}\n{}", SECP256K1_KERNEL_SOURCE, MNEMONIC_KERNEL_SOURCE))
            .devices(device)
            .build(&context)
        {
            Ok(prog) => {
                info!("OpenCL mnemonic pipeline kernel compiled successfully");
                Some(prog)
            }
            Err(e) => {
                tracing::warn!("Failed to compile mnemonic pipeline kernel: {e}. GPU mnemonic mode will be unavailable.");
                None
            }
        };

        Ok(Self {
            queue,
            program,
            secp256k1_program,
            mnemonic_program,
            device_name,
            max_work_group_size,
            max_compute_units,
        })
    }

    /// Device name string.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Max compute units.
    pub fn max_compute_units(&self) -> u32 {
        self.max_compute_units
    }

    /// Compute SHA-256 → RIPEMD-160 hashes for a batch of compressed public keys on GPU.
    ///
    /// `pubkeys` — flat array of `n * 33` bytes (compressed secp256k1 keys).
    /// Returns `n * 20` bytes of address hashes.
    pub fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> Result<Vec<u8>, GpuError> {
        let n = pubkeys.len() / PUBKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("GPU hashing batch of {} pubkeys", n);

        // Create buffers
        let pubkey_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(pubkeys.len())
            .copy_host_slice(pubkeys)
            .build()?;

        let hash_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(n * HASH_SIZE)
            .build()?;

        let matches_buf = Buffer::<u32>::builder()
            .queue(self.queue.clone())
            .len(n)
            .fill_val(0u32)
            .build()?;

        // Empty prefix — we just want the hashes, matching is done on CPU with bech32
        let prefix_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(1)
            .fill_val(0u8)
            .build()?;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name("compute_address_hashes")
            .queue(self.queue.clone())
            .global_work_size(n)
            .arg(&pubkey_buf)
            .arg(&hash_buf)
            .arg(&prefix_buf)
            .arg(0u32) // prefix_len = 0, no GPU-side matching
            .arg(&matches_buf)
            .arg(n as u32)
            .build()?;

        // Execute
        unsafe { kernel.enq()?; }
        self.queue.finish()?;

        // Read results
        let mut hashes = vec![0u8; n * HASH_SIZE];
        hash_buf.read(&mut hashes).enq()?;

        debug!("GPU batch complete: {} hashes computed", n);
        Ok(hashes)
    }

    /// Compute hashes and check for prefix matches on GPU.
    ///
    /// Returns indices of matching candidates.
    pub fn hash_and_match_batch(
        &self,
        pubkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u32>), GpuError> {
        let n = pubkeys.len() / PUBKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }

        let pubkey_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(pubkeys.len())
            .copy_host_slice(pubkeys)
            .build()?;

        let hash_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(n * HASH_SIZE)
            .build()?;

        let matches_buf = Buffer::<u32>::builder()
            .queue(self.queue.clone())
            .len(n)
            .fill_val(0u32)
            .build()?;

        let prefix_len = prefix_bytes.len();
        let prefix_buf = if prefix_len > 0 {
            Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .len(prefix_len)
                .copy_host_slice(prefix_bytes)
                .build()?
        } else {
            Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .len(1)
                .fill_val(0u8)
                .build()?
        };

        let kernel = Kernel::builder()
            .program(&self.program)
            .name("compute_address_hashes")
            .queue(self.queue.clone())
            .global_work_size(n)
            .arg(&pubkey_buf)
            .arg(&hash_buf)
            .arg(&prefix_buf)
            .arg(prefix_len as u32)
            .arg(&matches_buf)
            .arg(n as u32)
            .build()?;

        unsafe { kernel.enq()?; }
        self.queue.finish()?;

        let mut hashes = vec![0u8; n * HASH_SIZE];
        hash_buf.read(&mut hashes).enq()?;

        let mut matches = vec![0u32; n];
        matches_buf.read(&mut matches).enq()?;

        Ok((hashes, matches))
    }

    /// Check if the secp256k1 kernel is available (for raw key mode).
    pub fn has_secp256k1_kernel(&self) -> bool {
        self.secp256k1_program.is_some()
    }

    /// Generate public keys and address hashes from raw private keys entirely on GPU.
    ///
    /// This is the fast path — no BIP-39/BIP-32, just privkey → pubkey → hash.
    /// Private keys are 32 bytes each, big-endian.
    ///
    /// Returns `(pubkeys: N×33, hashes: N×20, matches: N×u32)`.
    pub fn generate_and_hash_batch(
        &self,
        privkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u32>), GpuError> {
        let program = self.secp256k1_program.as_ref().ok_or_else(|| {
            GpuError::Ocl(ocl::Error::from("secp256k1 kernel not compiled"))
        })?;

        let n = privkeys.len() / PRIVKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("GPU secp256k1 batch: {} private keys", n);

        // Input buffer: private keys
        let privkey_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(privkeys.len())
            .copy_host_slice(privkeys)
            .build()?;

        // Output buffers
        let pubkey_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(n * PUBKEY_SIZE)
            .build()?;

        let hash_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(n * HASH_SIZE)
            .build()?;

        let matches_buf = Buffer::<u32>::builder()
            .queue(self.queue.clone())
            .len(n)
            .fill_val(0u32)
            .build()?;

        let prefix_len = prefix_bytes.len();
        let prefix_buf = if prefix_len > 0 {
            Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .len(prefix_len)
                .copy_host_slice(prefix_bytes)
                .build()?
        } else {
            Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .len(1)
                .fill_val(0u8)
                .build()?
        };

        let kernel = Kernel::builder()
            .program(program)
            .name("generate_addresses")
            .queue(self.queue.clone())
            .global_work_size(n)
            .arg(&privkey_buf)
            .arg(&pubkey_buf)
            .arg(&hash_buf)
            .arg(&prefix_buf)
            .arg(prefix_len as u32)
            .arg(&matches_buf)
            .arg(n as u32)
            .build()?;

        unsafe { kernel.enq()?; }
        self.queue.finish()?;

        // Read results
        let mut pubkeys = vec![0u8; n * PUBKEY_SIZE];
        pubkey_buf.read(&mut pubkeys).enq()?;

        let mut hashes = vec![0u8; n * HASH_SIZE];
        hash_buf.read(&mut hashes).enq()?;

        let mut matches = vec![0u32; n];
        matches_buf.read(&mut matches).enq()?;

        debug!("GPU secp256k1 batch complete: {} keys processed", n);
        Ok((pubkeys, hashes, matches))
    }

    /// Suggested batch size for hybrid mode.
    ///
    /// For the hybrid pipeline, we want large batches (32K-64K+) to
    /// amortize GPU kernel launch overhead. The CPU keygen threads
    /// fill the queue in parallel, so we can afford big batches.
    pub fn suggested_batch_size(&self) -> usize {
        // Target: enough work items to saturate all CUs with large batches
        // Each CU can run many wavefronts (64 threads each on AMD)
        let waves_per_cu = 16; // More waves to keep GPU busy during kernel dispatch
        let wave_size = 64; // AMD wavefront size
        let base = self.max_compute_units as usize * waves_per_cu * wave_size;
        // Round up to power-of-2, minimum 32K for amortized dispatch overhead
        base.max(32_768).next_power_of_two()
    }

    /// Batch size for pure GPU mode — maximum GPU occupancy.
    ///
    /// Uses larger batches (64K-128K) to maximize GPU utilization.
    /// RX 9070 XT: 32 CUs × 64 wavefront × 8 waves = 16K minimum,
    /// so 64K-128K provides excellent occupancy with amortized dispatch.
    pub fn pure_gpu_batch_size(&self) -> usize {
        // For pure GPU mode, go bigger to maximize GPU occupancy
        let waves_per_cu = 32; // More waves for pure GPU saturation
        let wave_size = 64; // AMD wavefront size
        let base = self.max_compute_units as usize * waves_per_cu * wave_size;
        // Minimum 65536 (64K), cap at 131072 (128K) for memory efficiency
        base.max(65_536).min(131_072).next_power_of_two()
    }

    /// Check if the mnemonic pipeline kernel is available.
    pub fn has_mnemonic_kernel(&self) -> bool {
        self.mnemonic_program.is_some()
    }

    /// Batch size for mnemonic GPU mode.
    /// Smaller than raw mode because PBKDF2 (2048 rounds) is heavy per work item.
    pub fn mnemonic_batch_size(&self) -> usize {
        // PBKDF2 is ~4096 SHA-512 compressions per candidate — much heavier
        // Use smaller batches to avoid GPU timeouts
        let waves_per_cu = 4;
        let wave_size = 64;
        let base = self.max_compute_units as usize * waves_per_cu * wave_size;
        base.max(2_048).min(8_192)
    }

    /// Run the full mnemonic pipeline on GPU.
    /// Takes mnemonic UTF-8 strings (zero-padded to 256 bytes each) + their lengths.
    /// Returns (derived_privkeys: N×32, hashes: N×20, matches: N×u32).
    pub fn mnemonic_batch(
        &self,
        mnemonics_flat: &[u8],  // N × 256 bytes, zero-padded
        mnemonic_lens: &[u32],  // N lengths
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u32>), GpuError> {
        let program = self.mnemonic_program.as_ref().ok_or_else(|| {
            GpuError::Ocl(ocl::Error::from("Mnemonic pipeline kernel not compiled"))
        })?;

        let n = mnemonic_lens.len();
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("GPU mnemonic batch: {} candidates", n);

        let mnemonics_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(mnemonics_flat.len())
            .copy_host_slice(mnemonics_flat)
            .build()?;

        let lens_buf = Buffer::<u32>::builder()
            .queue(self.queue.clone())
            .len(n)
            .copy_host_slice(mnemonic_lens)
            .build()?;

        let privkeys_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(n * 32)
            .build()?;

        let hashes_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(n * 20)
            .build()?;

        let prefix_buf = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(1)
            .fill_val(0u8)
            .build()?;

        let matches_buf = Buffer::<u32>::builder()
            .queue(self.queue.clone())
            .len(n)
            .fill_val(0u32)
            .build()?;

        let kernel = Kernel::builder()
            .program(program)
            .name("mnemonic_to_address")
            .queue(self.queue.clone())
            .global_work_size(n)
            .arg(&mnemonics_buf)
            .arg(&lens_buf)
            .arg(&privkeys_buf)
            .arg(&hashes_buf)
            .arg(&prefix_buf)
            .arg(0u32)
            .arg(&matches_buf)
            .arg(n as u32)
            .build()?;

        unsafe { kernel.enq()?; }
        self.queue.finish()?;

        let mut privkeys = vec![0u8; n * 32];
        privkeys_buf.read(&mut privkeys).enq()?;

        let mut hashes = vec![0u8; n * 20];
        hashes_buf.read(&mut hashes).enq()?;

        let mut matches = vec![0u32; n];
        matches_buf.read(&mut matches).enq()?;

        debug!("GPU mnemonic batch complete: {} candidates processed", n);
        Ok((privkeys, hashes, matches))
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device_name)
            .field("compute_units", &self.max_compute_units)
            .field("max_workgroup", &self.max_work_group_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // This just shouldn't panic
        let _ = is_available();
    }

    /// Test that GPU SHA-256 → RIPEMD-160 matches the CPU reference implementation.
    /// Uses known test vectors and the same crypto crates as the address crate.
    #[test]
    fn test_gpu_hash_matches_cpu() {
        use cosmos_vanity_address::pubkey_to_address_bytes;

        // Skip if no GPU available
        if !is_available() {
            eprintln!("No GPU available, skipping GPU hash test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Could not init GPU: {e}, skipping");
                return;
            }
        };

        // Known test public keys (33 bytes each, compressed secp256k1)
        let test_pubkeys: Vec<[u8; 33]> = vec![
            // Test vector 1: from the address crate test
            hex::decode("02394bc53633366a2ab9b5d4a7b6a9cfd9f11d576e45e1e2049a2e397b6e1a4f2e")
                .unwrap().try_into().unwrap(),
            // Test vector 2: all zeros with 0x02 prefix
            {
                let mut k = [0u8; 33];
                k[0] = 0x02;
                k
            },
            // Test vector 3: all 0xFF with 0x03 prefix
            {
                let mut k = [0xFFu8; 33];
                k[0] = 0x03;
                k
            },
            // Test vector 4: sequential bytes
            {
                let mut k = [0u8; 33];
                k[0] = 0x02;
                for i in 1..33 {
                    k[i] = i as u8;
                }
                k
            },
        ];

        // Flatten for GPU
        let mut flat_pubkeys = Vec::with_capacity(test_pubkeys.len() * 33);
        for pk in &test_pubkeys {
            flat_pubkeys.extend_from_slice(pk);
        }

        // GPU hash
        let gpu_hashes = ctx.hash_pubkeys_batch(&flat_pubkeys)
            .expect("GPU hashing failed");

        // Compare each key
        for (i, pk) in test_pubkeys.iter().enumerate() {
            // CPU reference: SHA-256 → RIPEMD-160
            let cpu_hash = pubkey_to_address_bytes(pk).unwrap();

            let gpu_hash = &gpu_hashes[i * 20..(i + 1) * 20];

            assert_eq!(
                cpu_hash.as_slice(), gpu_hash,
                "Hash mismatch for test vector {i}!\n  pubkey: {}\n  CPU:    {}\n  GPU:    {}",
                hex::encode(pk),
                hex::encode(cpu_hash),
                hex::encode(gpu_hash),
            );
        }

        eprintln!("All {} test vectors matched between GPU and CPU!", test_pubkeys.len());
    }

    /// Test with a real derived keypair from mnemonic.
    #[test]
    fn test_gpu_hash_with_derived_key() {
        use cosmos_vanity_address::{pubkey_to_address_bytes, pubkey_to_bech32};
        use cosmos_vanity_keyderiv::{derive_keypair_from_mnemonic, DEFAULT_COSMOS_PATH};

        if !is_available() {
            eprintln!("No GPU available, skipping");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Could not init GPU: {e}, skipping");
                return;
            }
        };

        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
        let key = derive_keypair_from_mnemonic(mnemonic, DEFAULT_COSMOS_PATH).unwrap();
        let pk = key.public_key_bytes();

        // CPU
        let cpu_hash = pubkey_to_address_bytes(pk).unwrap();
        let cpu_addr = pubkey_to_bech32(pk, "cosmos").unwrap();

        // GPU
        let gpu_hashes = ctx.hash_pubkeys_batch(pk).unwrap();
        let gpu_hash = &gpu_hashes[0..20];

        // Build bech32 from GPU hash
        let mut gpu_addr_bytes = [0u8; 20];
        gpu_addr_bytes.copy_from_slice(gpu_hash);
        let gpu_addr = cosmos_vanity_address::encode_bech32("cosmos", &gpu_addr_bytes).unwrap();

        assert_eq!(
            cpu_hash.as_slice(), gpu_hash,
            "Derived key hash mismatch!\n  CPU addr: {cpu_addr}\n  GPU addr: {gpu_addr}\n  CPU hash: {}\n  GPU hash: {}",
            hex::encode(cpu_hash),
            hex::encode(gpu_hash),
        );

        eprintln!("Derived key test passed: {cpu_addr}");
    }

    /// Test GPU secp256k1 with known test vector: private key = 1 (generator point).
    #[test]
    fn test_gpu_secp256k1_known_vector() {
        if !is_available() {
            eprintln!("No GPU available, skipping GPU secp256k1 test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Could not init GPU: {e}, skipping");
                return;
            }
        };

        if !ctx.has_secp256k1_kernel() {
            eprintln!("secp256k1 kernel not compiled, skipping");
            return;
        }

        // Private key = 1 → public key is the generator point G
        let mut privkey = [0u8; 32];
        privkey[31] = 1; // big-endian: 0x00...01

        let (pubkeys, _hashes, _matches) = ctx
            .generate_and_hash_batch(&privkey, &[])
            .expect("GPU secp256k1 failed");

        // Expected compressed pubkey for privkey=1 (generator point G):
        // 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        let expected_hex = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
        let expected = hex::decode(expected_hex).unwrap();

        assert_eq!(
            pubkeys.len(), 33,
            "Expected 33-byte pubkey, got {} bytes", pubkeys.len()
        );

        assert_eq!(
            hex::encode(&pubkeys[..33]),
            expected_hex,
            "GPU pubkey for privkey=1 does not match expected generator point!\n  GPU:      {}\n  Expected: {}",
            hex::encode(&pubkeys[..33]),
            expected_hex,
        );

        eprintln!("✅ GPU secp256k1 known vector test passed! pubkey={}", hex::encode(&pubkeys[..33]));
    }

    /// Test GPU secp256k1 matches CPU reference for multiple random keys.
    #[test]
    fn test_gpu_secp256k1_matches_cpu() {
        use cosmos_vanity_keyderiv::pubkey_from_privkey;

        if !is_available() {
            eprintln!("No GPU available, skipping");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Could not init GPU: {e}, skipping");
                return;
            }
        };

        if !ctx.has_secp256k1_kernel() {
            eprintln!("secp256k1 kernel not compiled, skipping");
            return;
        }

        // Test with several known private keys
        let test_privkeys: Vec<[u8; 32]> = vec![
            // privkey = 1
            {
                let mut k = [0u8; 32];
                k[31] = 1;
                k
            },
            // privkey = 2
            {
                let mut k = [0u8; 32];
                k[31] = 2;
                k
            },
            // privkey = 7
            {
                let mut k = [0u8; 32];
                k[31] = 7;
                k
            },
            // A "real-looking" privkey
            {
                let mut k = [0u8; 32];
                for i in 0..32 { k[i] = (i as u8 + 1) * 7; }
                k
            },
        ];

        // Flatten for GPU
        let mut flat_privkeys = Vec::with_capacity(test_privkeys.len() * 32);
        for pk in &test_privkeys {
            flat_privkeys.extend_from_slice(pk);
        }

        let (gpu_pubkeys, gpu_hashes, _) = ctx
            .generate_and_hash_batch(&flat_privkeys, &[])
            .expect("GPU secp256k1 batch failed");

        // Compare each
        for (i, privkey) in test_privkeys.iter().enumerate() {
            let cpu_pubkey = pubkey_from_privkey(privkey)
                .expect("CPU pubkey derivation failed");

            let gpu_pubkey = &gpu_pubkeys[i * 33..(i + 1) * 33];

            assert_eq!(
                cpu_pubkey.as_slice(), gpu_pubkey,
                "Pubkey mismatch for test vector {i}!\n  privkey: {}\n  CPU: {}\n  GPU: {}",
                hex::encode(privkey),
                hex::encode(cpu_pubkey),
                hex::encode(gpu_pubkey),
            );

            // Also verify hash matches
            let cpu_hash = cosmos_vanity_address::pubkey_to_address_bytes(&cpu_pubkey).unwrap();
            let gpu_hash = &gpu_hashes[i * 20..(i + 1) * 20];

            assert_eq!(
                cpu_hash.as_slice(), gpu_hash,
                "Hash mismatch for test vector {i}!\n  CPU: {}\n  GPU: {}",
                hex::encode(cpu_hash),
                hex::encode(gpu_hash),
            );
        }

        eprintln!("✅ All {} GPU secp256k1 test vectors matched CPU!", test_privkeys.len());
    }

    /// Test GPU mnemonic pipeline against known test vector.
    /// "monster asthma shaft average main office dial since rural guitar estate sight"
    /// → cosmos1u2gukdek3gtxgz6f89jgvh7pw2286smk48vxm4
    #[test]
    fn test_gpu_mnemonic_pipeline() {
        if !is_available() {
            eprintln!("No GPU available, skipping mnemonic pipeline test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Could not init GPU: {e}, skipping");
                return;
            }
        };

        if !ctx.has_mnemonic_kernel() {
            eprintln!("Mnemonic pipeline kernel not compiled, skipping");
            return;
        }

        let mnemonic = "monster asthma shaft average main office dial since rural guitar estate sight";
        let mnemonic_bytes = mnemonic.as_bytes();
        let mnemonic_len = mnemonic_bytes.len() as u32;

        // Zero-pad to 256 bytes
        let mut padded = vec![0u8; 256];
        padded[..mnemonic_bytes.len()].copy_from_slice(mnemonic_bytes);

        let (privkeys, hashes, _matches) = ctx
            .mnemonic_batch(&padded, &[mnemonic_len])
            .expect("GPU mnemonic pipeline failed");

        // Verify: derive on CPU for comparison
        let cpu_key = cosmos_vanity_keyderiv::derive_keypair_from_mnemonic(
            mnemonic,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
        ).unwrap();
        let cpu_addr = cosmos_vanity_address::pubkey_to_bech32(cpu_key.public_key_bytes(), "cosmos").unwrap();

        // GPU derived privkey should match CPU
        let gpu_privkey = &privkeys[..32];
        assert_eq!(
            cpu_key.secret_key_bytes(), gpu_privkey,
            "Private key mismatch!\n  CPU: {}\n  GPU: {}",
            hex::encode(cpu_key.secret_key_bytes()),
            hex::encode(gpu_privkey),
        );

        // GPU hash → bech32 should match CPU address
        let mut gpu_addr_bytes = [0u8; 20];
        gpu_addr_bytes.copy_from_slice(&hashes[..20]);
        let gpu_addr = cosmos_vanity_address::encode_bech32("cosmos", &gpu_addr_bytes).unwrap();

        assert_eq!(
            cpu_addr, gpu_addr,
            "Address mismatch!\n  CPU: {}\n  GPU: {}",
            cpu_addr, gpu_addr,
        );

        assert_eq!(cpu_addr, "cosmos1u2gukdek3gtxgz6f89jgvh7pw2286smk48vxm4");
        eprintln!("✅ GPU mnemonic pipeline test passed! {}", gpu_addr);
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    use ocl::{Buffer, Kernel};

    fn get_mnemonic_program() -> (GpuContext, ocl::Program) {
        let ctx = GpuContext::new().expect("No GPU");
        assert!(ctx.has_mnemonic_kernel(), "No mnemonic kernel");
        // Re-build the mnemonic program for our diagnostic kernels
        let prog = ctx.mnemonic_program.as_ref().unwrap().clone();
        (ctx, prog)
    }

    #[test]
    fn test_gpu_sha512() {
        let ctx = GpuContext::new().expect("No GPU");
        if !ctx.has_mnemonic_kernel() { return; }
        let prog = ctx.mnemonic_program.as_ref().unwrap();

        // Test vector: SHA-512("abc")
        let input = b"abc";
        let expected = "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f";

        let input_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(input.len()).copy_host_slice(input).build().unwrap();
        let output_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(64).fill_val(0u8).build().unwrap();

        let kernel = Kernel::builder()
            .program(prog)
            .name("test_sha512_kernel")
            .queue(ctx.queue.clone())
            .global_work_size(1)
            .arg(&input_buf)
            .arg(input.len() as u32)
            .arg(&output_buf)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        let mut result = vec![0u8; 64];
        output_buf.read(&mut result).enq().unwrap();

        eprintln!("GPU SHA-512(abc): {}", hex::encode(&result));
        eprintln!("Expected:         {}", expected);
        assert_eq!(hex::encode(&result), expected, "SHA-512 mismatch!");
        eprintln!("✅ SHA-512 test passed");
    }

    #[test]
    fn test_gpu_hmac_sha512() {
        let ctx = GpuContext::new().expect("No GPU");
        if !ctx.has_mnemonic_kernel() { return; }
        let prog = ctx.mnemonic_program.as_ref().unwrap();

        // Test: HMAC-SHA512(key="Bitcoin seed", data=known_seed)
        let key = b"Bitcoin seed";
        let seed_hex = "e2fcb5bec6933a405806d6e8461a562036d584fe55958e13f4705014c936f716db0009b8558ad35cd7d023d043b0c51bbb9a4dc810addd4cfbb4604115224dd8";
        let seed = hex::decode(seed_hex).unwrap();
        
        // Expected master key from CPU
        let expected_il = "4ce7e11819686786a7a01310c3b5e9c7257d724253525d4502840a7abac0e2c7";
        let expected_ir = "9930a92a22307366bc88347a037a203fe4ce27858c68cff91195c24dcd6aef14";

        let key_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(key.len()).copy_host_slice(key).build().unwrap();
        let msg_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(seed.len()).copy_host_slice(&seed).build().unwrap();
        let output_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(64).fill_val(0u8).build().unwrap();

        let kernel = Kernel::builder()
            .program(prog)
            .name("test_hmac_sha512_kernel")
            .queue(ctx.queue.clone())
            .global_work_size(1)
            .arg(&key_buf)
            .arg(key.len() as u32)
            .arg(&msg_buf)
            .arg(seed.len() as u32)
            .arg(&output_buf)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        let mut result = vec![0u8; 64];
        output_buf.read(&mut result).enq().unwrap();

        eprintln!("GPU HMAC IL: {}", hex::encode(&result[..32]));
        eprintln!("Expected IL: {}", expected_il);
        eprintln!("GPU HMAC IR: {}", hex::encode(&result[32..]));
        eprintln!("Expected IR: {}", expected_ir);
        assert_eq!(hex::encode(&result[..32]), expected_il, "HMAC IL mismatch!");
        assert_eq!(hex::encode(&result[32..]), expected_ir, "HMAC IR mismatch!");
        eprintln!("✅ HMAC-SHA512 test passed");
    }

    fn run_pbkdf2_test(ctx: &GpuContext, prog: &ocl::Program, password: &[u8], salt: &[u8], iterations: u32) -> Vec<u8> {
        let pw_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(password.len()).copy_host_slice(password).build().unwrap();
        let salt_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(salt.len()).copy_host_slice(salt).build().unwrap();
        let output_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(64).fill_val(0u8).build().unwrap();

        let kernel = Kernel::builder()
            .program(prog)
            .name("test_pbkdf2_kernel")
            .queue(ctx.queue.clone())
            .global_work_size(1)
            .arg(&pw_buf)
            .arg(password.len() as u32)
            .arg(&salt_buf)
            .arg(salt.len() as u32)
            .arg(iterations)
            .arg(&output_buf)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        let mut result = vec![0u8; 64];
        output_buf.read(&mut result).enq().unwrap();
        result
    }

    #[test]
    fn test_gpu_pbkdf2() {
        let ctx = GpuContext::new().expect("No GPU");
        if !ctx.has_mnemonic_kernel() { return; }
        let prog = ctx.mnemonic_program.as_ref().unwrap();

        let mnemonic = b"monster asthma shaft average main office dial since rural guitar estate sight";
        let salt = b"mnemonic";

        // Test with 1 iteration first
        let expected_1 = "4d61edadb848eb918ab4044174a7fd0a54d396ab3de52e0144793f17b6028f182a0690848144070756018101c51d3ef36ef19342e8c0b5d771ccacc5f31ecfb6";
        let result_1 = run_pbkdf2_test(&ctx, prog, mnemonic, salt, 1);
        eprintln!("GPU PBKDF2 1-iter: {}", hex::encode(&result_1));
        eprintln!("Expected 1-iter:   {}", expected_1);
        
        // Test with 2 iterations
        let expected_2 = "e4bed65f7fa4bfcba9ded26de4d81089a853feb2e80260a9ee41615b7b2d70147273b8fae564639371eaaf8dd2502ae01518678b711a37cc57ff10c52964ee7f";
        let result_2 = run_pbkdf2_test(&ctx, prog, mnemonic, salt, 2);
        eprintln!("GPU PBKDF2 2-iter: {}", hex::encode(&result_2));
        eprintln!("Expected 2-iter:   {}", expected_2);

        // Test HMAC with key=76 bytes, msg=12 bytes (same shape as PBKDF2 U1)
        {
            let test_key = vec![0x41u8; 76]; // 'A' * 76
            let test_msg = vec![0x42u8; 12]; // 'B' * 12
            let expected_hmac = "aec0594990ee8164428b606233d47d68f553f7a4c15b4bbfe97bb46d8e15be73f89e2785f637b0d20351a9bb784d070b2cc03a9e19ddcdc109fe40201eddba19";
            
            let key_buf = Buffer::<u8>::builder()
                .queue(ctx.queue.clone()).len(76).copy_host_slice(&test_key).build().unwrap();
            let msg_buf = Buffer::<u8>::builder()
                .queue(ctx.queue.clone()).len(12).copy_host_slice(&test_msg).build().unwrap();
            let out_buf = Buffer::<u8>::builder()
                .queue(ctx.queue.clone()).len(64).fill_val(0u8).build().unwrap();
            
            let k = Kernel::builder()
                .program(prog)
                .name("test_hmac_sha512_kernel")
                .queue(ctx.queue.clone())
                .global_work_size(1)
                .arg(&key_buf)
                .arg(76u32)
                .arg(&msg_buf)
                .arg(12u32)
                .arg(&out_buf)
                .build().unwrap();
            unsafe { k.enq().unwrap(); }
            let mut hmac_result = vec![0u8; 64];
            out_buf.read(&mut hmac_result).enq().unwrap();
            eprintln!("GPU HMAC(A*76, B*12): {}", hex::encode(&hmac_result));
            eprintln!("Expected:             {}", expected_hmac);
            assert_eq!(hex::encode(&hmac_result), expected_hmac, "HMAC 76-byte key test failed!");
            eprintln!("✅ HMAC 76-byte key test passed");
        }

        // Also test HMAC directly with same inputs as PBKDF2 U1
        {
            let salt_ext: Vec<u8> = salt.iter().copied().chain([0, 0, 0, 1].iter().copied()).collect();
            let key_buf = Buffer::<u8>::builder()
                .queue(ctx.queue.clone()).len(mnemonic.len()).copy_host_slice(mnemonic).build().unwrap();
            let msg_buf = Buffer::<u8>::builder()
                .queue(ctx.queue.clone()).len(salt_ext.len()).copy_host_slice(&salt_ext).build().unwrap();
            let out_buf = Buffer::<u8>::builder()
                .queue(ctx.queue.clone()).len(64).fill_val(0u8).build().unwrap();

            let k = Kernel::builder()
                .program(prog)
                .name("test_hmac_sha512_kernel")
                .queue(ctx.queue.clone())
                .global_work_size(1)
                .arg(&key_buf)
                .arg(mnemonic.len() as u32)
                .arg(&msg_buf)
                .arg(salt_ext.len() as u32)
                .arg(&out_buf)
                .build().unwrap();
            unsafe { k.enq().unwrap(); }
            let mut hmac_result = vec![0u8; 64];
            out_buf.read(&mut hmac_result).enq().unwrap();
            eprintln!("GPU direct HMAC(mnemonic, salt||1): {}", hex::encode(&hmac_result));
            eprintln!("Expected:                            {}", expected_1);
            assert_eq!(hex::encode(&hmac_result), expected_1, "Direct HMAC for U1 mismatch - pointer issue!");
        }

        assert_eq!(hex::encode(&result_1), expected_1, "PBKDF2 1-iter mismatch!");
        assert_eq!(hex::encode(&result_2), expected_2, "PBKDF2 2-iter mismatch!");
        
        // Full 2048 iterations
        let expected_seed = "e2fcb5bec6933a405806d6e8461a562036d584fe55958e13f4705014c936f716db0009b8558ad35cd7d023d043b0c51bbb9a4dc810addd4cfbb4604115224dd8";
        let result = run_pbkdf2_test(&ctx, prog, mnemonic, salt, 2048);
        eprintln!("GPU PBKDF2 2048:   {}", hex::encode(&result));
        eprintln!("Expected 2048:     {}", expected_seed);
        assert_eq!(hex::encode(&result), expected_seed, "PBKDF2 2048-iter mismatch!");
        eprintln!("✅ PBKDF2 test passed");
    }

    #[test]
    fn test_gpu_bip32() {
        let ctx = GpuContext::new().expect("No GPU");
        if !ctx.has_mnemonic_kernel() { return; }
        let prog = ctx.mnemonic_program.as_ref().unwrap();

        let seed_hex = "e2fcb5bec6933a405806d6e8461a562036d584fe55958e13f4705014c936f716db0009b8558ad35cd7d023d043b0c51bbb9a4dc810addd4cfbb4604115224dd8";
        let seed = hex::decode(seed_hex).unwrap();

        let expected_keys = [
            "4ce7e11819686786a7a01310c3b5e9c7257d724253525d4502840a7abac0e2c7", // master
            "b47ebdd2866a6deabadb573fae5da2760fe6dd7ac5b71a8246e77cfdd375a7d1", // 44'
            "330b95558f5640a804e51922e71d7ba8bc26169b730a1f1c262adc9e9365a339", // 118'
            "8eb2adf81551e45133ec870200c9b338efb5df52af20d4b40f60fd10695f553b", // 0'
            "2ec9f25625f0043b96b13e0436bb597f8e4cceef31b246c9c13717722193597b", // 0
            "9094ec80ab633ffa6888a996ed97fb46e0880526cae8582461655377d20c08c1", // 0 (final)
        ];

        let seed_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(64).copy_host_slice(&seed).build().unwrap();
        let output_buf = Buffer::<u8>::builder()
            .queue(ctx.queue.clone()).len(32 * 6).fill_val(0u8).build().unwrap();

        let kernel = Kernel::builder()
            .program(prog)
            .name("test_bip32_kernel")
            .queue(ctx.queue.clone())
            .global_work_size(1)
            .arg(&seed_buf)
            .arg(&output_buf)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        let mut result = vec![0u8; 32 * 6];
        output_buf.read(&mut result).enq().unwrap();

        let names = ["master", "44'", "118'", "0'", "0 (change)", "0 (index)"];
        let mut all_ok = true;
        for (i, (name, exp)) in names.iter().zip(expected_keys.iter()).enumerate() {
            let got = hex::encode(&result[i*32..(i+1)*32]);
            let ok = got == *exp;
            eprintln!("{} {} key: {}", if ok { "✅" } else { "❌" }, name, got);
            if !ok {
                eprintln!("   Expected:  {}", exp);
                all_ok = false;
            }
        }
        assert!(all_ok, "BIP-32 derivation mismatch!");
    }
}
