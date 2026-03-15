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

        Ok(Self {
            queue,
            program,
            secp256k1_program,
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
}
