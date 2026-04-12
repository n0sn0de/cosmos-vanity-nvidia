//! # cosmos-vanity-bench
//!
//! Benchmarking utilities for measuring vanity address generation performance.

use std::time::Instant;

use cosmos_vanity_address::pubkey_to_bech32;
use cosmos_vanity_keyderiv::generate_random_keypair;

/// Result of a benchmark run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchResult {
    /// Number of addresses generated
    pub iterations: u64,

    /// Total time taken
    pub duration_secs: f64,

    /// Addresses per second
    pub addresses_per_sec: f64,

    /// Average time per address in microseconds
    pub avg_us_per_address: f64,

    /// Whether GPU was used
    pub gpu: bool,

    /// Number of threads used
    pub threads: usize,
}

/// Benchmark CPU address generation rate.
///
/// Generates `iterations` random keypairs and addresses, measuring throughput.
pub fn bench_cpu_throughput(iterations: u64, hrp: &str) -> BenchResult {
    let start = Instant::now();

    for _ in 0..iterations {
        let key = generate_random_keypair().expect("keygen failed");
        let _address = pubkey_to_bech32(key.public_key_bytes(), hrp).expect("address gen failed");
    }

    let duration = start.elapsed();
    let secs = duration.as_secs_f64();

    BenchResult {
        iterations,
        duration_secs: secs,
        addresses_per_sec: iterations as f64 / secs,
        avg_us_per_address: (secs * 1_000_000.0) / iterations as f64,
        gpu: false,
        threads: 1,
    }
}

/// Benchmark just the key derivation step (no address encoding).
pub fn bench_key_derivation(iterations: u64) -> BenchResult {
    let start = Instant::now();

    for _ in 0..iterations {
        let _key = generate_random_keypair().expect("keygen failed");
    }

    let duration = start.elapsed();
    let secs = duration.as_secs_f64();

    BenchResult {
        iterations,
        duration_secs: secs,
        addresses_per_sec: iterations as f64 / secs,
        avg_us_per_address: (secs * 1_000_000.0) / iterations as f64,
        gpu: false,
        threads: 1,
    }
}

/// Benchmark just the address hashing step (SHA-256 + RIPEMD-160 + Bech32).
pub fn bench_address_hashing(iterations: u64, hrp: &str) -> BenchResult {
    // Pre-generate a key to isolate hashing
    let key = generate_random_keypair().expect("keygen failed");
    let pubkey = key.public_key_bytes().to_vec();

    let start = Instant::now();

    for _ in 0..iterations {
        let _address = pubkey_to_bech32(&pubkey, hrp).expect("address gen failed");
    }

    let duration = start.elapsed();
    let secs = duration.as_secs_f64();

    BenchResult {
        iterations,
        duration_secs: secs,
        addresses_per_sec: iterations as f64 / secs,
        avg_us_per_address: (secs * 1_000_000.0) / iterations as f64,
        gpu: false,
        threads: 1,
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark Results:")?;
        writeln!(f, "  Iterations:     {}", self.iterations)?;
        writeln!(f, "  Duration:       {:.3}s", self.duration_secs)?;
        writeln!(
            f,
            "  Throughput:     {:.0} addr/sec",
            self.addresses_per_sec
        )?;
        writeln!(f, "  Avg per addr:   {:.1} µs", self.avg_us_per_address)?;
        writeln!(f, "  GPU:            {}", self.gpu)?;
        writeln!(f, "  Threads:        {}", self.threads)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_cpu_throughput() {
        let result = bench_cpu_throughput(10, "cosmos");
        assert_eq!(result.iterations, 10);
        assert!(result.addresses_per_sec > 0.0);
        assert!(!result.gpu);
    }

    #[test]
    fn test_bench_key_derivation() {
        let result = bench_key_derivation(10);
        assert_eq!(result.iterations, 10);
        assert!(result.addresses_per_sec > 0.0);
    }

    #[test]
    fn test_bench_address_hashing() {
        let result = bench_address_hashing(100, "cosmos");
        assert_eq!(result.iterations, 100);
        // Hashing should be much faster than full keygen
        assert!(result.addresses_per_sec > 0.0);
    }

    #[test]
    fn test_bench_result_display() {
        let result = BenchResult {
            iterations: 1000,
            duration_secs: 1.5,
            addresses_per_sec: 666.67,
            avg_us_per_address: 1500.0,
            gpu: false,
            threads: 4,
        };
        let display = format!("{result}");
        assert!(display.contains("667")); // 666.67 rounds to 667
    }
}
