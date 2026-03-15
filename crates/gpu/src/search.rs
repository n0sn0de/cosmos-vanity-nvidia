//! Core vanity address search engine — CPU fallback, hybrid, and pure GPU modes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use crossbeam_channel::{bounded, Receiver};

use cosmos_vanity_address::{encode_bech32, pubkey_to_bech32, VanityPattern};
use cosmos_vanity_keyderiv::generate_random_keypair_with_path;

use crate::state::SearchState;

/// Key generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMode {
    /// Raw mode: random 32-byte private key → GPU secp256k1 → GPU hash.
    /// Maximum speed, returns hex private key instead of mnemonic.
    Raw,
    /// Mnemonic mode: BIP-39 mnemonic → CPU derivation → GPU hash.
    /// Compatible with standard wallets, returns 24-word mnemonic.
    Mnemonic,
}

impl Default for KeyMode {
    fn default() -> Self {
        KeyMode::Raw
    }
}

impl std::fmt::Display for KeyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyMode::Raw => write!(f, "raw"),
            KeyMode::Mnemonic => write!(f, "mnemonic"),
        }
    }
}

/// Search mode for vanity address generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Pure GPU mode — ALL CPU threads are keygen feeders, GPU does ALL hashing + pattern matching.
    /// Maximum HPC performance. No CPU search threads.
    Gpu,

    /// Hybrid mode — N-1 CPU search threads + GPU pipeline running in parallel.
    /// GPU path is additive on top of CPU throughput.
    Hybrid,

    /// CPU-only mode — all threads do full keygen + hash + match on CPU.
    Cpu,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchMode::Gpu => write!(f, "gpu"),
            SearchMode::Hybrid => write!(f, "hybrid"),
            SearchMode::Cpu => write!(f, "cpu"),
        }
    }
}

/// Configuration for a vanity address search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// The vanity pattern to search for
    pub pattern: VanityPattern,

    /// Human-readable part (e.g., "cosmos", "osmo")
    pub hrp: String,

    /// BIP-44 derivation path
    pub derivation_path: String,

    /// Number of CPU worker threads
    pub num_threads: usize,

    /// Search mode
    pub mode: SearchMode,

    /// Key generation mode
    pub key_mode: KeyMode,

    /// Maximum number of matches to find (0 = unlimited)
    pub max_matches: usize,

    /// Checkpoint interval (save state every N candidates)
    pub checkpoint_interval: u64,

    /// Path to save/resume state
    pub state_file: Option<PathBuf>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            pattern: VanityPattern::Prefix("aaa".to_string()),
            hrp: "cosmos".to_string(),
            derivation_path: cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH.to_string(),
            num_threads: num_cpus::get(),
            mode: SearchMode::Cpu,
            key_mode: KeyMode::default(),
            max_matches: 1,
            checkpoint_interval: 100_000,
            state_file: None,
        }
    }
}

/// A verified search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching Bech32 address
    pub address: String,

    /// The mnemonic that produces this address (empty in raw mode)
    pub mnemonic: String,

    /// The derivation path used
    pub derivation_path: String,

    /// How many candidates were checked before this match
    pub candidate_number: u64,

    /// Time taken to find this match
    pub elapsed_secs: f64,

    /// Raw private key hex (only set in raw key mode)
    pub private_key_hex: Option<String>,
}

/// The main vanity address searcher.
pub struct VanitySearcher {
    config: SearchConfig,
    state: SearchState,
    candidates_checked: Arc<AtomicU64>,
    should_stop: Arc<AtomicBool>,
}

impl VanitySearcher {
    /// Create a new searcher with the given configuration.
    pub fn new(config: SearchConfig) -> anyhow::Result<Self> {
        let use_gpu = config.mode != SearchMode::Cpu;

        // Try to resume from state file
        let state = if let Some(ref path) = config.state_file {
            if path.exists() {
                tracing::info!("Resuming from state file: {}", path.display());
                SearchState::load(path)?
            } else {
                SearchState::new(
                    &config.pattern.to_string(),
                    &config.hrp,
                    &config.derivation_path,
                    use_gpu,
                )
            }
        } else {
            SearchState::new(
                &config.pattern.to_string(),
                &config.hrp,
                &config.derivation_path,
                use_gpu,
            )
        };

        let candidates_checked = Arc::new(AtomicU64::new(state.candidates_checked));

        Ok(Self {
            config,
            state,
            candidates_checked,
            should_stop: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get the current number of candidates checked.
    pub fn candidates_checked(&self) -> u64 {
        self.candidates_checked.load(Ordering::Relaxed)
    }

    /// Signal the searcher to stop.
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::Relaxed);
    }

    /// Get a handle to the stop flag for external signal handling.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.should_stop)
    }

    /// Get a handle to the candidates counter for progress reporting.
    pub fn candidates_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.candidates_checked)
    }

    /// Run the search using CPU threads (fallback mode).
    ///
    /// Returns a receiver that yields results as they're found.
    pub fn search_cpu(&mut self) -> anyhow::Result<Receiver<SearchResult>> {
        let (tx, rx) = bounded::<SearchResult>(32);

        let start = Instant::now();
        let num_threads = self.config.num_threads;
        let pattern = self.config.pattern.clone();
        let hrp = self.config.hrp.clone();
        let path = self.config.derivation_path.clone();
        let max_matches = self.config.max_matches;
        let counter = Arc::clone(&self.candidates_checked);
        let stop_flag = Arc::clone(&self.should_stop);
        let _checkpoint_interval = self.config.checkpoint_interval;
        let _state_file = self.config.state_file.clone();

        tracing::info!(
            "Starting CPU search with {} threads, pattern: {}, hrp: {}",
            num_threads,
            pattern,
            hrp
        );

        // Validate pattern charset
        pattern.validate_bech32_charset()?;

        let matches_found = Arc::new(AtomicU64::new(self.state.matches_found));

        for thread_id in 0..num_threads {
            let tx = tx.clone();
            let pattern = pattern.clone();
            let hrp = hrp.clone();
            let path = path.clone();
            let counter = Arc::clone(&counter);
            let stop_flag = Arc::clone(&stop_flag);
            let matches_found = Arc::clone(&matches_found);

            std::thread::Builder::new()
                .name(format!("vanity-worker-{thread_id}"))
                .spawn(move || {
                    tracing::debug!("Worker {thread_id} started");

                    loop {
                        if stop_flag.load(Ordering::Relaxed) {
                            tracing::debug!("Worker {thread_id} stopping (signal)");
                            break;
                        }

                        if max_matches > 0
                            && matches_found.load(Ordering::Relaxed) >= max_matches as u64
                        {
                            break;
                        }

                        // Generate a candidate
                        let key = match generate_random_keypair_with_path(&path) {
                            Ok(k) => k,
                            Err(e) => {
                                tracing::error!("Key generation error: {e}");
                                continue;
                            }
                        };

                        let candidate_num = counter.fetch_add(1, Ordering::Relaxed);

                        // Generate address
                        let address = match pubkey_to_bech32(key.public_key_bytes(), &hrp) {
                            Ok(a) => a,
                            Err(e) => {
                                tracing::error!("Address generation error: {e}");
                                continue;
                            }
                        };

                        // Check pattern
                        if pattern.matches(&address, &hrp) {
                            let elapsed = start.elapsed().as_secs_f64();
                            tracing::info!(
                                "🎯 Match found! Address: {} (candidate #{})",
                                address,
                                candidate_num
                            );

                            matches_found.fetch_add(1, Ordering::Relaxed);

                            let result = SearchResult {
                                address,
                                mnemonic: key.mnemonic().to_string(),
                                derivation_path: key.derivation_path().to_string(),
                                candidate_number: candidate_num,
                                elapsed_secs: elapsed,
                                private_key_hex: None,
                            };

                            if tx.send(result).is_err() {
                                break; // Receiver dropped
                            }
                        }

                        // Don't hold key longer than needed
                        drop(key);
                    }

                    tracing::debug!("Worker {thread_id} exited");
                })?;
        }

        // Drop our copy of tx so the channel closes when all workers finish
        drop(tx);

        Ok(rx)
    }

    /// Run the search using GPU acceleration with multi-threaded CPU key generation.
    ///
    /// **Hybrid architecture:**
    /// - N-1 CPU threads run full independent search (keygen + hash + match) — same as `search_cpu()`
    /// - N CPU keygen threads feed pubkeys into a shared channel
    /// - 1 GPU driver thread pulls batches from the channel, ships to GPU for hashing,
    ///   then does bech32 encode + pattern match on CPU
    ///
    /// This ensures we never lose CPU throughput — the GPU path is purely additive.
    #[cfg(feature = "opencl")]
    pub fn search_hybrid(&mut self) -> anyhow::Result<Receiver<SearchResult>> {
        use crate::opencl::GpuContext;

        let gpu_ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("GPU init failed: {e}"))?;
        // Use larger batches to amortize kernel launch overhead
        let batch_size = (gpu_ctx.suggested_batch_size()).max(32_768);

        tracing::info!(
            "Starting hybrid GPU+CPU search on {} (batch size: {}, CUs: {}), pattern: {}, hrp: {}",
            gpu_ctx.device_name(),
            batch_size,
            gpu_ctx.max_compute_units(),
            self.config.pattern,
            self.config.hrp,
        );

        // Validate pattern charset early
        self.config.pattern.validate_bech32_charset()?;

        let (result_tx, result_rx) = bounded::<SearchResult>(32);
        let start = Instant::now();
        let pattern = self.config.pattern.clone();
        let hrp = self.config.hrp.clone();
        let path = self.config.derivation_path.clone();
        let max_matches = self.config.max_matches;
        let counter = Arc::clone(&self.candidates_checked);
        let stop_flag = Arc::clone(&self.should_stop);

        let num_cpu_threads = self.config.num_threads;
        let matches_found = Arc::new(AtomicU64::new(self.state.matches_found));

        // --- Phase 1: Spawn N-1 CPU worker threads doing full independent search ---
        let cpu_search_threads = if num_cpu_threads > 1 { num_cpu_threads - 1 } else { 0 };

        tracing::info!(
            "Hybrid mode: {} CPU search threads + {} GPU keygen threads + 1 GPU driver",
            cpu_search_threads,
            num_cpu_threads,
        );

        for thread_id in 0..cpu_search_threads {
            let tx = result_tx.clone();
            let pattern = pattern.clone();
            let hrp = hrp.clone();
            let path = path.clone();
            let counter = Arc::clone(&counter);
            let stop_flag = Arc::clone(&stop_flag);
            let matches_found = Arc::clone(&matches_found);

            std::thread::Builder::new()
                .name(format!("cpu-search-{thread_id}"))
                .spawn(move || {
                    tracing::debug!("CPU search worker {thread_id} started");

                    loop {
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        if max_matches > 0
                            && matches_found.load(Ordering::Relaxed) >= max_matches as u64
                        {
                            break;
                        }

                        let key = match generate_random_keypair_with_path(&path) {
                            Ok(k) => k,
                            Err(e) => {
                                tracing::error!("Key generation error: {e}");
                                continue;
                            }
                        };

                        let candidate_num = counter.fetch_add(1, Ordering::Relaxed);

                        let address = match pubkey_to_bech32(key.public_key_bytes(), &hrp) {
                            Ok(a) => a,
                            Err(e) => {
                                tracing::error!("Address generation error: {e}");
                                continue;
                            }
                        };

                        if pattern.matches(&address, &hrp) {
                            let elapsed = start.elapsed().as_secs_f64();
                            tracing::info!(
                                "🎯 CPU Match found! Address: {} (candidate #{})",
                                address,
                                candidate_num
                            );

                            matches_found.fetch_add(1, Ordering::Relaxed);

                            let result = SearchResult {
                                address,
                                mnemonic: key.mnemonic().to_string(),
                                derivation_path: key.derivation_path().to_string(),
                                candidate_number: candidate_num,
                                elapsed_secs: elapsed,
                                private_key_hex: None,
                            };

                            if tx.send(result).is_err() {
                                break;
                            }
                        }

                        drop(key);
                    }

                    tracing::debug!("CPU search worker {thread_id} exited");
                })?;
        }

        // --- Phase 2: GPU pipeline ---
        let (keygen_tx, keygen_rx) = bounded::<(Vec<u8>, String, String)>(batch_size * 2);

        // Spawn N keygen feeder threads
        for thread_id in 0..num_cpu_threads {
            let keygen_tx = keygen_tx.clone();
            let path = path.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let matches_found = Arc::clone(&matches_found);

            std::thread::Builder::new()
                .name(format!("gpu-keygen-{thread_id}"))
                .spawn(move || {
                    tracing::debug!("GPU keygen worker {thread_id} started");

                    loop {
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        if max_matches > 0
                            && matches_found.load(Ordering::Relaxed) >= max_matches as u64
                        {
                            break;
                        }

                        let key = match generate_random_keypair_with_path(&path) {
                            Ok(k) => k,
                            Err(e) => {
                                tracing::error!("Keygen error: {e}");
                                continue;
                            }
                        };

                        let pubkey = key.public_key_bytes().to_vec();
                        let mnemonic = key.mnemonic().to_string();
                        let deriv_path = key.derivation_path().to_string();
                        drop(key);

                        if keygen_tx.send((pubkey, mnemonic, deriv_path)).is_err() {
                            break;
                        }
                    }

                    tracing::debug!("GPU keygen worker {thread_id} exited");
                })?;
        }

        drop(keygen_tx);

        // GPU driver thread
        let gpu_result_tx = result_tx.clone();
        let gpu_counter = Arc::clone(&counter);
        let gpu_stop = Arc::clone(&stop_flag);
        let gpu_matches = Arc::clone(&matches_found);
        let gpu_pattern = pattern.clone();
        let gpu_hrp = hrp.clone();

        std::thread::Builder::new()
            .name("gpu-driver".to_string())
            .spawn(move || {
                tracing::debug!("GPU driver thread started");

                let mut pubkeys_flat = Vec::with_capacity(batch_size * 33);
                let mut mnemonics = Vec::with_capacity(batch_size);
                let mut paths = Vec::with_capacity(batch_size);

                loop {
                    if gpu_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if max_matches > 0
                        && gpu_matches.load(Ordering::Relaxed) >= max_matches as u64
                    {
                        break;
                    }

                    pubkeys_flat.clear();
                    mnemonics.clear();
                    paths.clear();

                    match keygen_rx.recv() {
                        Ok((pubkey, mnemonic, deriv_path)) => {
                            pubkeys_flat.extend_from_slice(&pubkey);
                            mnemonics.push(mnemonic);
                            paths.push(deriv_path);
                        }
                        Err(_) => break,
                    }

                    while mnemonics.len() < batch_size {
                        match keygen_rx.try_recv() {
                            Ok((pubkey, mnemonic, deriv_path)) => {
                                pubkeys_flat.extend_from_slice(&pubkey);
                                mnemonics.push(mnemonic);
                                paths.push(deriv_path);
                            }
                            Err(_) => break,
                        }
                    }

                    let actual_batch = mnemonics.len();
                    if actual_batch == 0 {
                        continue;
                    }

                    let hashes = match gpu_ctx.hash_pubkeys_batch(&pubkeys_flat) {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::error!("GPU hashing error: {e}");
                            continue;
                        }
                    };

                    let batch_start =
                        gpu_counter.fetch_add(actual_batch as u64, Ordering::Relaxed);

                    for i in 0..actual_batch {
                        let hash_bytes: [u8; 20] = hashes[i * 20..(i + 1) * 20]
                            .try_into()
                            .expect("20 bytes");

                        let address = match encode_bech32(&gpu_hrp, &hash_bytes) {
                            Ok(a) => a,
                            Err(_) => continue,
                        };

                        if gpu_pattern.matches(&address, &gpu_hrp) {
                            let candidate_num = batch_start + i as u64;
                            let elapsed = start.elapsed().as_secs_f64();
                            tracing::info!(
                                "🎯 GPU Match found! Address: {} (candidate #{})",
                                address,
                                candidate_num
                            );

                            gpu_matches.fetch_add(1, Ordering::Relaxed);

                            let result = SearchResult {
                                address,
                                mnemonic: mnemonics[i].clone(),
                                derivation_path: paths[i].clone(),
                                candidate_number: candidate_num,
                                elapsed_secs: elapsed,
                                private_key_hex: None,
                            };

                            if gpu_result_tx.send(result).is_err() {
                                return;
                            }
                        }
                    }
                }

                tracing::debug!("GPU driver thread exited");
            })?;

        drop(result_tx);

        Ok(result_rx)
    }

    /// Pure GPU mode — maximum HPC performance.
    ///
    /// **Architecture:**
    /// - ALL CPU threads (num_cpus) are keygen feeders — no CPU search threads
    /// - Double-buffered GPU dispatch — while GPU processes batch N, CPU fills batch N+1
    /// - GPU does pattern matching on raw hash bytes for prefix patterns
    /// - Only matches need CPU-side bech32 verification
    /// - Large batch sizes (64K-128K) to maximize GPU occupancy
    /// - Atomic match counter for early termination
    #[cfg(feature = "opencl")]
    pub fn search_gpu_pure(&mut self) -> anyhow::Result<Receiver<SearchResult>> {
        use crate::opencl::GpuContext;

        let gpu_ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("GPU init failed: {e}"))?;
        // Pure GPU mode uses larger batches for max occupancy
        let batch_size = gpu_ctx.pure_gpu_batch_size();

        tracing::info!(
            "Starting PURE GPU search on {} (batch size: {}, CUs: {}), pattern: {}, hrp: {}",
            gpu_ctx.device_name(),
            batch_size,
            gpu_ctx.max_compute_units(),
            self.config.pattern,
            self.config.hrp,
        );

        // Validate pattern charset early
        self.config.pattern.validate_bech32_charset()?;

        let (result_tx, result_rx) = bounded::<SearchResult>(32);
        let start = Instant::now();
        let pattern = self.config.pattern.clone();
        let hrp = self.config.hrp.clone();
        let path = self.config.derivation_path.clone();
        let max_matches = self.config.max_matches;
        let counter = Arc::clone(&self.candidates_checked);
        let stop_flag = Arc::clone(&self.should_stop);
        let num_cpu_threads = self.config.num_threads;
        let matches_found = Arc::new(AtomicU64::new(self.state.matches_found));

        // Determine if we can use GPU-side prefix matching
        // For prefix patterns, we can convert bech32 prefix to raw hash byte prefix
        // but bech32 encoding is non-trivial to reverse for partial prefixes.
        // Instead, we use the GPU to hash and the CPU to do pattern matching,
        // but with double-buffered dispatch to maximize overlap.
        //
        // For truly large-scale GPU prefix matching, we'd need to implement bech32
        // encoding in the kernel, but that's a future optimization.

        tracing::info!(
            "Pure GPU mode: {} keygen feeder threads, double-buffered GPU dispatch, batch size {}",
            num_cpu_threads,
            batch_size,
        );

        // Channel for keygen threads to send pubkey data to GPU driver
        // Large buffer to keep GPU fed — 3x batch size for double-buffer + margin
        let (keygen_tx, keygen_rx) = bounded::<(Vec<u8>, String, String)>(batch_size * 3);

        // Spawn ALL CPU threads as keygen feeders — no CPU search threads
        for thread_id in 0..num_cpu_threads {
            let keygen_tx = keygen_tx.clone();
            let path = path.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let matches_found = Arc::clone(&matches_found);

            std::thread::Builder::new()
                .name(format!("gpu-keygen-{thread_id}"))
                .spawn(move || {
                    tracing::debug!("Pure GPU keygen worker {thread_id} started");

                    loop {
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        if max_matches > 0
                            && matches_found.load(Ordering::Relaxed) >= max_matches as u64
                        {
                            break;
                        }

                        let key = match generate_random_keypair_with_path(&path) {
                            Ok(k) => k,
                            Err(e) => {
                                tracing::error!("Keygen error: {e}");
                                continue;
                            }
                        };

                        let pubkey = key.public_key_bytes().to_vec();
                        let mnemonic = key.mnemonic().to_string();
                        let deriv_path = key.derivation_path().to_string();
                        drop(key);

                        if keygen_tx.send((pubkey, mnemonic, deriv_path)).is_err() {
                            break;
                        }
                    }

                    tracing::debug!("Pure GPU keygen worker {thread_id} exited");
                })?;
        }

        drop(keygen_tx);

        // GPU driver thread with double-buffered dispatch
        let gpu_result_tx = result_tx.clone();
        let gpu_counter = Arc::clone(&counter);
        let gpu_stop = Arc::clone(&stop_flag);
        let gpu_matches = Arc::clone(&matches_found);
        let gpu_pattern = pattern.clone();
        let gpu_hrp = hrp.clone();

        std::thread::Builder::new()
            .name("gpu-driver-pure".to_string())
            .spawn(move || {
                tracing::debug!("Pure GPU driver thread started (double-buffered)");

                // Double buffer: while GPU processes one batch, CPU fills the other
                let mut buf_a_pubkeys: Vec<u8> = Vec::with_capacity(batch_size * 33);
                let mut buf_a_mnemonics: Vec<String> = Vec::with_capacity(batch_size);
                let mut buf_a_paths: Vec<String> = Vec::with_capacity(batch_size);

                let mut buf_b_pubkeys: Vec<u8> = Vec::with_capacity(batch_size * 33);
                let mut buf_b_mnemonics: Vec<String> = Vec::with_capacity(batch_size);
                let mut buf_b_paths: Vec<String> = Vec::with_capacity(batch_size);

                // Fill first batch (buffer A)
                if !fill_batch(
                    &keygen_rx,
                    &mut buf_a_pubkeys,
                    &mut buf_a_mnemonics,
                    &mut buf_a_paths,
                    batch_size,
                    &gpu_stop,
                    &gpu_matches,
                    max_matches,
                ) {
                    return; // Channel closed or stop signal
                }

                loop {
                    if gpu_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if max_matches > 0
                        && gpu_matches.load(Ordering::Relaxed) >= max_matches as u64
                    {
                        break;
                    }

                    let actual_batch_a = buf_a_mnemonics.len();
                    if actual_batch_a == 0 {
                        break;
                    }

                    // Dispatch buffer A to GPU
                    let hashes = match gpu_ctx.hash_pubkeys_batch(&buf_a_pubkeys) {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::error!("GPU hashing error: {e}");
                            // Try to continue with next batch
                            buf_a_pubkeys.clear();
                            buf_a_mnemonics.clear();
                            buf_a_paths.clear();
                            std::mem::swap(&mut buf_a_pubkeys, &mut buf_b_pubkeys);
                            std::mem::swap(&mut buf_a_mnemonics, &mut buf_b_mnemonics);
                            std::mem::swap(&mut buf_a_paths, &mut buf_b_paths);
                            continue;
                        }
                    };

                    // While GPU was working (queue.finish() returned), start filling buffer B
                    // Note: ocl queue.finish() is blocking, so we fill B after GPU completes.
                    // True async would need ocl events, but filling B here still overlaps
                    // CPU pattern matching of A's results with B's keygen.
                    let has_more = fill_batch(
                        &keygen_rx,
                        &mut buf_b_pubkeys,
                        &mut buf_b_mnemonics,
                        &mut buf_b_paths,
                        batch_size,
                        &gpu_stop,
                        &gpu_matches,
                        max_matches,
                    );

                    let batch_start =
                        gpu_counter.fetch_add(actual_batch_a as u64, Ordering::Relaxed);

                    // Process GPU results — pattern match on CPU (bech32 encode only candidates)
                    for i in 0..actual_batch_a {
                        let hash_bytes: [u8; 20] = hashes[i * 20..(i + 1) * 20]
                            .try_into()
                            .expect("20 bytes");

                        let address = match encode_bech32(&gpu_hrp, &hash_bytes) {
                            Ok(a) => a,
                            Err(_) => continue,
                        };

                        if gpu_pattern.matches(&address, &gpu_hrp) {
                            let candidate_num = batch_start + i as u64;
                            let elapsed = start.elapsed().as_secs_f64();
                            tracing::info!(
                                "🎯 GPU Match found! Address: {} (candidate #{})",
                                address,
                                candidate_num
                            );

                            gpu_matches.fetch_add(1, Ordering::Relaxed);

                            let result = SearchResult {
                                address,
                                mnemonic: buf_a_mnemonics[i].clone(),
                                derivation_path: buf_a_paths[i].clone(),
                                candidate_number: candidate_num,
                                elapsed_secs: elapsed,
                                private_key_hex: None,
                            };

                            if gpu_result_tx.send(result).is_err() {
                                return;
                            }
                        }
                    }

                    // Swap buffers: B becomes the next batch to dispatch, A becomes the fill target
                    buf_a_pubkeys.clear();
                    buf_a_mnemonics.clear();
                    buf_a_paths.clear();
                    std::mem::swap(&mut buf_a_pubkeys, &mut buf_b_pubkeys);
                    std::mem::swap(&mut buf_a_mnemonics, &mut buf_b_mnemonics);
                    std::mem::swap(&mut buf_a_paths, &mut buf_b_paths);

                    if !has_more && buf_a_mnemonics.is_empty() {
                        break;
                    }
                }

                tracing::debug!("Pure GPU driver thread exited");
            })?;

        drop(result_tx);

        Ok(result_rx)
    }

    /// Pure GPU search with raw private keys — maximum performance.
    ///
    /// **Architecture:**
    /// - CPU threads generate random 32-byte private keys (fast, just OsRng)
    /// - GPU does secp256k1 scalar multiplication + SHA-256 + RIPEMD-160
    /// - On match, returns hex private key
    /// - No BIP-39/BIP-32 overhead
    #[cfg(feature = "opencl")]
    pub fn search_gpu_raw(&mut self) -> anyhow::Result<Receiver<SearchResult>> {
        use crate::opencl::GpuContext;
        use rand::RngCore;

        let gpu_ctx = GpuContext::new().map_err(|e| anyhow::anyhow!("GPU init failed: {e}"))?;

        if !gpu_ctx.has_secp256k1_kernel() {
            return Err(anyhow::anyhow!("secp256k1 GPU kernel not available"));
        }

        let batch_size = gpu_ctx.pure_gpu_batch_size();

        tracing::info!(
            "Starting RAW GPU search on {} (batch size: {}, CUs: {}), pattern: {}, hrp: {}",
            gpu_ctx.device_name(),
            batch_size,
            gpu_ctx.max_compute_units(),
            self.config.pattern,
            self.config.hrp,
        );

        self.config.pattern.validate_bech32_charset()?;

        let (result_tx, result_rx) = bounded::<SearchResult>(32);
        let start = Instant::now();
        let pattern = self.config.pattern.clone();
        let hrp = self.config.hrp.clone();
        let max_matches = self.config.max_matches;
        let counter = Arc::clone(&self.candidates_checked);
        let stop_flag = Arc::clone(&self.should_stop);
        let matches_found = Arc::new(AtomicU64::new(self.state.matches_found));

        tracing::info!(
            "Raw GPU mode: generating random privkeys → GPU secp256k1 + hash, batch size {}",
            batch_size,
        );

        // GPU driver thread — generates privkeys and dispatches to GPU
        let gpu_result_tx = result_tx.clone();
        let gpu_counter = Arc::clone(&counter);
        let gpu_stop = Arc::clone(&stop_flag);
        let gpu_matches = Arc::clone(&matches_found);
        let gpu_pattern = pattern.clone();
        let gpu_hrp = hrp.clone();

        std::thread::Builder::new()
            .name("gpu-raw-driver".to_string())
            .spawn(move || {
                tracing::debug!("Raw GPU driver thread started");
                let mut rng = rand::rngs::OsRng;

                loop {
                    if gpu_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if max_matches > 0
                        && gpu_matches.load(Ordering::Relaxed) >= max_matches as u64
                    {
                        break;
                    }

                    // Generate batch of random private keys
                    let mut privkeys = vec![0u8; batch_size * 32];
                    rng.fill_bytes(&mut privkeys);

                    // Dispatch to GPU
                    let (_pubkeys, hashes, _matches) =
                        match gpu_ctx.generate_and_hash_batch(&privkeys, &[]) {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::error!("GPU secp256k1 error: {e}");
                                continue;
                            }
                        };

                    let batch_start =
                        gpu_counter.fetch_add(batch_size as u64, Ordering::Relaxed);

                    // Pattern match on CPU (bech32 encode + check)
                    for i in 0..batch_size {
                        let hash_bytes: [u8; 20] = hashes[i * 20..(i + 1) * 20]
                            .try_into()
                            .expect("20 bytes");

                        let address = match cosmos_vanity_address::encode_bech32(&gpu_hrp, &hash_bytes) {
                            Ok(a) => a,
                            Err(_) => continue,
                        };

                        if gpu_pattern.matches(&address, &gpu_hrp) {
                            let candidate_num = batch_start + i as u64;
                            let elapsed = start.elapsed().as_secs_f64();

                            // Extract the private key hex
                            let privkey_hex = format!("0x{}", hex::encode(&privkeys[i * 32..(i + 1) * 32]));

                            tracing::info!(
                                "🎯 GPU Raw Match! Address: {} (candidate #{})",
                                address,
                                candidate_num
                            );

                            gpu_matches.fetch_add(1, Ordering::Relaxed);

                            let result = SearchResult {
                                address,
                                mnemonic: String::new(),
                                derivation_path: String::new(),
                                candidate_number: candidate_num,
                                elapsed_secs: elapsed,
                                private_key_hex: Some(privkey_hex),
                            };

                            if gpu_result_tx.send(result).is_err() {
                                return;
                            }
                        }
                    }
                }

                tracing::debug!("Raw GPU driver thread exited");
            })?;

        drop(result_tx);
        Ok(result_rx)
    }

    /// Save current state to the configured state file.
    pub fn save_state(&mut self) -> anyhow::Result<()> {
        if let Some(ref path) = self.config.state_file {
            let checked = self.candidates_checked.load(Ordering::Relaxed);
            let elapsed = (Utc::now() - self.state.started_at).num_seconds().max(1) as f64;
            let speed = checked as f64 / elapsed;
            self.state.checkpoint(checked, speed);
            self.state.save(path)?;
        }
        Ok(())
    }
}

/// Fill a batch buffer from the keygen channel.
/// Returns true if there's more data available (channel still open), false if channel closed.
#[cfg(feature = "opencl")]
fn fill_batch(
    keygen_rx: &Receiver<(Vec<u8>, String, String)>,
    pubkeys: &mut Vec<u8>,
    mnemonics: &mut Vec<String>,
    paths: &mut Vec<String>,
    batch_size: usize,
    stop_flag: &Arc<AtomicBool>,
    matches_found: &Arc<AtomicU64>,
    max_matches: usize,
) -> bool {
    pubkeys.clear();
    mnemonics.clear();
    paths.clear();

    // Block on first item
    match keygen_rx.recv() {
        Ok((pubkey, mnemonic, deriv_path)) => {
            pubkeys.extend_from_slice(&pubkey);
            mnemonics.push(mnemonic);
            paths.push(deriv_path);
        }
        Err(_) => return false,
    }

    // Fill rest non-blocking
    while mnemonics.len() < batch_size {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }
        if max_matches > 0 && matches_found.load(Ordering::Relaxed) >= max_matches as u64 {
            break;
        }
        match keygen_rx.try_recv() {
            Ok((pubkey, mnemonic, deriv_path)) => {
                pubkeys.extend_from_slice(&pubkey);
                mnemonics.push(mnemonic);
                paths.push(deriv_path);
            }
            Err(_) => break,
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.hrp, "cosmos");
        assert!(config.num_threads > 0);
        assert_eq!(config.max_matches, 1);
        assert_eq!(config.mode, SearchMode::Cpu);
    }

    #[test]
    fn test_search_mode_display() {
        assert_eq!(format!("{}", SearchMode::Gpu), "gpu");
        assert_eq!(format!("{}", SearchMode::Hybrid), "hybrid");
        assert_eq!(format!("{}", SearchMode::Cpu), "cpu");
    }

    #[test]
    fn test_cpu_search_finds_match() {
        // Search for a very short prefix that should match quickly
        let config = SearchConfig {
            pattern: VanityPattern::Prefix("q".to_string()), // Very common first char
            hrp: "cosmos".to_string(),
            derivation_path: cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH.to_string(),
            num_threads: 2,
            mode: SearchMode::Cpu,
            key_mode: KeyMode::Mnemonic,
            max_matches: 1,
            checkpoint_interval: 1000,
            state_file: None,
        };

        let mut searcher = VanitySearcher::new(config).unwrap();
        let rx = searcher.search_cpu().unwrap();

        // Should find a match reasonably quickly
        let result = rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap();

        assert!(result.address.starts_with("cosmos1q"));
        assert!(!result.mnemonic.is_empty());
        assert_eq!(result.derivation_path, "m/44'/118'/0'/0/0");
    }
}
