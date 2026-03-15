//! Core vanity address search engine — CPU fallback and GPU dispatch.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use crossbeam_channel::{bounded, Receiver};

use cosmos_vanity_address::{pubkey_to_bech32, VanityPattern};
use cosmos_vanity_keyderiv::generate_random_keypair_with_path;

use crate::state::SearchState;

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

    /// Whether to attempt GPU acceleration
    pub use_gpu: bool,

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
            use_gpu: false,
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

    /// The mnemonic that produces this address
    pub mnemonic: String,

    /// The derivation path used
    pub derivation_path: String,

    /// How many candidates were checked before this match
    pub candidate_number: u64,

    /// Time taken to find this match
    pub elapsed_secs: f64,
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
                    config.use_gpu,
                )
            }
        } else {
            SearchState::new(
                &config.pattern.to_string(),
                &config.hrp,
                &config.derivation_path,
                config.use_gpu,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.hrp, "cosmos");
        assert!(config.num_threads > 0);
        assert_eq!(config.max_matches, 1);
    }

    #[test]
    fn test_cpu_search_finds_match() {
        // Search for a very short prefix that should match quickly
        let config = SearchConfig {
            pattern: VanityPattern::Prefix("q".to_string()), // Very common first char
            hrp: "cosmos".to_string(),
            derivation_path: cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH.to_string(),
            num_threads: 2,
            use_gpu: false,
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
