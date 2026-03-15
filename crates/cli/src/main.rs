//! # cosmos-vanity CLI
//!
//! Command-line interface for Cosmos vanity address generation
//! with AMD GPU acceleration via ROCm/OpenCL.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use tracing_subscriber::{fmt, EnvFilter};

use cosmos_vanity_address::VanityPattern;
use cosmos_vanity_gpu::{KeyMode, SearchConfig, SearchMode, VanitySearcher};
use cosmos_vanity_verify::verify_match;

/// Cosmos Vanity Address Generator — AMD GPU accelerated
#[derive(Parser, Debug)]
#[command(name = "cosmos-vanity")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info", global = true)]
    log_level: String,

    /// Output format
    #[arg(long, default_value = "text", global = true)]
    format: OutputFormat,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Search for a vanity address
    Search {
        /// Pattern to search for
        #[arg(short, long)]
        pattern: String,

        /// Pattern match type
        #[arg(short = 't', long, default_value = "prefix")]
        match_type: MatchType,

        /// Human-readable part (cosmos, osmo, juno, etc.)
        #[arg(long, default_value = "cosmos")]
        hrp: String,

        /// BIP-44 derivation path
        #[arg(long, default_value = "m/44'/118'/0'/0/0")]
        path: String,

        /// Number of CPU threads (default: all cores)
        #[arg(short = 'j', long)]
        threads: Option<usize>,

        /// Search mode: gpu (pure GPU, default), hybrid (CPU+GPU), cpu (CPU only)
        #[arg(short, long, default_value = "gpu")]
        mode: SearchModeArg,

        /// Key mode: raw (fast, privkey only) or mnemonic (BIP-39 compatible)
        #[arg(short = 'k', long, default_value = "raw")]
        key_mode: KeyModeArg,

        /// Maximum number of matches to find
        #[arg(short = 'n', long, default_value = "1")]
        max_matches: usize,

        /// State file for resuming searches
        #[arg(long)]
        state_file: Option<PathBuf>,

        /// Checkpoint interval (save state every N candidates)
        #[arg(long, default_value = "100000")]
        checkpoint_interval: u64,
    },

    /// Run benchmarks
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: u64,

        /// HRP to use for benchmarking
        #[arg(long, default_value = "cosmos")]
        hrp: String,
    },

    /// Verify a mnemonic produces a specific address
    Verify {
        /// The mnemonic phrase
        #[arg(short, long)]
        mnemonic: String,

        /// Expected address
        #[arg(short, long)]
        address: String,

        /// HRP
        #[arg(long, default_value = "cosmos")]
        hrp: String,

        /// Derivation path
        #[arg(long, default_value = "m/44'/118'/0'/0/0")]
        path: String,
    },

    /// Generate a single random address (for testing)
    Generate {
        /// HRP
        #[arg(long, default_value = "cosmos")]
        hrp: String,

        /// Derivation path
        #[arg(long, default_value = "m/44'/118'/0'/0/0")]
        path: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum MatchType {
    Prefix,
    Suffix,
    Contains,
    Regex,
}

#[derive(Debug, Clone, ValueEnum)]
enum SearchModeArg {
    /// Pure GPU mode — ALL CPU threads feed keys, GPU does all hashing. Maximum performance.
    Gpu,
    /// Hybrid mode — CPU search threads + GPU pipeline in parallel.
    Hybrid,
    /// CPU-only mode — no GPU acceleration.
    Cpu,
}

impl SearchModeArg {
    fn to_search_mode(&self) -> SearchMode {
        match self {
            SearchModeArg::Gpu => SearchMode::Gpu,
            SearchModeArg::Hybrid => SearchMode::Hybrid,
            SearchModeArg::Cpu => SearchMode::Cpu,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            SearchModeArg::Gpu => "gpu (pure GPU)",
            SearchModeArg::Hybrid => "hybrid (CPU+GPU)",
            SearchModeArg::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum KeyModeArg {
    /// Raw mode: fast privkey generation, returns hex private key
    Raw,
    /// Mnemonic mode: BIP-39 compatible, returns 24-word mnemonic
    Mnemonic,
}

impl KeyModeArg {
    fn to_key_mode(&self) -> KeyMode {
        match self {
            KeyModeArg::Raw => KeyMode::Raw,
            KeyModeArg::Mnemonic => KeyMode::Mnemonic,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&cli.log_level));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    match cli.command {
        Commands::Search {
            pattern,
            match_type,
            hrp,
            path,
            threads,
            mode,
            key_mode,
            max_matches,
            state_file,
            checkpoint_interval,
        } => {
            let vanity_pattern = match match_type {
                MatchType::Prefix => VanityPattern::Prefix(pattern),
                MatchType::Suffix => VanityPattern::Suffix(pattern),
                MatchType::Contains => VanityPattern::Contains(pattern),
                MatchType::Regex => VanityPattern::Regex(pattern),
            };

            // Validate pattern charset early
            vanity_pattern
                .validate_bech32_charset()
                .context("Invalid pattern")?;

            // Determine effective mode — fall back from GPU/hybrid to CPU if no OpenCL
            let effective_mode = resolve_mode(&mode);

            let effective_key_mode = key_mode.to_key_mode();

            let config = SearchConfig {
                pattern: vanity_pattern.clone(),
                hrp: hrp.clone(),
                derivation_path: path,
                num_threads: threads.unwrap_or_else(num_cpus::get),
                mode: effective_mode,
                key_mode: effective_key_mode,
                max_matches,
                checkpoint_interval,
                state_file,
            };

            let mode_label = match effective_mode {
                SearchMode::Gpu => "gpu (pure GPU)",
                SearchMode::Hybrid => "hybrid (CPU+GPU)",
                SearchMode::Cpu => "cpu",
            };

            // Calculate difficulty estimate
            let (expected_candidates, difficulty_label) = estimate_difficulty(&vanity_pattern);

            println!("🔍 Cosmos Vanity Address Search");
            println!("   Pattern:  {vanity_pattern}");
            println!("   HRP:      {hrp}");
            println!("   Threads:  {}", config.num_threads);
            println!("   Mode:     {mode_label}");
            println!("   Key mode: {effective_key_mode}");
            println!("   Max hits: {max_matches}");
            println!("   Difficulty: ~{} avg candidates ({})", format_number(expected_candidates), difficulty_label);
            println!();

            let mut searcher = VanitySearcher::new(config)?;

            // Set up Ctrl+C handler
            let stop_flag = searcher.stop_flag();
            ctrlc_handler(stop_flag);

            // Progress bar
            let counter = searcher.candidates_counter();
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [{elapsed_precise}] {msg}")
                    .expect("template"),
            );

            let rx = match effective_mode {
                SearchMode::Gpu => {
                    #[cfg(feature = "opencl")]
                    {
                        if effective_key_mode == KeyMode::Raw {
                            match searcher.search_gpu_raw() {
                                Ok(rx) => rx,
                                Err(e) => {
                                    eprintln!("⚠️  GPU raw mode failed ({e}), falling back to mnemonic GPU mode");
                                    match searcher.search_gpu_pure() {
                                        Ok(rx) => rx,
                                        Err(e2) => {
                                            eprintln!("⚠️  GPU init failed ({e2}), falling back to CPU");
                                            searcher.search_cpu()?
                                        }
                                    }
                                }
                            }
                        } else {
                            match searcher.search_gpu_pure() {
                                Ok(rx) => rx,
                                Err(e) => {
                                    eprintln!("⚠️  GPU init failed ({e}), falling back to CPU");
                                    searcher.search_cpu()?
                                }
                            }
                        }
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        // Should not reach here due to resolve_mode, but just in case
                        searcher.search_cpu()?
                    }
                }
                SearchMode::Hybrid => {
                    #[cfg(feature = "opencl")]
                    {
                        match searcher.search_hybrid() {
                            Ok(rx) => rx,
                            Err(e) => {
                                eprintln!("⚠️  GPU init failed ({e}), falling back to CPU");
                                searcher.search_cpu()?
                            }
                        }
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        searcher.search_cpu()?
                    }
                }
                SearchMode::Cpu => {
                    searcher.search_cpu()?
                }
            };

            // Progress update loop in a separate thread
            let pb_clone = pb.clone();
            let counter_clone = counter.clone();
            let stop_for_progress = searcher.stop_flag();
            let search_start = Instant::now();
            std::thread::spawn(move || {
                let mut last_count: u64 = 0;
                let mut last_time = search_start;
                let mut smoothed_rate: f64 = 0.0;

                // Let it warm up for 1 second before showing ETA
                std::thread::sleep(Duration::from_secs(1));

                loop {
                    if stop_for_progress.load(Ordering::Relaxed) {
                        break;
                    }
                    let checked = counter_clone.load(Ordering::Relaxed);
                    let now = Instant::now();
                    let dt = now.duration_since(last_time).as_secs_f64();

                    if dt > 0.0 {
                        let instant_rate = (checked - last_count) as f64 / dt;
                        // Exponential moving average for smooth display
                        if smoothed_rate == 0.0 {
                            smoothed_rate = instant_rate;
                        } else {
                            smoothed_rate = smoothed_rate * 0.7 + instant_rate * 0.3;
                        }
                    }

                    last_count = checked;
                    last_time = now;

                    let elapsed = search_start.elapsed().as_secs_f64();
                    let overall_rate = if elapsed > 0.0 { checked as f64 / elapsed } else { 0.0 };

                    let eta_str = if smoothed_rate > 0.0 && expected_candidates > checked {
                        let remaining = expected_candidates - checked;
                        let eta_secs = remaining as f64 / smoothed_rate;
                        format_duration(eta_secs)
                    } else if checked >= expected_candidates {
                        "any moment...".to_string()
                    } else {
                        "calculating...".to_string()
                    };

                    let progress_pct = if expected_candidates > 0 {
                        ((checked as f64 / expected_candidates as f64) * 100.0).min(999.9)
                    } else {
                        0.0
                    };

                    pb_clone.set_message(format!(
                        "{} checked | {}/s | {:.1}% of avg | ETA: {}",
                        format_number(checked),
                        format_number(overall_rate as u64),
                        progress_pct,
                        eta_str,
                    ));
                    std::thread::sleep(Duration::from_millis(500));
                }
            });

            let mut found = 0;
            for result in rx.iter() {
                found += 1;
                pb.suspend(|| {
                    if let Some(ref privkey_hex) = result.private_key_hex {
                        // Raw key mode — verify with privkey
                        let verified = cosmos_vanity_verify::verify_privkey_address(
                            privkey_hex, &hrp, &result.address,
                        ).unwrap_or(false);

                        match cli.format {
                            OutputFormat::Text => {
                                println!();
                                println!("🎯 Match #{found} found!");
                                println!("   Address:     {}", result.address);
                                println!("   Private Key: {} (KEEP SECRET!)", privkey_hex);
                                println!("   Candidate:   #{}", result.candidate_number);
                                println!("   Time:        {:.2}s", result.elapsed_secs);
                                println!("   Verified:    {}", if verified { "✅" } else { "❌" });
                                println!();
                                println!("   ⚠️  SECURITY: Store your private key securely. Anyone with this key controls the wallet.");
                                println!();
                            }
                            OutputFormat::Json => {
                                let json = serde_json::json!({
                                    "address": result.address,
                                    "private_key": privkey_hex,
                                    "candidate_number": result.candidate_number,
                                    "elapsed_secs": result.elapsed_secs,
                                    "verified": verified,
                                    "key_mode": "raw",
                                });
                                println!("{}", serde_json::to_string_pretty(&json).unwrap());
                            }
                        }

                        if !verified {
                            eprintln!("⚠️  Match FAILED verification! Private key does not produce expected address.");
                        }
                    } else {
                    // Mnemonic mode — verify with mnemonic
                    match verify_match(
                        &result.mnemonic,
                        &result.derivation_path,
                        &hrp,
                        &result.address,
                        &vanity_pattern,
                    ) {
                        Ok(verification) if verification.verified => {
                            match cli.format {
                                OutputFormat::Text => {
                                    println!();
                                    println!("🎯 Match #{found} found!");
                                    println!("   Address:    {}", result.address);
                                    println!("   Mnemonic:   {}", result.mnemonic);
                                    println!("   Path:       {}", result.derivation_path);
                                    println!("   Candidate:  #{}", result.candidate_number);
                                    println!("   Time:       {:.2}s", result.elapsed_secs);
                                    println!("   Verified:   ✅");
                                    println!();
                                }
                                OutputFormat::Json => {
                                    let json = serde_json::json!({
                                        "address": result.address,
                                        "mnemonic": result.mnemonic,
                                        "derivation_path": result.derivation_path,
                                        "candidate_number": result.candidate_number,
                                        "elapsed_secs": result.elapsed_secs,
                                        "verified": true,
                                        "key_mode": "mnemonic",
                                    });
                                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                                }
                            }
                        }
                        Ok(verification) => {
                            eprintln!(
                                "⚠️  Match FAILED verification: {}",
                                verification.error.unwrap_or_default()
                            );
                        }
                        Err(e) => {
                            eprintln!("❌ Verification error: {e}");
                        }
                    }
                    } // end else (mnemonic mode)
                });

                if max_matches > 0 && found >= max_matches {
                    searcher.stop();
                    break;
                }
            }

            pb.finish_with_message(format!(
                "Done! {} candidates checked, {} matches found",
                counter.load(Ordering::Relaxed),
                found
            ));

            // Save final state
            if let Err(e) = searcher.save_state() {
                eprintln!("Warning: failed to save state: {e}");
            }
        }

        Commands::Bench { iterations, hrp } => {
            println!("📊 Running benchmarks ({iterations} iterations)...");
            println!();

            let full = cosmos_vanity_bench::bench_cpu_throughput(iterations, &hrp);
            println!("Full Pipeline (keygen + address):");
            print!("{full}");

            let keygen = cosmos_vanity_bench::bench_key_derivation(iterations);
            println!("Key Derivation Only:");
            print!("{keygen}");

            let hashing = cosmos_vanity_bench::bench_address_hashing(iterations, &hrp);
            println!("Address Hashing Only:");
            print!("{hashing}");
        }

        Commands::Verify {
            mnemonic,
            address,
            hrp,
            path,
        } => {
            let verified = cosmos_vanity_verify::verify_address(&mnemonic, &path, &hrp, &address)?;

            if verified {
                println!("✅ Verified: mnemonic produces address {address}");
            } else {
                let key = cosmos_vanity_keyderiv::derive_keypair_from_mnemonic(&mnemonic, &path)?;
                let actual = cosmos_vanity_address::pubkey_to_bech32(key.public_key_bytes(), &hrp)?;
                println!("❌ Mismatch!");
                println!("   Expected: {address}");
                println!("   Actual:   {actual}");
                std::process::exit(1);
            }
        }

        Commands::Generate { hrp, path } => {
            let key = cosmos_vanity_keyderiv::generate_random_keypair_with_path(&path)?;
            let address = cosmos_vanity_address::pubkey_to_bech32(key.public_key_bytes(), &hrp)?;

            match cli.format {
                OutputFormat::Text => {
                    println!("Address:  {address}");
                    println!("Mnemonic: {}", key.mnemonic());
                    println!("Path:     {}", key.derivation_path());
                }
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "address": address,
                        "mnemonic": key.mnemonic(),
                        "derivation_path": key.derivation_path(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                }
            }
        }
    }

    Ok(())
}

/// Resolve the requested search mode to an effective mode.
/// Falls back from GPU/hybrid to CPU if OpenCL is not available.
fn resolve_mode(requested: &SearchModeArg) -> SearchMode {
    match requested {
        SearchModeArg::Cpu => SearchMode::Cpu,
        SearchModeArg::Gpu | SearchModeArg::Hybrid => {
            #[cfg(feature = "opencl")]
            {
                if cosmos_vanity_gpu::opencl::is_available() {
                    requested.to_search_mode()
                } else {
                    eprintln!(
                        "⚠️  No OpenCL GPU found, falling back to CPU mode (requested: {})",
                        requested.label()
                    );
                    SearchMode::Cpu
                }
            }
            #[cfg(not(feature = "opencl"))]
            {
                eprintln!(
                    "⚠️  OpenCL support not compiled in, falling back to CPU mode (requested: {})",
                    requested.label()
                );
                SearchMode::Cpu
            }
        }
    }
}

fn ctrlc_handler(stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n⚡ Interrupted — saving state and exiting...");
        stop_flag.store(true, Ordering::Relaxed);
    });
}

/// Estimate the expected number of candidates to check for a pattern.
/// Bech32 has 32 characters, so prefix of length N → 32^N expected candidates.
fn estimate_difficulty(pattern: &VanityPattern) -> (u64, String) {
    match pattern {
        VanityPattern::Prefix(p) => {
            let n = p.len();
            let expected = 32u64.saturating_pow(n as u32);
            let label = match n {
                1 => "trivial — seconds".to_string(),
                2 => "easy — seconds".to_string(),
                3 => "moderate — seconds to minutes".to_string(),
                4 => "hard — minutes".to_string(),
                5 => "very hard — minutes to hours".to_string(),
                6 => "extreme — hours to days".to_string(),
                _ => format!("insane — {n} chars, good luck"),
            };
            (expected, label)
        }
        VanityPattern::Suffix(s) => {
            let n = s.len();
            let expected = 32u64.saturating_pow(n as u32);
            (expected, format!("{n}-char suffix"))
        }
        VanityPattern::Contains(s) => {
            // Contains is easier than prefix — roughly 32^N / address_len
            let n = s.len();
            let expected = 32u64.saturating_pow(n as u32) / 38; // ~38 chars in a cosmos address
            (expected.max(1), format!("{n}-char contains"))
        }
        VanityPattern::Regex(_) => {
            (0, "regex — cannot estimate".to_string())
        }
    }
}

/// Format a large number with comma separators.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format seconds into human-readable duration.
fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        "< 1s".to_string()
    } else if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        let m = (secs / 60.0).floor();
        let s = secs % 60.0;
        format!("{:.0}m {:.0}s", m, s)
    } else if secs < 86400.0 {
        let h = (secs / 3600.0).floor();
        let m = ((secs % 3600.0) / 60.0).floor();
        format!("{:.0}h {:.0}m", h, m)
    } else {
        let d = (secs / 86400.0).floor();
        let h = ((secs % 86400.0) / 3600.0).floor();
        format!("{:.0}d {:.0}h", d, h)
    }
}
