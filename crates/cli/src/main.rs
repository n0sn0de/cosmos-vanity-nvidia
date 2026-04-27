//! # cosmos-vanity CLI
//!
//! Command-line interface for Cosmos vanity address generation
//! in the NVIDIA/CUDA-focused repo split.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use tracing_subscriber::{fmt, EnvFilter};

use cosmos_vanity_address::VanityPattern;
use cosmos_vanity_gpu::{GpuApi, KeyMode, SearchConfig, SearchMode, VanitySearcher};
use cosmos_vanity_verify::verify_match;

/// Cosmos Vanity Address Generator — GPU accelerated
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

    /// Print mnemonic/private key secrets to stdout (unsafe; redacted by default)
    #[arg(long, global = true)]
    unsafe_print_secrets: bool,
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

        /// GPU backend API: auto (prefer CUDA, then OpenCL), opencl, or cuda
        #[arg(long, default_value = "auto")]
        gpu_api: GpuApiArg,

        /// Key mode: raw (fast, privkey only) or mnemonic (BIP-39 compatible)
        #[arg(short = 'k', long, default_value = "raw")]
        key_mode: KeyModeArg,

        /// Mnemonic word count: 12 (128-bit, Keplr default) or 24 (256-bit)
        #[arg(short = 'w', long, default_value = "12", value_parser = parse_mnemonic_words)]
        words: u8,

        /// Maximum number of matches to find
        #[arg(short = 'n', long, default_value = "1")]
        max_matches: usize,

        /// State file for resuming searches
        #[arg(long)]
        state_file: Option<PathBuf>,

        /// Write secret material to a new JSON Lines file with restrictive permissions
        #[arg(long)]
        secret_output_file: Option<PathBuf>,

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
        /// The mnemonic phrase (unsafe: leaks via argv/history; prefer --mnemonic-file)
        #[arg(short, long, hide = true, conflicts_with = "mnemonic_file")]
        mnemonic: Option<String>,

        /// Read the mnemonic phrase from a file instead of argv
        #[arg(long)]
        mnemonic_file: Option<PathBuf>,

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

        /// Write secret material to a new JSON file with restrictive permissions
        #[arg(long)]
        secret_output_file: Option<PathBuf>,
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
    #[cfg(any(feature = "opencl", feature = "cuda"))]
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
enum GpuApiArg {
    Auto,
    Opencl,
    Cuda,
}

impl GpuApiArg {
    fn to_gpu_api(&self) -> GpuApi {
        match self {
            GpuApiArg::Auto => GpuApi::Auto,
            GpuApiArg::Opencl => GpuApi::OpenCl,
            GpuApiArg::Cuda => GpuApi::Cuda,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

fn parse_mnemonic_words(value: &str) -> std::result::Result<u8, String> {
    match value.parse::<u8>() {
        Ok(12) => Ok(12),
        Ok(24) => Ok(24),
        Ok(other) => Err(format!(
            "invalid mnemonic word count: {other} (expected 12 or 24)"
        )),
        Err(e) => Err(format!("invalid mnemonic word count: {e}")),
    }
}

struct SecretSink {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl SecretSink {
    fn create(path: Option<&Path>) -> Result<Option<Self>> {
        let Some(path) = path else {
            return Ok(None);
        };

        let file = create_secret_file(path)?;
        Ok(Some(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
        }))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write_json_line(&mut self, value: &serde_json::Value) -> Result<()> {
        serde_json::to_writer(&mut self.writer, value)
            .context("failed to serialize secret output")?;
        self.writer
            .write_all(b"\n")
            .context("failed to write secret output")?;
        self.writer
            .flush()
            .context("failed to flush secret output")?;
        Ok(())
    }
}

fn create_secret_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    options.open(path).with_context(|| {
        format!(
            "failed to create secret output file {} (refusing to overwrite existing files)",
            path.display()
        )
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cli.log_level));

    fmt().with_env_filter(filter).with_target(false).init();

    match cli.command {
        Commands::Search {
            pattern,
            match_type,
            hrp,
            path,
            threads,
            mode,
            gpu_api,
            key_mode,
            max_matches,
            state_file,
            checkpoint_interval,
            secret_output_file,
            words,
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

            let gpu_api = gpu_api.to_gpu_api();
            #[cfg(any(feature = "opencl", feature = "cuda"))]
            let requested_path_is_default = path == cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH;

            // Determine effective mode — fall back from GPU/hybrid to CPU if no compatible GPU backend is available.
            let effective_mode = resolve_mode(&mode, gpu_api);

            let effective_key_mode = key_mode.to_key_mode();

            if effective_key_mode == KeyMode::Raw && effective_mode != SearchMode::Gpu {
                anyhow::bail!(
                    "raw key mode requires pure GPU search with a working raw GPU backend; \
                     CPU, hybrid, and GPU-fallback paths do not return raw private keys. \
                     Use '-m gpu -k raw' on a supported GPU backend or switch to '-k mnemonic'."
                );
            }

            let config = SearchConfig {
                pattern: vanity_pattern.clone(),
                hrp: hrp.clone(),
                derivation_path: path,
                num_threads: threads.unwrap_or_else(num_cpus::get),
                mode: effective_mode,
                key_mode: effective_key_mode,
                gpu_api,
                max_matches,
                checkpoint_interval,
                mnemonic_words: words,
                state_file,
            };
            let mut secret_sink = SecretSink::create(secret_output_file.as_deref())?;

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
            if effective_mode != SearchMode::Cpu {
                println!("   GPU API:  {gpu_api}");
            }
            if effective_key_mode == KeyMode::Mnemonic {
                println!("   Words:    {words}");
            }
            println!("   Max hits: {max_matches}");
            if let Some(sink) = &secret_sink {
                println!("   Secret file: {}", sink.path().display());
            }
            println!(
                "   Difficulty: ~{} avg candidates ({})",
                format_number(expected_candidates),
                difficulty_label
            );
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
                    #[cfg(any(feature = "opencl", feature = "cuda"))]
                    {
                        if effective_key_mode == KeyMode::Raw {
                            match searcher.search_gpu_raw() {
                                Ok(rx) => rx,
                                Err(e) => {
                                    anyhow::bail!(
                                        "GPU raw mode failed ({e}); raw key mode cannot safely \
                                         fall back to mnemonic or CPU search because those paths \
                                         do not return raw private keys"
                                    );
                                }
                            }
                        } else if !requested_path_is_default {
                            eprintln!(
                                "⚠️  GPU mnemonic kernel only supports the default Cosmos derivation path; \
                                 using CPU-derived GPU hashing for the requested custom path"
                            );
                            match searcher.search_gpu_pure() {
                                Ok(rx) => rx,
                                Err(e) => {
                                    eprintln!("⚠️  GPU init failed ({e}), falling back to CPU");
                                    searcher.search_cpu()?
                                }
                            }
                        } else {
                            // Try GPU mnemonic pipeline first (PBKDF2+BIP32+secp256k1 on GPU)
                            match searcher.search_gpu_mnemonic() {
                                Ok(rx) => rx,
                                Err(e) => {
                                    eprintln!("⚠️  GPU mnemonic pipeline failed ({e}), falling back to CPU mnemonic GPU mode");
                                    match searcher.search_gpu_pure() {
                                        Ok(rx) => rx,
                                        Err(e2) => {
                                            eprintln!(
                                                "⚠️  GPU init failed ({e2}), falling back to CPU"
                                            );
                                            searcher.search_cpu()?
                                        }
                                    }
                                }
                            }
                        }
                    }
                    #[cfg(not(any(feature = "opencl", feature = "cuda")))]
                    {
                        // Should not reach here due to resolve_mode, but just in case
                        searcher.search_cpu()?
                    }
                }
                SearchMode::Hybrid => {
                    #[cfg(any(feature = "opencl", feature = "cuda"))]
                    {
                        match searcher.search_hybrid() {
                            Ok(rx) => rx,
                            Err(e) => {
                                eprintln!("⚠️  GPU init failed ({e}), falling back to CPU");
                                searcher.search_cpu()?
                            }
                        }
                    }
                    #[cfg(not(any(feature = "opencl", feature = "cuda")))]
                    {
                        searcher.search_cpu()?
                    }
                }
                SearchMode::Cpu => searcher.search_cpu()?,
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
                    let overall_rate = if elapsed > 0.0 {
                        checked as f64 / elapsed
                    } else {
                        0.0
                    };

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
                let mut accepted = false;
                let mut fatal_error: Option<anyhow::Error> = None;

                pb.suspend(|| {
                    if effective_key_mode == KeyMode::Raw {
                        let Some(privkey_hex) = result.private_key_hex.as_deref() else {
                            eprintln!("⚠️  Match skipped: raw mode result did not include a private key.");
                            return;
                        };

                        // Raw key mode — verify with privkey before counting or printing.
                        let verified = cosmos_vanity_verify::verify_privkey_address(
                            privkey_hex,
                            &hrp,
                            &result.address,
                        )
                        .unwrap_or(false);

                        if !verified {
                            eprintln!("⚠️  Match skipped: private key failed address verification.");
                            return;
                        }

                        let secret_json = serde_json::json!({
                            "type": "search_match",
                            "address": &result.address,
                            "private_key": privkey_hex,
                            "candidate_number": result.candidate_number,
                            "elapsed_secs": result.elapsed_secs,
                            "verified": true,
                            "key_mode": "raw",
                        });
                        if let Some(sink) = &mut secret_sink {
                            if let Err(e) = sink.write_json_line(&secret_json) {
                                fatal_error = Some(e);
                                return;
                            }
                        }

                        found += 1;
                        accepted = true;

                        match cli.format {
                            OutputFormat::Text => {
                                println!();
                                println!("🎯 Match #{found} found!");
                                println!("   Address:     {}", result.address);
                                if cli.unsafe_print_secrets {
                                    println!("   Private Key: {} (KEEP SECRET!)", privkey_hex);
                                } else {
                                    println!(
                                        "   Private Key: <redacted; rerun with --unsafe-print-secrets to display>"
                                    );
                                }
                                println!("   Candidate:   #{}", result.candidate_number);
                                println!("   Time:        {:.2}s", result.elapsed_secs);
                                println!("   Verified:    ✅");
                                println!();
                                if cli.unsafe_print_secrets {
                                    println!("   ⚠️  SECURITY: Store your private key securely. Anyone with this key controls the wallet.");
                                } else {
                                    println!("   🔒 Secret output is redacted by default.");
                                }
                                println!();
                            }
                            OutputFormat::Json => {
                                let json = if cli.unsafe_print_secrets {
                                    serde_json::json!({
                                        "address": result.address,
                                        "private_key": privkey_hex,
                                        "candidate_number": result.candidate_number,
                                        "elapsed_secs": result.elapsed_secs,
                                        "verified": true,
                                        "key_mode": "raw",
                                        "secret_redacted": false,
                                    })
                                } else {
                                    serde_json::json!({
                                        "address": result.address,
                                        "candidate_number": result.candidate_number,
                                        "elapsed_secs": result.elapsed_secs,
                                        "verified": true,
                                        "key_mode": "raw",
                                        "secret_redacted": true,
                                    })
                                };
                                println!("{}", serde_json::to_string_pretty(&json).unwrap());
                            }
                        }
                    } else {
                        // Mnemonic mode — verify with mnemonic before counting or printing.
                        match verify_match(
                            &result.mnemonic,
                            &result.derivation_path,
                            &hrp,
                            &result.address,
                            &vanity_pattern,
                        ) {
                            Ok(verification) if verification.verified => {
                                let secret_json = serde_json::json!({
                                    "type": "search_match",
                                    "address": &result.address,
                                    "mnemonic": &result.mnemonic,
                                    "derivation_path": &result.derivation_path,
                                    "candidate_number": result.candidate_number,
                                    "elapsed_secs": result.elapsed_secs,
                                    "verified": true,
                                    "key_mode": "mnemonic",
                                });
                                if let Some(sink) = &mut secret_sink {
                                    if let Err(e) = sink.write_json_line(&secret_json) {
                                        fatal_error = Some(e);
                                        return;
                                    }
                                }

                                found += 1;
                                accepted = true;

                                match cli.format {
                                    OutputFormat::Text => {
                                        println!();
                                        println!("🎯 Match #{found} found!");
                                        println!("   Address:    {}", result.address);
                                        if cli.unsafe_print_secrets {
                                            println!("   Mnemonic:   {}", result.mnemonic);
                                        } else {
                                            println!(
                                                "   Mnemonic:   <redacted; rerun with --unsafe-print-secrets to display>"
                                            );
                                        }
                                        println!("   Path:       {}", result.derivation_path);
                                        println!("   Candidate:  #{}", result.candidate_number);
                                        println!("   Time:       {:.2}s", result.elapsed_secs);
                                        println!("   Verified:   ✅");
                                        if !cli.unsafe_print_secrets {
                                            println!("   🔒 Secret output is redacted by default.");
                                        }
                                        println!();
                                    }
                                    OutputFormat::Json => {
                                        let json = if cli.unsafe_print_secrets {
                                            serde_json::json!({
                                                "address": result.address,
                                                "mnemonic": result.mnemonic,
                                                "derivation_path": result.derivation_path,
                                                "candidate_number": result.candidate_number,
                                                "elapsed_secs": result.elapsed_secs,
                                                "verified": true,
                                                "key_mode": "mnemonic",
                                                "secret_redacted": false,
                                            })
                                        } else {
                                            serde_json::json!({
                                                "address": result.address,
                                                "derivation_path": result.derivation_path,
                                                "candidate_number": result.candidate_number,
                                                "elapsed_secs": result.elapsed_secs,
                                                "verified": true,
                                                "key_mode": "mnemonic",
                                                "secret_redacted": true,
                                            })
                                        };
                                        println!("{}", serde_json::to_string_pretty(&json).unwrap());
                                    }
                                }
                            }
                            Ok(verification) => {
                                eprintln!(
                                    "⚠️  Match skipped: {}",
                                    verification.error.unwrap_or_default()
                                );
                            }
                            Err(e) => {
                                eprintln!("❌ Verification error: {e}");
                            }
                        }
                    }
                });

                if let Some(e) = fatal_error {
                    searcher.stop();
                    return Err(e);
                }

                if accepted && max_matches > 0 && found >= max_matches {
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
            mnemonic_file,
            address,
            hrp,
            path,
        } => {
            let mnemonic = match (mnemonic, mnemonic_file) {
                (Some(mnemonic), None) => {
                    eprintln!(
                        "⚠️  --mnemonic exposes wallet secrets via argv/history; prefer --mnemonic-file."
                    );
                    mnemonic
                }
                (None, Some(path)) => std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read mnemonic file {}", path.display()))?
                    .trim()
                    .to_string(),
                (None, None) => {
                    anyhow::bail!("provide --mnemonic-file <path> (preferred) or legacy --mnemonic")
                }
                (Some(_), Some(_)) => unreachable!("clap enforces conflicts_with"),
            };

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

        Commands::Generate {
            hrp,
            path,
            secret_output_file,
        } => {
            let key = cosmos_vanity_keyderiv::generate_random_keypair_with_path(&path)?;
            let address = cosmos_vanity_address::pubkey_to_bech32(key.public_key_bytes(), &hrp)?;
            let mut secret_sink = SecretSink::create(secret_output_file.as_deref())?;
            let secret_json = serde_json::json!({
                "type": "generate",
                "address": &address,
                "mnemonic": key.mnemonic(),
                "derivation_path": key.derivation_path(),
                "key_mode": "mnemonic",
            });
            if let Some(sink) = &mut secret_sink {
                sink.write_json_line(&secret_json)?;
            }

            match cli.format {
                OutputFormat::Text => {
                    println!("Address:  {address}");
                    if cli.unsafe_print_secrets {
                        println!("Mnemonic: {}", key.mnemonic());
                    } else {
                        println!(
                            "Mnemonic: <redacted; rerun with --unsafe-print-secrets to display>"
                        );
                    }
                    println!("Path:     {}", key.derivation_path());
                    if let Some(sink) = &secret_sink {
                        println!("Secret file: {}", sink.path().display());
                    }
                    if !cli.unsafe_print_secrets {
                        println!("🔒 Secret output is redacted by default.");
                    }
                }
                OutputFormat::Json => {
                    let json = if cli.unsafe_print_secrets {
                        serde_json::json!({
                            "address": address,
                            "mnemonic": key.mnemonic(),
                            "derivation_path": key.derivation_path(),
                            "secret_output_file": secret_sink.as_ref().map(|sink| sink.path().display().to_string()),
                            "secret_redacted": false,
                        })
                    } else {
                        serde_json::json!({
                            "address": address,
                            "derivation_path": key.derivation_path(),
                            "secret_output_file": secret_sink.as_ref().map(|sink| sink.path().display().to_string()),
                            "secret_redacted": true,
                        })
                    };
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                }
            }
        }
    }

    Ok(())
}

/// Resolve the requested search mode to an effective mode.
/// Falls back from GPU/hybrid to CPU if no compatible GPU backend is available.
fn resolve_mode(requested: &SearchModeArg, _gpu_api: GpuApi) -> SearchMode {
    match requested {
        SearchModeArg::Cpu => SearchMode::Cpu,
        SearchModeArg::Gpu | SearchModeArg::Hybrid => {
            #[cfg(any(feature = "opencl", feature = "cuda"))]
            {
                if cosmos_vanity_gpu::is_gpu_api_available(_gpu_api) {
                    requested.to_search_mode()
                } else {
                    eprintln!(
                        "⚠️  No compatible GPU backend found for API '{}', falling back to CPU mode (requested: {})",
                        _gpu_api,
                        requested.label()
                    );
                    SearchMode::Cpu
                }
            }
            #[cfg(not(any(feature = "opencl", feature = "cuda")))]
            {
                eprintln!(
                    "⚠️  GPU support not compiled in, falling back to CPU mode (requested: {})",
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
        VanityPattern::Regex(_) => (0, "regex — cannot estimate".to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mnemonic_words_accepts_only_supported_counts() {
        assert_eq!(parse_mnemonic_words("12").unwrap(), 12);
        assert_eq!(parse_mnemonic_words("24").unwrap(), 24);
        assert!(parse_mnemonic_words("13").is_err());
    }

    #[test]
    fn cli_rejects_invalid_word_count() {
        let err = Cli::try_parse_from([
            "cosmos-vanity",
            "search",
            "-p",
            "q",
            "-k",
            "mnemonic",
            "-w",
            "13",
        ])
        .unwrap_err();

        assert!(err.to_string().contains("expected 12 or 24"));
    }

    #[test]
    fn unsafe_print_secrets_is_global() {
        let cli =
            Cli::try_parse_from(["cosmos-vanity", "generate", "--unsafe-print-secrets"]).unwrap();

        assert!(cli.unsafe_print_secrets);
        assert!(matches!(cli.command, Commands::Generate { .. }));
    }

    fn unique_secret_path(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cosmos-vanity-{name}-{}-{nanos}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn secret_sink_writes_json_line_and_refuses_overwrite() {
        let path = unique_secret_path("secret-sink");

        {
            let mut sink = SecretSink::create(Some(&path)).unwrap().unwrap();
            sink.write_json_line(&serde_json::json!({"mnemonic": "test secret"}))
                .unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("test secret"));

        let err = match SecretSink::create(Some(&path)) {
            Err(e) => e,
            Ok(_) => panic!("secret sink overwrote an existing file"),
        };
        assert!(err.to_string().contains("refusing to overwrite"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn generate_accepts_secret_output_file() {
        let cli = Cli::try_parse_from([
            "cosmos-vanity",
            "generate",
            "--secret-output-file",
            "/tmp/secrets.jsonl",
        ])
        .unwrap();

        match cli.command {
            Commands::Generate {
                secret_output_file, ..
            } => {
                assert_eq!(
                    secret_output_file.unwrap(),
                    PathBuf::from("/tmp/secrets.jsonl")
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn verify_accepts_mnemonic_file_without_argv_secret() {
        let cli = Cli::try_parse_from([
            "cosmos-vanity",
            "verify",
            "--mnemonic-file",
            "/tmp/mnemonic.txt",
            "--address",
            "cosmos1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqnrql8a",
        ])
        .unwrap();

        match cli.command {
            Commands::Verify {
                mnemonic,
                mnemonic_file,
                ..
            } => {
                assert!(mnemonic.is_none());
                assert_eq!(mnemonic_file.unwrap(), PathBuf::from("/tmp/mnemonic.txt"));
            }
            _ => panic!("expected verify command"),
        }
    }
}
