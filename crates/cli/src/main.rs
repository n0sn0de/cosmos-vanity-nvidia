//! # cosmos-vanity CLI
//!
//! Command-line interface for Cosmos vanity address generation
//! with AMD GPU acceleration via ROCm/OpenCL.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use tracing_subscriber::{fmt, EnvFilter};

use cosmos_vanity_address::VanityPattern;
use cosmos_vanity_gpu::{SearchConfig, VanitySearcher};
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

        /// Use GPU acceleration (requires OpenCL/ROCm)
        #[arg(long)]
        gpu: bool,

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
            gpu,
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

            let config = SearchConfig {
                pattern: vanity_pattern.clone(),
                hrp: hrp.clone(),
                derivation_path: path,
                num_threads: threads.unwrap_or_else(num_cpus::get),
                use_gpu: gpu,
                max_matches,
                checkpoint_interval,
                state_file,
            };

            println!("🔍 Cosmos Vanity Address Search");
            println!("   Pattern:  {vanity_pattern}");
            println!("   HRP:      {hrp}");
            println!("   Threads:  {}", config.num_threads);
            println!("   GPU:      {gpu}");
            println!("   Max hits: {max_matches}");
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

            let rx = searcher.search_cpu()?;

            // Progress update loop in a separate thread
            let pb_clone = pb.clone();
            let counter_clone = counter.clone();
            let stop_for_progress = searcher.stop_flag();
            std::thread::spawn(move || {
                loop {
                    if stop_for_progress.load(Ordering::Relaxed) {
                        break;
                    }
                    let checked = counter_clone.load(Ordering::Relaxed);
                    pb_clone.set_message(format!("{checked} candidates checked"));
                    std::thread::sleep(Duration::from_millis(200));
                }
            });

            let mut found = 0;
            for result in rx.iter() {
                found += 1;
                pb.suspend(|| {
                    // Verify the match
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

fn ctrlc_handler(stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n⚡ Interrupted — saving state and exiting...");
        stop_flag.store(true, Ordering::Relaxed);
    });
}
