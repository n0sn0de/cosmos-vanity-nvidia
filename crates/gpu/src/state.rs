//! Resumable search state — save/restore progress across sessions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Persistent search state for resuming interrupted searches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchState {
    /// Total candidates checked so far
    pub candidates_checked: u64,

    /// Total matches found
    pub matches_found: u64,

    /// When the search started
    pub started_at: DateTime<Utc>,

    /// Last checkpoint time
    pub last_checkpoint: DateTime<Utc>,

    /// The pattern being searched for
    pub pattern: String,

    /// The HRP being used
    pub hrp: String,

    /// Derivation path
    pub derivation_path: String,

    /// Average speed (candidates/sec)
    pub avg_speed: f64,

    /// Whether GPU acceleration is in use
    pub gpu_enabled: bool,

    /// Results found so far
    pub results: Vec<SerializableResult>,
}

/// A serializable search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableResult {
    pub address: String,
    pub mnemonic: String,
    pub derivation_path: String,
    pub found_at: DateTime<Utc>,
    pub candidate_number: u64,
}

impl SearchState {
    /// Create a new search state.
    pub fn new(pattern: &str, hrp: &str, derivation_path: &str, gpu_enabled: bool) -> Self {
        let now = Utc::now();
        Self {
            candidates_checked: 0,
            matches_found: 0,
            started_at: now,
            last_checkpoint: now,
            pattern: pattern.to_string(),
            hrp: hrp.to_string(),
            derivation_path: derivation_path.to_string(),
            avg_speed: 0.0,
            gpu_enabled,
            results: Vec::new(),
        }
    }

    /// Save state to a JSON file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        tracing::debug!("Saved search state to {}", path.display());
        Ok(())
    }

    /// Load state from a JSON file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let state: Self = serde_json::from_str(&json)?;
        tracing::info!(
            "Resumed search: {} candidates checked, {} matches found",
            state.candidates_checked,
            state.matches_found
        );
        Ok(state)
    }

    /// Update the checkpoint with current stats.
    pub fn checkpoint(&mut self, candidates_checked: u64, speed: f64) {
        self.candidates_checked = candidates_checked;
        self.last_checkpoint = Utc::now();
        self.avg_speed = speed;
    }

    /// Record a match.
    pub fn record_match(&mut self, result: SerializableResult) {
        self.matches_found += 1;
        self.results.push(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_roundtrip() {
        let state = SearchState::new("abc", "cosmos", "m/44'/118'/0'/0/0", false);

        let json = serde_json::to_string_pretty(&state).unwrap();
        let loaded: SearchState = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.pattern, "abc");
        assert_eq!(loaded.hrp, "cosmos");
        assert_eq!(loaded.candidates_checked, 0);
    }

    #[test]
    fn test_checkpoint_update() {
        let mut state = SearchState::new("test", "cosmos", "m/44'/118'/0'/0/0", false);
        state.checkpoint(1000, 500.0);

        assert_eq!(state.candidates_checked, 1000);
        assert_eq!(state.avg_speed, 500.0);
    }

    #[test]
    fn test_record_match() {
        let mut state = SearchState::new("test", "cosmos", "m/44'/118'/0'/0/0", false);
        state.record_match(SerializableResult {
            address: "cosmos1test...".to_string(),
            mnemonic: "word ".repeat(24).trim().to_string(),
            derivation_path: "m/44'/118'/0'/0/0".to_string(),
            found_at: Utc::now(),
            candidate_number: 42,
        });

        assert_eq!(state.matches_found, 1);
        assert_eq!(state.results.len(), 1);
    }
}
