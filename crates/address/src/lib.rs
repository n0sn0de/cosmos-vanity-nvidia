//! # cosmos-vanity-address
//!
//! Cosmos SDK address generation: compressed public key → SHA-256 → RIPEMD-160 → Bech32.
//!
//! Supports configurable Human-Readable Parts (HRPs) for different Cosmos chains.

use bech32::{Bech32, Hrp};
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};

/// Well-known Cosmos chain HRPs.
pub mod hrps {
    pub const COSMOS: &str = "cosmos";
    pub const OSMOSIS: &str = "osmo";
    pub const JUNO: &str = "juno";
    pub const STARGAZE: &str = "stars";
    pub const AKASH: &str = "akash";
    pub const REGEN: &str = "regen";
    pub const SENTINEL: &str = "sent";
    pub const PERSISTENCE: &str = "persistence";
    pub const KAVA: &str = "kava";
    pub const EVMOS: &str = "evmos";
    pub const INJECTIVE: &str = "inj";
    pub const CELESTIA: &str = "celestia";
    pub const DYDX: &str = "dydx";
    pub const NOBLE: &str = "noble";
    pub const STRIDE: &str = "stride";
}

/// Errors from address generation.
#[derive(Debug, thiserror::Error)]
pub enum AddressError {
    #[error("invalid public key length: expected 33 bytes (compressed), got {0}")]
    InvalidPubKeyLength(usize),

    #[error("bech32 encoding error: {0}")]
    Bech32(String),

    #[error("invalid HRP: {0}")]
    InvalidHrp(String),
}

/// Hash a compressed public key to get the 20-byte Cosmos address hash.
///
/// Algorithm: `RIPEMD160(SHA256(pubkey))`
pub fn pubkey_to_address_bytes(compressed_pubkey: &[u8]) -> Result<[u8; 20], AddressError> {
    if compressed_pubkey.len() != 33 {
        return Err(AddressError::InvalidPubKeyLength(compressed_pubkey.len()));
    }

    // SHA-256
    let sha_hash = Sha256::digest(compressed_pubkey);

    // RIPEMD-160
    let ripemd_hash = Ripemd160::digest(&sha_hash);

    let mut result = [0u8; 20];
    result.copy_from_slice(&ripemd_hash);
    Ok(result)
}

/// Encode a 20-byte address hash as a Bech32 address with the given HRP.
pub fn encode_bech32(hrp: &str, address_bytes: &[u8; 20]) -> Result<String, AddressError> {
    let hrp = Hrp::parse(hrp).map_err(|e| AddressError::InvalidHrp(e.to_string()))?;
    let encoded =
        bech32::encode::<Bech32>(hrp, address_bytes).map_err(|e| AddressError::Bech32(e.to_string()))?;
    Ok(encoded)
}

/// Generate a Cosmos Bech32 address from a compressed public key and HRP.
///
/// This is the main entry point: pubkey → SHA256 → RIPEMD160 → Bech32.
pub fn pubkey_to_bech32(compressed_pubkey: &[u8], hrp: &str) -> Result<String, AddressError> {
    let address_bytes = pubkey_to_address_bytes(compressed_pubkey)?;
    encode_bech32(hrp, &address_bytes)
}

/// Check if a Bech32 address matches a given prefix pattern (after the HRP + "1" separator).
///
/// For example, if the address is `cosmos1abc...` and the pattern is `abc`,
/// this checks if the address part starts with `abc`.
pub fn matches_prefix(address: &str, hrp: &str, prefix: &str) -> bool {
    // Bech32 format: {hrp}1{data}
    let separator = format!("{hrp}1");
    if let Some(data_part) = address.strip_prefix(&separator) {
        data_part.starts_with(prefix)
    } else {
        false
    }
}

/// Check if a Bech32 address matches a given suffix pattern.
pub fn matches_suffix(address: &str, suffix: &str) -> bool {
    // Last 6 chars of bech32 are checksum, so we check before that
    // Actually, for vanity purposes, users typically want visual matching
    // including the checksum. But the checksum is deterministic from the data,
    // so matching suffix is valid.
    address.ends_with(suffix)
}

/// Check if a Bech32 address contains a given substring (after the HRP separator).
pub fn matches_contains(address: &str, hrp: &str, substring: &str) -> bool {
    let separator = format!("{hrp}1");
    if let Some(data_part) = address.strip_prefix(&separator) {
        data_part.contains(substring)
    } else {
        false
    }
}

/// A pattern matcher for vanity address searching.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum VanityPattern {
    /// Match the start of the address (after hrp1)
    Prefix(String),
    /// Match the end of the address
    Suffix(String),
    /// Match anywhere in the address (after hrp1)
    Contains(String),
    /// Match with regex-like pattern
    Regex(String),
}

impl VanityPattern {
    /// Check if the given address matches this pattern.
    pub fn matches(&self, address: &str, hrp: &str) -> bool {
        match self {
            VanityPattern::Prefix(p) => matches_prefix(address, hrp, p),
            VanityPattern::Suffix(s) => matches_suffix(address, s),
            VanityPattern::Contains(s) => matches_contains(address, hrp, s),
            VanityPattern::Regex(pattern) => {
                // Basic regex support — compile on each call (caller should cache if hot path)
                if let Ok(re) = regex_lite::Regex::new(pattern) {
                    let separator = format!("{hrp}1");
                    if let Some(data_part) = address.strip_prefix(&separator) {
                        re.is_match(data_part)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Validate that the pattern only contains valid Bech32 characters.
    ///
    /// Bech32 charset: qpzry9x8gf2tvdw0s3jn54khce6mua7l
    pub fn validate_bech32_charset(&self) -> Result<(), AddressError> {
        const BECH32_CHARS: &str = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

        let pattern_str = match self {
            VanityPattern::Prefix(s) | VanityPattern::Suffix(s) | VanityPattern::Contains(s) => s,
            VanityPattern::Regex(_) => return Ok(()), // Skip validation for regex
        };

        for c in pattern_str.chars() {
            if !BECH32_CHARS.contains(c) {
                return Err(AddressError::Bech32(format!(
                    "character '{c}' is not valid in Bech32 addresses. Valid chars: {BECH32_CHARS}"
                )));
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for VanityPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VanityPattern::Prefix(p) => write!(f, "prefix:{p}"),
            VanityPattern::Suffix(s) => write!(f, "suffix:{s}"),
            VanityPattern::Contains(s) => write!(f, "contains:{s}"),
            VanityPattern::Regex(r) => write!(f, "regex:{r}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known test vector: "abandon" x23 + "art" mnemonic on m/44'/118'/0'/0/0
    // We test with a known pubkey instead.
    const TEST_PUBKEY_HEX: &str =
        "02394bc53633366a2ab9b5d4a7b6a9cfd9f11d576e45e1e2049a2e397b6e1a4f2e";

    fn test_pubkey() -> Vec<u8> {
        hex::decode(TEST_PUBKEY_HEX).unwrap()
    }

    #[test]
    fn test_pubkey_to_address_bytes() {
        let pubkey = test_pubkey();
        let addr_bytes = pubkey_to_address_bytes(&pubkey).unwrap();
        assert_eq!(addr_bytes.len(), 20);
    }

    #[test]
    fn test_pubkey_to_bech32() {
        let pubkey = test_pubkey();
        let address = pubkey_to_bech32(&pubkey, hrps::COSMOS).unwrap();
        assert!(address.starts_with("cosmos1"));
    }

    #[test]
    fn test_different_hrps() {
        let pubkey = test_pubkey();

        let cosmos_addr = pubkey_to_bech32(&pubkey, hrps::COSMOS).unwrap();
        let osmo_addr = pubkey_to_bech32(&pubkey, hrps::OSMOSIS).unwrap();
        let juno_addr = pubkey_to_bech32(&pubkey, hrps::JUNO).unwrap();

        assert!(cosmos_addr.starts_with("cosmos1"));
        assert!(osmo_addr.starts_with("osmo1"));
        assert!(juno_addr.starts_with("juno1"));

        // Same pubkey = same address bytes, different HRPs
        let cosmos_data = cosmos_addr.strip_prefix("cosmos1").unwrap();
        let osmo_data = osmo_addr.strip_prefix("osmo1").unwrap();
        // The data part encoding differs because HRP affects checksum
        assert_ne!(cosmos_data, osmo_data);
    }

    #[test]
    fn test_invalid_pubkey_length() {
        let short_key = vec![0u8; 32]; // Should be 33
        assert!(pubkey_to_address_bytes(&short_key).is_err());
    }

    #[test]
    fn test_pattern_matching() {
        let address = "cosmos1abc123def456";

        assert!(matches_prefix(address, "cosmos", "abc"));
        assert!(!matches_prefix(address, "cosmos", "xyz"));

        assert!(matches_suffix(address, "456"));
        assert!(!matches_suffix(address, "789"));

        assert!(matches_contains(address, "cosmos", "123"));
        assert!(!matches_contains(address, "cosmos", "zzz"));
    }

    #[test]
    fn test_vanity_pattern_validation() {
        let valid = VanityPattern::Prefix("aqc923".to_string());
        assert!(valid.validate_bech32_charset().is_ok());

        let invalid = VanityPattern::Prefix("ABC".to_string()); // uppercase not in bech32
        assert!(invalid.validate_bech32_charset().is_err());

        let invalid2 = VanityPattern::Prefix("abc".to_string()); // 'b' not in bech32
        assert!(invalid2.validate_bech32_charset().is_err());
    }

    #[test]
    fn test_end_to_end_with_keyderiv() {
        use cosmos_vanity_keyderiv::{derive_keypair_from_mnemonic, DEFAULT_COSMOS_PATH};

        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
        let key = derive_keypair_from_mnemonic(mnemonic, DEFAULT_COSMOS_PATH).unwrap();
        let address = pubkey_to_bech32(key.public_key_bytes(), hrps::COSMOS).unwrap();

        assert!(address.starts_with("cosmos1"));
        // Address should be ~44 characters
        assert!(address.len() > 40);
        assert!(address.len() < 50);
    }
}
