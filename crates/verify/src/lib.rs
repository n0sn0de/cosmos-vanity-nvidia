//! # cosmos-vanity-verify
//!
//! Deterministic CPU verification of GPU-found vanity address matches.
//!
//! Every match found by the GPU must be independently verified on the CPU
//! before being reported to the user. This ensures correctness even if
//! the GPU kernel has bugs.
//!
//! ## Verification Steps
//!
//! 1. Re-derive the keypair from the mnemonic using the exact derivation path
//! 2. Re-compute the Bech32 address from the derived public key
//! 3. Verify the address matches the claimed address exactly
//! 4. Verify the pattern match against the address

use cosmos_vanity_address::{pubkey_to_bech32, VanityPattern};
use cosmos_vanity_keyderiv::derive_keypair_from_mnemonic;

/// Errors from verification.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("key derivation failed: {0}")]
    KeyDerivation(#[from] cosmos_vanity_keyderiv::KeyDerivError),

    #[error("address generation failed: {0}")]
    AddressGeneration(#[from] cosmos_vanity_address::AddressError),

    #[error("address mismatch: expected {expected}, got {actual}")]
    AddressMismatch { expected: String, actual: String },

    #[error("pattern does not match the verified address")]
    PatternMismatch,
}

/// Result of a verification check.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the verification passed
    pub verified: bool,

    /// The re-derived address
    pub address: String,

    /// The original claimed address
    pub claimed_address: String,

    /// Details about any failure
    pub error: Option<String>,
}

/// Verify a vanity address match by re-deriving from the mnemonic.
///
/// This is the critical security function — never skip this step.
pub fn verify_match(
    mnemonic: &str,
    derivation_path: &str,
    hrp: &str,
    claimed_address: &str,
    pattern: &VanityPattern,
) -> Result<VerificationResult, VerifyError> {
    tracing::debug!("Verifying match: {claimed_address}");

    // Step 1: Re-derive the keypair
    let key = derive_keypair_from_mnemonic(mnemonic, derivation_path)?;

    // Step 2: Re-compute the address
    let verified_address = pubkey_to_bech32(key.public_key_bytes(), hrp)?;

    // Step 3: Compare addresses
    if verified_address != claimed_address {
        tracing::warn!(
            "Address mismatch! Claimed: {}, Verified: {}",
            claimed_address,
            verified_address
        );
        return Ok(VerificationResult {
            verified: false,
            address: verified_address.clone(),
            claimed_address: claimed_address.to_string(),
            error: Some(format!(
                "Address mismatch: claimed {claimed_address}, derived {verified_address}"
            )),
        });
    }

    // Step 4: Verify the pattern match
    if !pattern.matches(&verified_address, hrp) {
        tracing::warn!(
            "Pattern mismatch! Address {} doesn't match pattern {}",
            verified_address,
            pattern
        );
        return Ok(VerificationResult {
            verified: false,
            address: verified_address,
            claimed_address: claimed_address.to_string(),
            error: Some(format!("Pattern {pattern} does not match address")),
        });
    }

    tracing::info!("✅ Verified: {verified_address}");

    Ok(VerificationResult {
        verified: true,
        address: verified_address,
        claimed_address: claimed_address.to_string(),
        error: None,
    })
}

/// Verify just that a mnemonic produces the expected address (without pattern check).
pub fn verify_address(
    mnemonic: &str,
    derivation_path: &str,
    hrp: &str,
    expected_address: &str,
) -> Result<bool, VerifyError> {
    let key = derive_keypair_from_mnemonic(mnemonic, derivation_path)?;
    let address = pubkey_to_bech32(key.public_key_bytes(), hrp)?;
    Ok(address == expected_address)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

    #[test]
    fn test_verify_correct_address() {
        // First, derive the actual address
        let key = derive_keypair_from_mnemonic(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
        )
        .unwrap();

        let address =
            pubkey_to_bech32(key.public_key_bytes(), "cosmos").unwrap();

        // Now verify it
        let result = verify_address(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
            "cosmos",
            &address,
        )
        .unwrap();

        assert!(result);
    }

    #[test]
    fn test_verify_wrong_address() {
        let result = verify_address(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
            "cosmos",
            "cosmos1invalidaddresshere",
        )
        .unwrap();

        assert!(!result);
    }

    #[test]
    fn test_verify_match_with_pattern() {
        let key = derive_keypair_from_mnemonic(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
        )
        .unwrap();

        let address =
            pubkey_to_bech32(key.public_key_bytes(), "cosmos").unwrap();

        // Get the first char after "cosmos1" for a guaranteed prefix match
        let first_char = address
            .strip_prefix("cosmos1")
            .unwrap()
            .chars()
            .next()
            .unwrap();

        let pattern = VanityPattern::Prefix(first_char.to_string());

        let result = verify_match(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
            "cosmos",
            &address,
            &pattern,
        )
        .unwrap();

        assert!(result.verified);
    }

    #[test]
    fn test_verify_match_pattern_mismatch() {
        let key = derive_keypair_from_mnemonic(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
        )
        .unwrap();

        let address =
            pubkey_to_bech32(key.public_key_bytes(), "cosmos").unwrap();

        // Use a pattern that definitely won't match (all z's)
        let pattern = VanityPattern::Prefix("zzzzzzzz".to_string());

        let result = verify_match(
            TEST_MNEMONIC,
            cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
            "cosmos",
            &address,
            &pattern,
        )
        .unwrap();

        assert!(!result.verified);
        assert!(result.error.is_some());
    }
}
