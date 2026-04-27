//! # cosmos-vanity-keyderiv
//!
//! BIP-39 mnemonic generation and BIP-32/44 HD key derivation for Cosmos SDK chains.
//!
//! ## Security
//!
//! All sensitive key material implements [`Zeroize`] and is zeroed on drop.
//! Mnemonics and private keys should never be logged or persisted in plaintext.

use bip39::Mnemonic;
use bitcoin::bip32::{DerivationPath, Xpriv};
use bitcoin::NetworkKind;
use secp256k1::Secp256k1;
use zeroize::{Zeroize, ZeroizeOnDrop};

pub use bip39;
pub use secp256k1;

/// Default Cosmos SDK derivation path: m/44'/118'/0'/0/0
pub const DEFAULT_COSMOS_PATH: &str = "m/44'/118'/0'/0/0";

/// Default coin type for Cosmos
pub const COSMOS_COIN_TYPE: u32 = 118;

/// Errors from key derivation operations.
#[derive(Debug, thiserror::Error)]
pub enum KeyDerivError {
    #[error("BIP-39 error: {0}")]
    Bip39(String),

    #[error("BIP-32 derivation error: {0}")]
    Bip32(String),

    #[error("secp256k1 error: {0}")]
    Secp256k1(#[from] secp256k1::Error),

    #[error("invalid derivation path: {0}")]
    InvalidPath(String),

    #[error("invalid mnemonic word count: {0} (expected 12 or 24)")]
    InvalidWordCount(u8),
}

/// A derived keypair with its associated mnemonic and derivation path.
///
/// # Security
///
/// This struct holds sensitive cryptographic material. It implements
/// [`ZeroizeOnDrop`] to ensure secrets are cleared from memory when dropped.
#[derive(Clone, ZeroizeOnDrop)]
pub struct DerivedKey {
    /// The BIP-39 mnemonic phrase.
    mnemonic_phrase: String,

    /// Raw seed bytes derived from the mnemonic
    seed: Vec<u8>,

    /// The derived private key bytes (32 bytes)
    secret_key_bytes: Vec<u8>,

    /// The compressed public key bytes (33 bytes)
    #[zeroize(skip)]
    public_key_bytes: Vec<u8>,

    /// The derivation path used
    #[zeroize(skip)]
    derivation_path: String,
}

impl DerivedKey {
    /// Returns the mnemonic phrase.
    pub fn mnemonic(&self) -> &str {
        &self.mnemonic_phrase
    }

    /// Returns the compressed public key bytes (33 bytes).
    pub fn public_key_bytes(&self) -> &[u8] {
        &self.public_key_bytes
    }

    /// Returns the derivation path.
    pub fn derivation_path(&self) -> &str {
        &self.derivation_path
    }

    /// Returns the secret key bytes (32 bytes).
    ///
    /// # Security
    ///
    /// Handle with extreme care. Never log or persist.
    pub fn secret_key_bytes(&self) -> &[u8] {
        &self.secret_key_bytes
    }
}

/// Derive compressed public key from raw 32-byte private key (for GPU verification).
///
/// This is a direct secp256k1 scalar multiplication without BIP-39/BIP-32.
/// Used to verify GPU-computed public keys against CPU reference.
pub fn pubkey_from_privkey(privkey: &[u8; 32]) -> Result<[u8; 33], KeyDerivError> {
    let secp = Secp256k1::new();
    let secret = secp256k1::SecretKey::from_slice(privkey)?;
    let public = secret.public_key(&secp);
    Ok(public.serialize())
}

/// Generate a random 24-word BIP-39 mnemonic and derive the keypair.
///
/// Uses the default Cosmos derivation path `m/44'/118'/0'/0/0`.
pub fn generate_random_keypair() -> Result<DerivedKey, KeyDerivError> {
    generate_random_keypair_with_path(DEFAULT_COSMOS_PATH)
}

/// Generate a random 24-word BIP-39 mnemonic and derive the keypair
/// using a custom derivation path.
pub fn generate_random_keypair_with_path(path: &str) -> Result<DerivedKey, KeyDerivError> {
    generate_random_keypair_with_words(path, 24)
}

/// Generate a random BIP-39 mnemonic with the specified word count and derive the keypair.
/// Supported: 12 words (128-bit entropy) or 24 words (256-bit entropy).
pub fn generate_random_keypair_with_words(
    path: &str,
    words: u8,
) -> Result<DerivedKey, KeyDerivError> {
    let entropy_len = match words {
        12 => 16,
        24 => 32,
        other => return Err(KeyDerivError::InvalidWordCount(other)),
    };
    let mut entropy = [0u8; 32];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut entropy[..entropy_len]);
    let mnemonic = Mnemonic::from_entropy(&entropy[..entropy_len])
        .map_err(|e| KeyDerivError::Bip39(e.to_string()))?;
    entropy.zeroize();
    derive_keypair_from_mnemonic(&mnemonic.to_string(), path)
}

/// Derive a keypair from an existing mnemonic phrase and derivation path.
pub fn derive_keypair_from_mnemonic(
    mnemonic_str: &str,
    path: &str,
) -> Result<DerivedKey, KeyDerivError> {
    let mnemonic: Mnemonic = mnemonic_str
        .parse()
        .map_err(|e: bip39::Error| KeyDerivError::Bip39(e.to_string()))?;

    // BIP-39 seed (no passphrase — standard for Cosmos wallets)
    let seed = mnemonic.to_seed("");

    // Parse derivation path
    let derivation_path: DerivationPath = path
        .parse()
        .map_err(|e: bitcoin::bip32::Error| KeyDerivError::InvalidPath(e.to_string()))?;

    // Derive the extended private key
    let xpriv = Xpriv::new_master(NetworkKind::Main, &seed)
        .map_err(|e| KeyDerivError::Bip32(e.to_string()))?;

    let secp = Secp256k1::new();
    let derived = xpriv
        .derive_priv(&secp, &derivation_path)
        .map_err(|e| KeyDerivError::Bip32(e.to_string()))?;

    // Get the public key
    let secret_key = derived.private_key;
    let public_key = secret_key.public_key(&secp);

    Ok(DerivedKey {
        mnemonic_phrase: mnemonic.to_string(),
        seed: seed.to_vec(),
        secret_key_bytes: secret_key.secret_bytes().to_vec(),
        public_key_bytes: public_key.serialize().to_vec(),
        derivation_path: path.to_string(),
    })
}

/// Build a derivation path for a given coin type with default account/change/index.
pub fn cosmos_derivation_path(coin_type: u32) -> String {
    format!("m/44'/{coin_type}'/0'/0/0")
}

/// Build a full custom derivation path.
pub fn custom_derivation_path(coin_type: u32, account: u32, change: u32, index: u32) -> String {
    format!("m/44'/{coin_type}'/{account}'/{change}/{index}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_keypair() {
        let key = generate_random_keypair().unwrap();
        assert_eq!(key.public_key_bytes().len(), 33); // compressed pubkey
        assert_eq!(key.secret_key_bytes().len(), 32);
        assert_eq!(key.derivation_path(), DEFAULT_COSMOS_PATH);

        // Mnemonic should be 24 words
        let words: Vec<&str> = key.mnemonic().split_whitespace().collect();
        assert_eq!(words.len(), 24);
    }

    #[test]
    fn test_deterministic_derivation() {
        // Known test vector — derive same mnemonic twice, get same keys
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

        let key1 = derive_keypair_from_mnemonic(mnemonic, DEFAULT_COSMOS_PATH).unwrap();
        let key2 = derive_keypair_from_mnemonic(mnemonic, DEFAULT_COSMOS_PATH).unwrap();

        assert_eq!(key1.public_key_bytes(), key2.public_key_bytes());
        assert_eq!(key1.secret_key_bytes(), key2.secret_key_bytes());
    }

    #[test]
    fn test_different_paths_give_different_keys() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

        let key1 = derive_keypair_from_mnemonic(mnemonic, "m/44'/118'/0'/0/0").unwrap();
        let key2 = derive_keypair_from_mnemonic(mnemonic, "m/44'/118'/0'/0/1").unwrap();

        assert_ne!(key1.public_key_bytes(), key2.public_key_bytes());
    }

    #[test]
    fn test_cosmos_derivation_path() {
        assert_eq!(cosmos_derivation_path(118), "m/44'/118'/0'/0/0");
        assert_eq!(cosmos_derivation_path(330), "m/44'/330'/0'/0/0"); // Terra
    }

    #[test]
    fn test_invalid_mnemonic() {
        let result = derive_keypair_from_mnemonic("invalid mnemonic words", DEFAULT_COSMOS_PATH);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_path() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
        let result = derive_keypair_from_mnemonic(mnemonic, "not/a/valid/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_word_count_rejected() {
        let result = generate_random_keypair_with_words(DEFAULT_COSMOS_PATH, 13);
        assert!(matches!(result, Err(KeyDerivError::InvalidWordCount(13))));
    }
}
