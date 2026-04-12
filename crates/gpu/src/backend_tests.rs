pub(crate) trait BackendHarness {
    fn label(&self) -> &'static str;
    fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> anyhow::Result<Vec<u8>>;
    fn has_secp256k1_kernel(&self) -> bool;
    fn generate_and_hash_batch(
        &self,
        privkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)>;
    fn has_mnemonic_kernel(&self) -> bool;
    fn mnemonic_batch(
        &self,
        mnemonics_flat: &[u8],
        mnemonic_lens: &[u32],
    ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)>;
}

pub(crate) fn assert_hash_matches_cpu<B: BackendHarness>(ctx: &B) {
    use cosmos_vanity_address::pubkey_to_address_bytes;

    let test_pubkeys: Vec<[u8; 33]> = vec![
        hex::decode("02394bc53633366a2ab9b5d4a7b6a9cfd9f11d576e45e1e2049a2e397b6e1a4f2e")
            .unwrap()
            .try_into()
            .unwrap(),
        {
            let mut k = [0u8; 33];
            k[0] = 0x02;
            k
        },
        {
            let mut k = [0xFFu8; 33];
            k[0] = 0x03;
            k
        },
        {
            let mut k = [0u8; 33];
            k[0] = 0x02;
            for (i, byte) in k.iter_mut().enumerate().skip(1) {
                *byte = i as u8;
            }
            k
        },
    ];

    let mut flat_pubkeys = Vec::with_capacity(test_pubkeys.len() * 33);
    for pk in &test_pubkeys {
        flat_pubkeys.extend_from_slice(pk);
    }

    let gpu_hashes = ctx
        .hash_pubkeys_batch(&flat_pubkeys)
        .expect("GPU hashing failed");

    for (i, pk) in test_pubkeys.iter().enumerate() {
        let cpu_hash = pubkey_to_address_bytes(pk).unwrap();
        let gpu_hash = &gpu_hashes[i * 20..(i + 1) * 20];
        assert_eq!(
            cpu_hash.as_slice(),
            gpu_hash,
            "{} hash mismatch for test vector {i}!\n  pubkey: {}\n  CPU:    {}\n  GPU:    {}",
            ctx.label(),
            hex::encode(pk),
            hex::encode(cpu_hash),
            hex::encode(gpu_hash),
        );
    }
}

pub(crate) fn assert_secp256k1_known_vector<B: BackendHarness>(ctx: &B) {
    if !ctx.has_secp256k1_kernel() {
        eprintln!("{} secp256k1 kernel not available, skipping", ctx.label());
        return;
    }

    let mut privkey = [0u8; 32];
    privkey[31] = 1;

    let (pubkeys, _hashes, _matches) = ctx
        .generate_and_hash_batch(&privkey, &[])
        .expect("GPU secp256k1 failed");

    let expected_hex = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    assert_eq!(pubkeys.len(), 33, "Expected 33-byte pubkey");
    assert_eq!(
        hex::encode(&pubkeys[..33]),
        expected_hex,
        "{} pubkey for privkey=1 did not match the generator point",
        ctx.label(),
    );
}

pub(crate) fn assert_secp256k1_matches_cpu<B: BackendHarness>(ctx: &B) {
    use cosmos_vanity_keyderiv::pubkey_from_privkey;

    if !ctx.has_secp256k1_kernel() {
        eprintln!("{} secp256k1 kernel not available, skipping", ctx.label());
        return;
    }

    let test_privkeys: Vec<[u8; 32]> = vec![
        {
            let mut k = [0u8; 32];
            k[31] = 1;
            k
        },
        {
            let mut k = [0u8; 32];
            k[31] = 2;
            k
        },
        {
            let mut k = [0u8; 32];
            k[31] = 7;
            k
        },
        {
            let mut k = [0u8; 32];
            for (i, byte) in k.iter_mut().enumerate() {
                *byte = (i as u8 + 1) * 7;
            }
            k
        },
    ];

    let mut flat_privkeys = Vec::with_capacity(test_privkeys.len() * 32);
    for pk in &test_privkeys {
        flat_privkeys.extend_from_slice(pk);
    }

    let (gpu_pubkeys, gpu_hashes, _) = ctx
        .generate_and_hash_batch(&flat_privkeys, &[])
        .expect("GPU secp256k1 batch failed");

    for (i, privkey) in test_privkeys.iter().enumerate() {
        let cpu_pubkey = pubkey_from_privkey(privkey).expect("CPU pubkey derivation failed");
        let gpu_pubkey = &gpu_pubkeys[i * 33..(i + 1) * 33];
        assert_eq!(
            cpu_pubkey.as_slice(),
            gpu_pubkey,
            "{} pubkey mismatch for test vector {i}!\n  privkey: {}\n  CPU: {}\n  GPU: {}",
            ctx.label(),
            hex::encode(privkey),
            hex::encode(cpu_pubkey),
            hex::encode(gpu_pubkey),
        );

        let cpu_hash = cosmos_vanity_address::pubkey_to_address_bytes(&cpu_pubkey).unwrap();
        let gpu_hash = &gpu_hashes[i * 20..(i + 1) * 20];
        assert_eq!(
            cpu_hash.as_slice(),
            gpu_hash,
            "{} hash mismatch for test vector {i}!\n  CPU: {}\n  GPU: {}",
            ctx.label(),
            hex::encode(cpu_hash),
            hex::encode(gpu_hash),
        );
    }
}

pub(crate) fn assert_mnemonic_pipeline<B: BackendHarness>(ctx: &B) {
    if !ctx.has_mnemonic_kernel() {
        eprintln!("{} mnemonic kernel not available, skipping", ctx.label());
        return;
    }

    let mnemonic = "monster asthma shaft average main office dial since rural guitar estate sight";
    let mnemonic_bytes = mnemonic.as_bytes();
    let mnemonic_len = mnemonic_bytes.len() as u32;

    let mut padded = vec![0u8; 256];
    padded[..mnemonic_bytes.len()].copy_from_slice(mnemonic_bytes);

    let (privkeys, hashes, _matches) = ctx
        .mnemonic_batch(&padded, &[mnemonic_len])
        .expect("GPU mnemonic pipeline failed");

    let cpu_key = cosmos_vanity_keyderiv::derive_keypair_from_mnemonic(
        mnemonic,
        cosmos_vanity_keyderiv::DEFAULT_COSMOS_PATH,
    )
    .unwrap();
    let cpu_addr =
        cosmos_vanity_address::pubkey_to_bech32(cpu_key.public_key_bytes(), "cosmos").unwrap();

    let gpu_privkey = &privkeys[..32];
    assert_eq!(
        cpu_key.secret_key_bytes(),
        gpu_privkey,
        "{} mnemonic pipeline private key mismatch\n  CPU: {}\n  GPU: {}",
        ctx.label(),
        hex::encode(cpu_key.secret_key_bytes()),
        hex::encode(gpu_privkey),
    );

    let mut gpu_addr_bytes = [0u8; 20];
    gpu_addr_bytes.copy_from_slice(&hashes[..20]);
    let gpu_addr = cosmos_vanity_address::encode_bech32("cosmos", &gpu_addr_bytes).unwrap();

    assert_eq!(
        cpu_addr,
        gpu_addr,
        "{} mnemonic pipeline address mismatch\n  CPU: {}\n  GPU: {}",
        ctx.label(),
        cpu_addr,
        gpu_addr,
    );
    assert_eq!(cpu_addr, "cosmos1u2gukdek3gtxgz6f89jgvh7pw2286smk48vxm4");
}
