use criterion::{criterion_group, criterion_main, Criterion};

use cosmos_vanity_address::pubkey_to_bech32;
use cosmos_vanity_keyderiv::generate_random_keypair;

fn bench_full_pipeline(c: &mut Criterion) {
    c.bench_function("full_pipeline_single", |b| {
        b.iter(|| {
            let key = generate_random_keypair().unwrap();
            let _address = pubkey_to_bech32(key.public_key_bytes(), "cosmos").unwrap();
        })
    });
}

fn bench_key_generation(c: &mut Criterion) {
    c.bench_function("key_generation", |b| {
        b.iter(|| {
            let _key = generate_random_keypair().unwrap();
        })
    });
}

fn bench_address_hashing(c: &mut Criterion) {
    let key = generate_random_keypair().unwrap();
    let pubkey = key.public_key_bytes().to_vec();

    c.bench_function("address_hashing", |b| {
        b.iter(|| {
            let _address = pubkey_to_bech32(&pubkey, "cosmos").unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_full_pipeline,
    bench_key_generation,
    bench_address_hashing
);
criterion_main!(benches);
