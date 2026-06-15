use criterion::{BatchSize, BenchmarkId, Criterion};
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use pickems::{
    datatypes::{Sigma, Teams},
    simulation::SwissSystem,
};

const SEED: u64 = 7_355_608;

fn cases() -> [SwissSystem; 5] {
    let mut swiss = SwissSystem::new(Teams::dummy().ratings, Sigma::default());
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);

    std::array::from_fn(|_| {
        let state = swiss;
        swiss.simulate_round(&mut rng);
        state
    })
}

pub fn bench(c: &mut Criterion) {
    let cases = cases();
    let mut group = c.benchmark_group("round_simulation");

    for (round, state) in cases.iter().enumerate() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(SEED + round as u64);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("round_{}", round + 1)),
            state,
            |b, state| {
                b.iter_batched_ref(
                    || (*state, rng.clone()),
                    |(swiss, rng)| swiss.simulate_round(rng),
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}
