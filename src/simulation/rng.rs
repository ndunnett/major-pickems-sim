use rand::prelude::*;

pub type RngType = rand::rngs::Xoshiro256PlusPlus;

const SEED: u64 = 7_355_608;

/// Random number generator initialised from given seed.
#[must_use]
#[inline]
pub fn seeded(seed: u64) -> RngType {
    RngType::seed_from_u64(seed)
}

/// Deterministic random number generator to use for testing/benchmarking.
#[must_use]
#[inline]
pub fn deterministic() -> RngType {
    seeded(SEED)
}

/// Randomly seeded random number generator.
#[must_use]
#[inline]
pub fn random() -> RngType {
    RngType::from_rng(&mut rand::rng())
}

/// Completely not-random RNG interface, implementing `rand` traits.
/// Always returns 50%, useful for eliminating micro statistical changes from tests.
/// Deliberately only available in tests.
#[cfg(test)]
pub struct HalfRng;

#[cfg(test)]
impl rand::TryRng for HalfRng {
    type Error = std::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(0x8000_0000)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(0x8000_0000_0000_0000)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            chunk.copy_from_slice(&self.try_next_u64()?.to_le_bytes()[..chunk.len()]);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_rng() {
        let mut rng = HalfRng;
        assert_eq!(rng.random::<f32>().to_bits(), 0.5_f32.to_bits());
    }
}
