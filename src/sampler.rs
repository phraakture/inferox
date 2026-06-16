//! Sampling strategies for turning logits into the next token id.

use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Configurable sampler with temperature, top-k, and top-p filtering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    /// Temperature for softmax. 0.0 means greedy argmax.
    pub temperature: f32,
    /// Keep only the top-k most likely tokens. 0 disables.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose cumulative probability exceeds
    /// this value. 1.0 disables.
    pub top_p: f32,
}

impl Sampler {
    /// Greedy sampler (always picks the highest logit).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }

    pub fn sample(&self, logits: &[f32], rng: &mut impl Rng) -> u32 {
        if self.temperature == 0.0 {
            return argmax(logits);
        }

        // temperature-scaled softmax
        let mut probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &z)| (i, (z / self.temperature).exp()))
            .collect();

        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut probs {
            *p /= sum;
        }

        // top-k
        if self.top_k > 0 {
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            probs.truncate(self.top_k);
        }

        // top-p (nucleus)
        if self.top_p < 1.0 {
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cumsum = 0.0f32;
            let cutoff = probs
                .iter()
                .position(|(_, p)| {
                    cumsum += p;
                    cumsum >= self.top_p
                })
                .map(|i| i + 1)
                .unwrap_or(probs.len());
            probs.truncate(cutoff);
        }

        // renormalize
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        let weights: Vec<f32> = probs.iter().map(|(_, p)| p / sum).collect();

        let dist = WeightedIndex::new(&weights).unwrap();
        probs[dist.sample(rng)].0 as u32
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_highest() {
        let sampler = Sampler::greedy();
        let mut rng = rand::thread_rng();
        let logits = vec![1.0, 3.0, 2.0];
        assert_eq!(sampler.sample(&logits, &mut rng), 1);
    }

    #[test]
    fn temperature_non_greedy_samples_from_distribution() {
        let sampler = Sampler {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut rng = rand::thread_rng();
        let logits = vec![0.0, 1.0, 0.0];
        // mostly token 1, but could be others
        let token = sampler.sample(&logits, &mut rng);
        assert!(token < 3);
    }

    #[test]
    fn top_k_truncates() {
        let sampler = Sampler {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        let mut rng = rand::thread_rng();
        let logits = vec![10.0, 1.0, 1.0, 1.0];
        let token = sampler.sample(&logits, &mut rng);
        // With top_k=2, token 0 is overwhelmingly likely; sampling should
        // never return indices >= 2 because they are truncated.
        assert!(token <= 1);
    }
}
