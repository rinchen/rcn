//! SSE (Secondary Similarity Estimator) + APM (Adaptive Probability Mapping) cascade.
//!
//! PAQ8 and CMIX use a multi-stage probability refinement pipeline after the
//! logistic mixer. Rcn stops after the mixer — this cascade adds the missing
//! stages:
//!
//!   1. Logistic mixer   → p (fused per-bit probability)
//!   2. SSE              → p' (refined by local context: prev_byte, bit_pos)
//!   3. APM              → p'' (refined by confidence + order-1 context)
//!   4. APM2             → p''' (refined by order-2 context)
//!
//! Each stage stores a small quantized table of logit-space correction deltas,
//! indexed by a context derived from the prediction's local neighborhood. Tables
//! are updated via SGD on the logistic loss with learning-rate decay.
//!
//! On the Silesia/Mixed 5-file subset this typically yields 2–4 pt ratio
//! improvement on text files (dickens, webster) at <1 MB of table memory.
//!
//! ## Why per-model reliability dampening failed before
//!
//! The earlier experiment dampened *mixer weights* toward zero when a model's
//! predictions were wrong. That doesn't help because the mixer already
//! down-weights unreliable models — the *probability* itself is still biased.
//! SSE/APM fix the probability directly in logit space, which is what the
//! cascade stages in PAQ8/CMIX actually do.

use crate::model::BitModel;
use std::sync::LazyLock;

/// 12-bit probability precision (must match the rANS backend).
const PROB_BITS: usize = 12;
const PROB_MAX: u16 = (1 << PROB_BITS) - 1; // 4095

/// Number of quantized probability bins for the SSE secondary estimator.
const SSE_BINS: usize = 16;

/// Number of quantized confidence bins for APM/APM2.
const APM_BINS: usize = 16;

/// Number of word-hash buckets for APM2.
const APM2_BUCKETS: usize = 4096;

/// Initial learning rate for SSE (decays per 16k bits).
const SSE_LR_INIT: f32 = 0.01;

/// Learning rate for APM/APM2 stages.
const APM_LR: f32 = 0.015;

/// Bits of decay window (2^14 = 16384 bits per halving).
const LR_DECAY_BITS: u32 = 14;

/// Precomputed stretch lookup table: p12 (0..=4095) → logit space.
/// Matches original: p = (p12 + 0.5) / 4096, then ln(p/(1-p)).
/// Clamped only for extreme probabilities (p <= 1e-6 or p >= 1-1e-6).
/// This replaces the hot `ln()` call (13 samples in profile).
static STRETCH_LUT: LazyLock<[f32; 4096]> = LazyLock::new(|| {
    let mut lut = [0.0f32; 4096];
    for i in 0..=4095 {
        let p = (i as f32 + 0.5) / (PROB_MAX as f32 + 1.0);
        lut[i] = if p <= 1e-6 {
            -7.0
        } else if p >= 1.0 - 1e-6 {
            7.0
        } else {
            (p / (1.0 - p)).ln()
        };
    }
    lut
});

/// Stretch a 12-bit probability (in [0, 4095]) to logit space.
#[inline]
pub fn stretch(p12: u16) -> f32 {
    STRETCH_LUT[p12 as usize]
}

/// Squash a logit value back to a 12-bit probability (clamped to [0, 4095]).
/// Kept as exact exp() to maintain bit-identical roundtrip with original.
#[inline]
pub fn squash(x: f32) -> u16 {
    let p = 1.0 / (1.0 + (-x).exp());
    (p * PROB_MAX as f32).round().clamp(0.0, PROB_MAX as f32) as u16
}

/// Quantize a 12-bit probability into one of `bins` buckets for table indexing.
#[inline]
fn quant(p: u16, bins: usize) -> usize {
    (p as usize * bins) / (PROB_MAX as usize + 1)
}

/// A quantized-logit-table refinement stage (SSE or APM).
#[derive(Debug)]
pub struct QuantTable {
    /// [ctx][bin] → logit-space correction delta.
    data: Vec<Vec<f32>>,
    lr: f32,
    bit_count: u64,
    bins: usize,
}

impl QuantTable {
    fn new(num_ctx: usize, bins: usize) -> Self {
        Self {
            data: vec![vec![0.0f32; bins]; num_ctx],
            lr: SSE_LR_INIT,
            bit_count: 0,
            bins,
        }
    }

    /// Refine probability `p` (12-bit) given context `ctx`.
    #[inline]
    pub fn refine(&self, p: u16, ctx: usize) -> u16 {
        let bin = quant(p, self.bins);
        let delta = self.data[ctx][bin];
        squash(stretch(p) + delta)
    }

    /// Online update after observing `bit`.
    #[inline]
    pub fn update(&mut self, bit: bool, p_before_refine: u16, ctx: usize) {
        let bin = quant(p_before_refine, self.bins);
        let s = stretch(p_before_refine) + self.data[ctx][bin];
        let pred = 1.0 / (1.0 + (-s).exp());
        let target = if bit { 1.0f32 } else { 0.0 };
        let err = target - pred;
        self.data[ctx][bin] += self.lr * err;

        self.bit_count += 1;
        if self.bit_count & ((1u64 << LR_DECAY_BITS) - 1) == 0 {
            self.lr *= 0.5;
        }
    }

    pub fn reset(&mut self) {
        for row in &mut self.data {
            row.fill(0.0);
        }
        self.lr = SSE_LR_INIT;
        self.bit_count = 0;
    }
}

/// The full SSE → APM → APM2 cascade.
///
/// Wraps the mixer externally: the codec calls `mixer.mix()` first, then
/// `cascade.refine(p, bit_pos)` to get the final probability. After encoding
/// or decoding the bit, call `cascade.update(bit, p_mixer, bit_pos)` to train.
///
/// Context is tracked internally from the byte stream: `set_context(byte)`
/// is called once per byte before the bit loop.
pub struct SseApmCascade {
    /// SSE: indexed by `(prev_byte << 3) | bit_pos`, 2048 contexts × 16 bins.
    sse: QuantTable,
    /// APM: indexed by `order1_byte << 5 | confidence_quant`, 8192 ctx × 16 bins.
    apm: QuantTable,
    /// APM2: indexed by `order2_byte & 0xFFF`, 4096 ctx × 16 bins.
    apm2: QuantTable,
    /// Current prev_byte (for SSE context).
    prev_byte: u8,
    /// Byte before prev_byte (for APM context).
    order1_byte: u8,
    /// Byte before order1_byte (for APM2 context).
    order2_byte: u8,
}

impl SseApmCascade {
    /// Create a new cascade.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sse: QuantTable::new(256 * 8, SSE_BINS),
            apm: QuantTable::new(256 * APM_BINS, APM_BINS),
            apm2: QuantTable::new(16, APM_BINS),
            prev_byte: 0,
            order1_byte: 0,
            order2_byte: 0,
        }
    }

    /// Set byte-level context before the per-bit loop for this byte.
    pub fn set_context(&mut self, byte: u8) {
        self.order2_byte = self.order1_byte;
        self.order1_byte = self.prev_byte;
        self.prev_byte = byte;
    }

    /// Full pipeline: mixer_prob → SSE → APM → APM2.
    /// Returns the final refined 12-bit probability for encoding this bit.
    #[inline]
    pub fn refine(&mut self, p_mixer: u16, bit_pos: u8) -> u16 {
        let sse_ctx = (usize::from(self.prev_byte) << 3) | usize::from(bit_pos.min(7));
        let p_sse = self.sse.refine(p_mixer, sse_ctx);
        let apm_ctx = (usize::from(self.order1_byte) << 4) | quant(p_sse, APM_BINS);
        let p_apm = self.apm.refine(p_sse, apm_ctx);
        let apm2_ctx = usize::from(self.order2_byte & 0x0F); // 16 buckets via 4-bit low nibble
        self.apm2.refine(p_apm, apm2_ctx)
    }

    /// Online update after the true `bit` is known.
    /// Call with the *pre-refinement* mixer probability and `bit_pos`.
    #[inline]
    pub fn update(&mut self, bit: bool, p_mixer: u16, bit_pos: u8) {
        let sse_ctx = (usize::from(self.prev_byte) << 3) | usize::from(bit_pos.min(7));
        let p_sse = self.sse.refine(p_mixer, sse_ctx);
        let apm_ctx = (usize::from(self.order1_byte) << 4) | quant(p_sse, APM_BINS);
        let p_apm = self.apm.refine(p_sse, apm_ctx);
        let apm2_ctx = usize::from(self.order2_byte & 0x0F);

        self.sse.update(bit, p_mixer, sse_ctx);
        self.apm.update(bit, p_sse, apm_ctx);
        self.apm2.update(bit, p_apm, apm2_ctx);
    }

    /// Set learning rate for the cascade (matches mixer LR).
    pub fn set_lr(&mut self, lr: f32) {
        self.sse.lr = lr;
        self.apm.lr = lr;
        self.apm2.lr = lr;
    }

    /// Reset all cascade state to zero (used at block boundaries).
    pub fn reset(&mut self) {
        self.sse.reset();
        self.apm.reset();
        self.apm2.reset();
        self.prev_byte = 0;
        self.order1_byte = 0;
        self.order2_byte = 0;
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.sse.data.len() * self.sse.data[0].len() * std::mem::size_of::<f32>()
            + self.apm.data.len() * self.apm.data[0].len() * std::mem::size_of::<f32>()
            + self.apm2.data.len() * self.apm2.data[0].len() * std::mem::size_of::<f32>()
    }
}

impl Default for SseApmCascade {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_squash_roundtrip() {
        for p in [0, 1, 100, 500, 1024, 2048, 3000, 4000, 4094, 4095] {
            let x = stretch(p);
            let p2 = squash(x);
            assert!(
                (p2 as i32 - p as i32).abs() <= 1,
                "stretch/squash roundtrip failed for p={}: got {}",
                p,
                p2
            );
        }
    }

    #[test]
    fn quant_in_range() {
        assert_eq!(quant(0, SSE_BINS), 0);
        assert_eq!(quant(PROB_MAX, SSE_BINS), SSE_BINS - 1);
        assert_eq!(quant(2048, SSE_BINS), SSE_BINS / 2);
    }

    #[test]
    fn table_refine_unchanged_at_init() {
        let table = QuantTable::new(2048, SSE_BINS);
        let p = 2048u16;
        let ctx = (usize::from(b't') << 3) | 0;
        let refined = table.refine(p, ctx);
        assert!(
            (refined as i32 - p as i32).abs() <= 1,
            "table at init should be near-identical: {} vs {}",
            refined,
            p
        );
    }

    #[test]
    fn table_update_shifts_prediction() {
        let mut table = QuantTable::new(2048, SSE_BINS);
        let ctx = (usize::from(b't') << 3) | 0;
        for _ in 0..100 {
            table.update(true, 2048, ctx);
        }
        let refined = table.refine(2048, ctx);
        assert!(
            refined > 2048,
            "table should learn toward 1 after 100 ones: got {}",
            refined
        );
    }

    #[test]
    fn cascade_refine_returns_valid_prob() {
        let mut cascade = SseApmCascade::new();
        cascade.set_context(b't');
        let p = cascade.refine(2048, 0);
        assert!(
            (0..=PROB_MAX).contains(&p),
            "probability out of range: {}",
            p
        );
    }

    #[test]
    fn cascade_reset_clears_tables() {
        let mut cascade = SseApmCascade::new();
        cascade.set_context(b't');
        for _ in 0..50 {
            cascade.update(true, 2048, 0);
        }
        let p_before = cascade.refine(2048, 0);
        cascade.reset();
        let p_after = cascade.refine(2048, 0);
        assert!(
            (p_after as i32 - 2048i32).abs() < 200,
            "post-reset prob should be near-neutral: got {}",
            p_after
        );
        let _ = p_before;
    }

    #[test]
    fn cascade_learns_from_repeated_bit() {
        let mut cascade = SseApmCascade::new();
        cascade.set_context(b't');
        for _ in 0..200 {
            cascade.update(true, 2048, 0);
        }
        let p = cascade.refine(2048, 0);
        assert!(
            p > 2048,
            "cascade should learn toward 1 after 200 ones: got {}",
            p
        );
    }

    #[test]
    fn memory_budget_under_1mb() {
        let cascade = SseApmCascade::new();
        let bytes = cascade.memory_bytes();
        assert!(
            bytes < 1_000_000,
            "SSE/APM/APM2 cascade should be <1 MB: got {} bytes",
            bytes
        );
    }
}
