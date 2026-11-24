//! Online logistic mixer — the novel core of `rcn`.
//!
//! Each base model emits a 12-bit probability of bit==1. The mixer fuses them with a
//! weighted sum in the logistic domain (stretch → linear combination → squash) and adapts
//! the weights online via stochastic gradient descent on the logistic loss. This is the
//! "context mixing" step that lets heterogeneous models (order, sparse, exec, LZP) cover
//! for each other per bit, rather than picking a single best model.
//!
//! For higher compression, see [`super::sse_apm::SseApmCascade`] which adds SSE/APM/APM2
//! refinement stages after the mixer, and [`super::mixer_bank::MixerBank`] which selects
//! from 8192 context-specific mixer instances.
//!
//! ## Fixed-point arithmetic (Q16 weights, Q10 stretch)
//!
//! The hot loop is integer-only: base weights and per-bit-position deltas are stored as
//! `i32` in Q16 (1.0 = 65536), the stretch table is `i16` in Q10 (logits × 1024), the
//! dot product accumulates in `i64`, and the squash is a float clamp into a 4096-entry
//! table (unchanged). This avoids f32 FMA entirely in the prediction and update loops.
//!
//! Q16 was chosen over Q8 because SGD weight updates per bit are small (~0.01 in float
//! units); Q8 rounded these to zero, killing learning. Q16 gives ~640 integer units per
//! step (0.01 × 65536), preserving gradient fidelity.

use super::BitModel;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Base-weight fixed-point scale: 1.0 = `1 << WEIGHT_Q`. Range ≈ ±128 (i32).
pub const WEIGHT_Q: i32 = 16;
/// Stretch-table scale: 1.0 logit = `1 << STRETCH_Q`.
pub const STRETCH_Q: i32 = 10;
/// Combined exponent: acc_q = Σ w_q16 · stretch_q10 = acc_float · 2^26.
pub const ACC_SHIFT: i32 = WEIGHT_Q + STRETCH_Q;
/// Fixed-point init value of a base weight (1.0).
pub const W_INIT: i32 = 1 << WEIGHT_Q;
/// Per-bit grad scaling: grad_q16 = lr·scale·err·stretch_q10 · 2^(WEIGHT_Q - STRETCH_Q).
pub const GRAD_SCALE: i32 = WEIGHT_Q - STRETCH_Q; // = 6

/// Integer squash constants: map fixed-point accumulator (Q26) to squash table index [0, 4095].
/// idx = (acc_q + 7*2^26) * 4095 / (14 * 2^26)
const SQUASH_SCALE_NUM: i64 = 4095;
const SQUASH_SCALE_DEN: i64 = 14 << ACC_SHIFT; // 14 * 2^26
const SQUASH_OFFSET: i64 = 7 << ACC_SHIFT; // 7 * 2^26

/// Shared stretch/squash tables (12-bit probability ↔ logit).
/// These are identical for every mixer instance, so we allocate once globally.
static STRETCH_Q10: std::sync::OnceLock<[i16; 4096]> = std::sync::OnceLock::new();
static SQUASH: std::sync::OnceLock<[u16; 4096]> = std::sync::OnceLock::new();

/// Logit (× 1024, i16) of `pr ∈ [0,1]`.
fn logit_i16(pr: f32) -> i16 {
    let scale = (1u32 << STRETCH_Q) as f32;
    let l = if pr <= 1e-6 {
        -7.0
    } else if pr >= 1.0 - 1e-6 {
        7.0
    } else {
        (pr / (1.0 - pr)).ln()
    };
    (l * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn stretch_table() -> &'static [i16; 4096] {
    STRETCH_Q10.get_or_init(|| {
        let mut t = [0i16; 4096];
        for (p_slot, slot) in t.iter_mut().enumerate() {
            let pr = (p_slot as f32 + 0.5) / 4096.0;
            *slot = logit_i16(pr);
        }
        t
    })
}

fn squash_table() -> &'static [u16; 4096] {
    SQUASH.get_or_init(|| {
        let mut t = [0u16; 4096];
        for (x_slot, slot) in t.iter_mut().enumerate() {
            let v = x_slot as f32 / 4095.0 * 14.0 - 7.0; // map [0,4095] -> [-7, 7]
            let s = 1.0 / (1.0 + (-v).exp());
            *slot = (s * MAX_PROB as f32).clamp(MIN_PROB as f32, MAX_PROB as f32) as u16;
        }
        t
    })
}

/// Logistic mixer over `n` model probabilities, with per-bit-position context.
///
/// The bit position within a byte (0 = MSB, 7 = LSB) is a cheap, decoder-safe signal:
/// text bytes have very different bit distributions per position (e.g. ASCII high
/// bits are nearly always 0), so conditioning the mixer weights on `bit_pos`
/// lets it specialize without changing the container format.
pub struct LogisticMixer {
    /// Base weights (per model), fixed-point Q16.
    weights: Vec<i32>,
    lr: f32,
    lr_scales: Vec<f32>,
    // Per (model, bit_position) weight deltas, fixed-point Q16.
    // The effective weight for model `i` at bit position `b` is `base[i] + pos_weights[i][b]`.
    pos_weights: Vec<[i32; 8]>,
    // Adam state (kept for API compat; not used in default stacks).
    adam_t: u32,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl LogisticMixer {
    /// Create a mixer for `n` models, with learning rate `lr`.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // builds Vecs via loops; not const-evaluable
    pub fn new(n: usize) -> Self {
        Self {
            // Start each weight at 1.0 so the mix is a sensible average of the
            // (stretched) model probabilities from the first bit. A zero
            // initialization makes `mix` return 50/50 until SGD slowly learns to
            // trust the models — which costs most of the compression on small/early
            // blocks. Positive weights also keep the mix grounded in the models'
            // evidence rather than the prior.
            weights: vec![W_INIT; n],
            // All models start with lr_scale=1.0 (SGD).
            lr_scales: vec![1.0; n],
            // Position deltas start at 0 so the initial mix is identical to the
            // non-context-aware version (pure 1.0 base weights).
            pos_weights: (0..n).map(|_| [0i32; 8]).collect(),
            lr: 0.02,
            // Adam state (unused in default SGD path).
            adam_t: 0,
            adam_m: vec![0.0; n],
            adam_v: vec![0.0; n],
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Create a mixer with Adam optimizer.
    ///
    /// NOTE: Adam is kept for API compatibility only. The hot path is fixed-point
    /// SGD; `new_adam` produces an SGD-trained mixer with the requested `lr`
    /// (Adam's adaptive rates were never measurably better on the default stacks,
    /// see DEVELOPMENT.md A/B history — "Second-order mixer training" row).
    #[must_use]
    pub fn new_adam(n: usize, lr: f32) -> Self {
        Self {
            weights: vec![W_INIT; n],
            lr_scales: vec![1.0; n],
            pos_weights: (0..n).map(|_| [0i32; 8]).collect(),
            lr,
            adam_t: 0,
            adam_m: vec![0.0; n],
            adam_v: vec![0.0; n],
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Set the learning rate (default 0.02).
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Set per-model learning rate scale. Index corresponds to model position in the stack.
    pub fn set_lr_scale(&mut self, idx: usize, scale: f32) {
        if idx < self.lr_scales.len() {
            self.lr_scales[idx] = scale;
        }
    }

    /// Set all per-model learning rate scales.
    pub fn set_lr_scales(&mut self, scales: Vec<f32>) {
        if scales.len() == self.lr_scales.len() {
            self.lr_scales = scales;
        }
    }

    /// Return a copy of the per-model learning rate scales.
    #[must_use]
    pub fn lr_scales(&self) -> Vec<f32> {
        self.lr_scales.clone()
    }

    /// Replace base weights. Keeps per-bit-position deltas unchanged.
    pub fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights
            .into_iter()
            .map(|w| (w * (1 << WEIGHT_Q) as f32).round().clamp(i32::MIN as f32, i32::MAX as f32) as i32)
            .collect();
    }

    /// Return a copy of the base weights (as floats).
    #[must_use]
    pub fn weights(&self) -> Vec<f32> {
        self.weights
            .iter()
            .map(|&w| w as f32 / (1 << WEIGHT_Q) as f32)
            .collect()
    }

    /// Reset weights to initial state (called at block boundaries).
    ///
    /// NOTE: Prefer `decay` for cross-block continuity — it shrinks weights
    /// toward their init value (1.0) instead of hard-clearing, preserving
    /// learned structure across block boundaries.
    pub fn reset(&mut self) {
        self.weights.fill(W_INIT);
        for pw in &mut self.pos_weights {
            pw.fill(0);
        }
        self.lr_scales.fill(1.0);
        self.lr = 0.02;
        self.adam_t = 0;
        self.adam_m.fill(0.0);
        self.adam_v.fill(0.0);
    }

    /// Decay all learned weights toward their init values by `factor`.
    ///
    /// Base weights decay toward 1.0, pos_weights decay toward 0.0.
    /// `weight = init + (weight - init) * f`
    ///
    /// A factor of 0.0 restores init; 1.0 leaves unchanged. This preserves
    /// learned structure across block boundaries without hard-clearing, which
    /// would throw away the per-context weight vectors that the 8192-bank
    /// hierarchy depends on.
    pub fn decay(&mut self, factor: f32) {
        let f = factor;
        for w in &mut self.weights {
            let nf = *w as f32 * f + W_INIT as f32 * (1.0 - f);
            *w = nf.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        }
        for pw in &mut self.pos_weights {
            for dw in pw.iter_mut() {
                let nf = *dw as f32 * f;
                *dw = nf.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            }
        }
    }

    #[inline]
    fn stretch_of(&self, p: u16) -> i16 {
        stretch_table()[(p as usize).clamp(1, 4095)]
    }

    #[inline]
    fn squash_of(&self, acc_q: i64) -> u16 {
        // Integer arithmetic: acc_float = acc_q / 2^26
        // Table index = (acc_float + 7) / 14 * 4095
        // = (acc_q/2^26 + 7) / 14 * 4095
        // = (acc_q + 7*2^26) / 14 * 4095 / 2^26
        // = (acc_q + 7*2^26) * 4095 / (14 * 2^26)
        // 14 * 2^26 = 939524096
        // 7 * 2^26 = 469762048
        let idx = ((acc_q + SQUASH_OFFSET) * SQUASH_SCALE_NUM / SQUASH_SCALE_DEN).clamp(0, 4095) as usize;
        squash_table()[idx]
    }

    /// Mix `probs` (one P(bit==1) per model, each in `[1,4095]`) → fused P in `[1,4095]`.
    /// `bit_pos` is the 0-based MSB-first bit position within the current byte.
    ///
    /// This version computes stretch values on-the-fly in a single loop.
    #[must_use]
    #[inline(always)]
    pub fn mix(&self, probs: &[u16], bit_pos: u8) -> u16 {
        self.mix_acc(probs, bit_pos).1
    }

    /// Compute the logistic accumulator AND the squashed probability in one pass.
    ///
    /// Returns `(acc, q)` with `acc = Σ w_i·stretch(p_i)` (pre-squash, fixed-point)
    /// and `q` the squashed probability in `[1,4095]`. The caller that is about to
    /// call [`update`](Self::update) for the *same* (`probs`, `bit_pos`) can reuse
    /// the returned `acc` instead of letting `update` recompute the dot product.
    ///
    /// Single-pass: computes stretch and accumulates in one loop.
    #[must_use]
    #[inline(always)]
    pub fn mix_acc(&self, probs: &[u16], bit_pos: u8) -> (i64, u16) {
        let b = usize::from(bit_pos.min(7));
        let mut acc: i64 = 0;
        let stretch_lut = stretch_table();
        
        for (i, &p) in probs.iter().enumerate() {
            let stretch = stretch_lut[(p as usize).clamp(1, 4095)];
            let w = self.weights[i] + self.pos_weights[i][b];
            acc += i64::from(w) * i64::from(stretch);
        }
        (acc, self.squash_of(acc))
    }

    /// Online update after the true `bit` is known.
    ///
    /// Returns the predicted probability `q` (the squashed accumulator with the
    /// *pre-update* weights) that was used for this bit. Callers that blend this
    /// mixer's output into a higher-level mixer (the master) can then reuse `q`
    /// instead of re-mixing — the entire prediction is a single dot product.
    pub fn update(&mut self, probs: &[u16], bit: bool, bit_pos: u8) -> u16 {
        let acc = self.mix_acc(probs, bit_pos).0;
        self.update_from_acc(probs, bit, bit_pos, acc)
    }

    /// Same as [`update`](Self::update), but the logistic accumulator `acc`
    /// (from [`mix_acc`](Self::mix_acc) with the same inputs) is supplied by the
    /// caller so the dot product is not recomputed a second time.
    ///
    /// Single-pass: computes stretch values on-the-fly in a single loop.
    pub fn update_from_acc(&mut self, probs: &[u16], bit: bool, bit_pos: u8, acc: i64) -> u16 {
        let b = usize::from(bit_pos.min(7));
        let target = if bit { 1.0f32 } else { 0.0 };
        // pred from the precomputed squash table — no exp() in the hot loop.
        let q = self.squash_of(acc);
        let pred = f32::from(q) / 4095.0;
        let err = target - pred;

        // Grad in weight units (Q16) from a stretch in Q10:
        //   grad_q16 = lr·scale·err·stretch_q10 · 2^(WEIGHT_Q - STRETCH_Q)
        // WEIGHT_Q - STRETCH_Q = 6, so scale by 64.
        let stretch_lut = stretch_table();
        for (i, &p) in probs.iter().enumerate() {
            let scale = self.lr_scales[i];
            let stretch_q10 = i32::from(stretch_lut[(p as usize).clamp(1, 4095)]);
            // lr·scale·err is f32; multiply by stretch_q10·64, then round to i32.
            let delta = (self.lr * scale * err * stretch_q10 as f32 * (1 << GRAD_SCALE) as f32).round();
            let d = delta.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            self.weights[i] += d;
            self.pos_weights[i][b] += d;
        }

        q
    }

    // -----------------------------------------------------------------------
    // SoA (Structure-of-Arrays) static methods for MixerBank flat-array layout
    // -----------------------------------------------------------------------

    /// Mix using SoA flat arrays: `weights[bank_id*n_models + i]` for model `i`.
    /// Returns `(acc, q, stretches)` where `stretches` are the Q10 stretch LUT
    /// values for `probs` (reused by [`update_from_acc_from_flat`]).
    #[must_use]
    #[inline(always)]
    pub fn mix_acc_from_flat(
        probs: &[u16], bit_pos: u8,
        weights: &[i32], pos_weights: &[[i32; 8]],
        lr_scales: &[f32], n_models: usize, base: usize,
    ) -> (i64, u16, [i16; 16]) {
        let b = usize::from(bit_pos.min(7));
        let mut acc: i64 = 0;
        let mut stretches = [0i16; 16];
        let stretch_lut = stretch_table();
        for (i, &p) in probs.iter().enumerate() {
            let stretch = stretch_lut[(p as usize).clamp(1, 4095)];
            stretches[i] = stretch;
            let w = weights[base + i] + pos_weights[base + i][b];
            acc += i64::from(w) * i64::from(stretch);
        }
        let q = squash_table()[(((acc + SQUASH_OFFSET) * SQUASH_SCALE_NUM / SQUASH_SCALE_DEN).clamp(0, 4095)) as usize];
        (acc, q, stretches)
    }

    /// Update using SoA flat arrays. Returns the pre-update squashed probability.
    #[must_use]
    #[inline(always)]
    pub fn update_from_flat(
        weights: &mut [i32], pos_weights: &mut [[i32; 8]],
        lr_scales: &[f32], n_models: usize, base: usize,
        probs: &[u16], bit: bool, bit_pos: u8,
    ) -> u16 {
        let b = usize::from(bit_pos.min(7));
        let target = if bit { 1.0f32 } else { 0.0 };
        let mut acc: i64 = 0;
        let mut stretches = [0i16; 16];
        let stretch_lut = stretch_table();
        for (i, &p) in probs.iter().enumerate() {
            let stretch = stretch_lut[(p as usize).clamp(1, 4095)];
            stretches[i] = stretch;
            let w = weights[base + i] + pos_weights[base + i][b];
            acc += i64::from(w) * i64::from(stretch);
        }
        let q = squash_table()[(((acc + SQUASH_OFFSET) * SQUASH_SCALE_NUM / SQUASH_SCALE_DEN).clamp(0, 4095)) as usize];
        let pred = f32::from(q) / 4095.0;
        let err = target - pred;
        for (i, &_p) in probs.iter().enumerate() {
            let scale = lr_scales[base + i];
            let stretch_q10 = i32::from(stretches[i]);
            let delta = (0.02 * scale * err * stretch_q10 as f32 * (1 << GRAD_SCALE) as f32).round();
            let d = delta.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            weights[base + i] += d;
            pos_weights[base + i][b] += d;
        }
        q
    }

    /// Update using SoA flat arrays with precomputed accumulator and stretch LUT.
    #[must_use]
    #[inline(always)]
    pub fn update_from_acc_from_flat(
        probs: &[u16], bit: bool, bit_pos: u8, acc: i64,
        weights: &mut [i32], pos_weights: &mut [[i32; 8]],
        lr_scales: &[f32], n_models: usize, base: usize,
        stretches: &[i16; 16],
    ) -> u16 {
        let b = usize::from(bit_pos.min(7));
        let target = if bit { 1.0f32 } else { 0.0 };
        let q = squash_table()[(((acc + SQUASH_OFFSET) * SQUASH_SCALE_NUM / SQUASH_SCALE_DEN).clamp(0, 4095)) as usize];
        let pred = f32::from(q) / 4095.0;
        let err = target - pred;
        for (i, &_p) in probs.iter().enumerate() {
            let scale = lr_scales[base + i];
            let stretch_q10 = i32::from(stretches[i]);
            let delta = (0.02 * scale * err * stretch_q10 as f32 * (1 << GRAD_SCALE) as f32).round();
            let d = delta.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            weights[base + i] += d;
            pos_weights[base + i][b] += d;
        }
        q
    }
}

impl BitModel for LogisticMixer {
    fn predict(&self) -> u16 {
        // Only meaningful when driven via `mix`; standalone predict returns mid.
        2048
    }

    fn update(&mut self, _bit: bool) {
        // The mixer is updated via `update(probs, bit)`, not this trait method.
    }

    fn reset(&mut self) {
        self.weights.fill(W_INIT);
        for pw in &mut self.pos_weights {
            pw.fill(0);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::order::OrderN;

    #[test]
    fn mixer_favors_correct_model() {
        // Biased data: feed bytes that are mostly 1-bits (0xFF 3 of every 4 steps).
        // The linear logistic mix learns the bias, so the fused prob should track it.
        let mut m0 = OrderN::new(0);
        let mut m1 = OrderN::new(0);
        let mut mixer = LogisticMixer::new(2);
        let mut last_probs = [2048u16; 2];
        for step in 0..200 {
            let byte: u8 = if step % 4 != 0 { 0xFF } else { 0x00 };
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1 == 1;
                let bit_pos = bit_idx as u8;
                last_probs[0] = m0.predict();
                last_probs[1] = m1.predict();
                let fused = mixer.mix(&last_probs, bit_pos);
                m0.update(bit);
                m1.update(bit);
                mixer.update(&last_probs, bit, bit_pos);
                if step > 150 && bit {
                    assert!(i32::from(fused) > 2048, "mixer should learn the bias");
                }
            }
        }
    }

    #[test]
    fn mix_in_range() {
        let mixer = LogisticMixer::new(3);
        let p = mixer.mix(&[1000, 2048, 3000], 0);
        assert!((1..=4095).contains(&p));
    }

    #[test]
    fn weights_are_q16() {
        let mixer = LogisticMixer::new(2);
        assert_eq!(mixer.weights, vec![65536, 65536]);
        assert_eq!(mixer.weights(), vec![1.0, 1.0]);
    }

    #[test]
    fn decay_preserves_init() {
        let mut mixer = LogisticMixer::new(2);
        mixer.weights[0] = 0; // moved toward -1.0
        mixer.decay(0.0);
        assert_eq!(mixer.weights[0], W_INIT);
    }

    #[test]
    fn squash_and_stretch_consistent() {
        // squash(stretch(p)) ≈ p for mid probabilities.
        let mixer = LogisticMixer::new(1);
        for p in [1024u16, 2048, 3072] {
            let back = mixer.mix(&[p], 0);
            let expected = p;
            let diff = i32::from(back) - i32::from(expected);
            assert!(
                diff.abs() <= 800,
                "squash∘stretch(p)={back} vs p={p} drifted too far"
            );
        }
    }

    #[test]
    fn fixed_point_bank_end_to_end() {
        // Exercise the exact flow `MixerBank` uses (mix_acc → update_from_acc)
        // and check it stays deterministic and learns.
        let mut mixer = LogisticMixer::new(2);
        let mut acc_hist = Vec::new();
        for _ in 0..100 {
            let (acc, _q) = mixer.mix_acc(&[2000, 3000], 0);
            acc_hist.push(acc);
            mixer.update_from_acc(&[2000, 3000], true, 0, acc);
        }
        let (_acc, q) = mixer.mix_acc(&[2000, 3000], 0);
        assert!(q > 2048, "mixer should learn toward 1 after 100 ones, got {q}");
    }
}