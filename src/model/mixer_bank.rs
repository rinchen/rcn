//! Two-level context-mixing hierarchy: bank mixers → global mixer → master mixer.
//!
//! CMIX / PAQ8 use a multi-layer mixer stack:
//!   1. **Bank mixers** (16384 instances): each selected by a context hash of
//!      order-1 / order-2 / word-context + bit position. One logistic mixer
//!      saturates at ~50% on repetitive corpora (dickens); per-context banks
//!      avoid this by specializing weights per context.
//!   2. **Global mixer**: a single shared logistic mixer over the same models,
//!      providing a context-agnostic fallback prediction.
//!   3. **Master mixer**: blends `[p_bank, p_global, p_lzp]` in logistic space.
//!      Only the master + the selected bank are trained per bit — never all 16384.
//!
//! Cross-block persistence: bank and master weights are **decayed** (not reset)
//! at block boundaries, preserving learned structure across the stream.
//!
//! Memory: 16384 mixers × 8 models × (1 base + 8 pos) × 4 bytes ≈ 4.5 MB for the
//! banks, plus 2 mixers (global + master).

use crate::model::mixer::{LogisticMixer, W_INIT};
use crate::model::ByteAssembler;

/// Number of mixer instances in the bank. 16384 gives 14 bits of context selection.
pub const NUM_MIXERS: usize = 16384;

/// Byte classes for mixer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteClass {
    Space = 0,
    Vowel = 1,
    Consonant = 2,
    Digit = 3,
    Punct = 4,
    Other = 5,
}

impl ByteClass {
    /// Classify a byte into one of six text-relevant classes.
    #[must_use]
    pub fn classify(b: u8) -> Self {
        match b {
            b' ' | b'\t' | b'\n' | b'\r' => ByteClass::Space,
            b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U' => ByteClass::Vowel,
            b'0'..=b'9' => ByteClass::Digit,
            b'.' | b',' | b';' | b':' | b'!' | b'?' | b'"' | b'\'' | b'-' | b'(' | b')' | b'['
            | b']' | b'{' | b'}' => ByteClass::Punct,
            _ if b.is_ascii_alphabetic() => ByteClass::Consonant,
            _ => ByteClass::Other,
        }
    }
}

/// Compute the bank ID from context.
///
/// Uses a 14-bit hash:
/// `(byte_class << 11) | (bit_pos << 8) | (order1 << 5) | (order2 << 2) | (word_hash & 3)`
///
/// The bit position is folded into the bank ID so each (context, bit-position) pair
/// gets its own weight vector — this is why the per-bit-position win worked:
/// text bytes have very different bit distributions per position.
///
/// `order2_byte` is the byte before `order1_byte`, providing two levels of
/// backward context for bank selection. Two bits of `word_hash` use the extra
/// bank bit relative to the older 13-bit / 8192 layout.
pub fn mixer_id(
    byte_class: ByteClass,
    bit_pos: u8,
    order1_byte: u8,
    order2_byte: u8,
    word_hash: usize,
) -> usize {
    let class_bits = (byte_class as usize) & 0x7;
    let bp = (bit_pos as usize) & 0x7;
    let o1 = (order1_byte as usize) & 0x7;
    let o2 = (order2_byte as usize) & 0x7;
    let wh = word_hash & 0x3;
    ((class_bits << 11) | (bp << 8) | (o1 << 5) | (o2 << 2) | wh) & 0x3FFF
}

/// Reusable per-bit accumulator state from [`MixerBank::mix_acc`].
///
/// Carries the three dot products (bank / global / master), squashed
/// bank/global/lzp inputs to the master, and the stretch LUT values used for
/// the bank mix so [`MixerBank::update_acc`] can skip re-looking them up.
#[derive(Clone, Copy)]
pub struct MixerAcc {
    acc_bank: i64,
    acc_global: i64,
    acc_master: i64,
    master_probs: [u16; 3],
    /// Stretch(Q10) of each base-model probability used for the bank mix (S11).
    bank_stretches: [i16; 16],
    n_stretches: u8,
}

/// Two-level mixer: 16384 bank mixers + a global mixer + a master mixer.
///
/// - `mixers`: context-specific bank selected by `mixer_id`. Only the selected
///   bank is trained per bit.
/// - `global_mixer`: context-agnostic fallback over the same models.
/// - `master_mixer`: blends `[p_bank, p_global, p_lzp_conf]` (3 inputs).
///
/// Encoder and decoder must call `mix`/`update`/`push_byte` with identical
/// inputs and in the same order. The `ByteAssembler` tracks the byte stream
/// for context hashing; it is deterministic from the coded bit stream, so
/// both sides stay in sync.
pub struct MixerBank {
    /// Bank mixers in SoA (Structure of Arrays) layout for contiguous memory access:
    /// - `bank_weights`: all base weights, size NUM_MIXERS * n_models
    /// - `bank_pos_weights`: all position weights, size NUM_MIXERS * n_models (as [i32; 8])
    /// - `bank_lr_scales`: all learning rate scales, size NUM_MIXERS * n_models
    /// - `bank_lr`: shared learning rate for all bank mixers
    bank_weights: Vec<i32>,
    bank_pos_weights: Vec<[i32; 8]>,
    bank_lr_scales: Vec<f32>,
    bank_lr: f32,
    /// Global context-agnostic mixer (same n_models).
    global_mixer: LogisticMixer,
    /// Master mixer: blends bank + global + lzp_conf (3 inputs).
    master_mixer: LogisticMixer,
    n_models: usize,
    /// Tracks byte history for bank context selection.
    asm: ByteAssembler,
}

/// Prefetch SoA bank weights for the upcoming bank base index (S11).
#[inline(always)]
fn prefetch_bank_weights(weights: &[i32], base: usize) {
    if base >= weights.len() {
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "no_avx2")))]
    {
        // Soft hint: next bank's weight run will be touched on the next bit.
        // SAFETY: pointer is within `weights` (checked above).
        unsafe {
            core::arch::x86_64::_mm_prefetch(
                weights.as_ptr().add(base) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(feature = "no_avx2"))))]
    {
        let _ = (weights, base);
    }
}

impl MixerBank {
    /// Create a two-level mixer hierarchy.
    ///
    /// `n_models` is the number of base bit-model probabilities fed into each
    /// bank and global mixer. The master mixer takes 3 inputs (bank, global,
    /// lzp_confidence).
    #[must_use]
    pub fn new(n_models: usize) -> Self {
        let total = NUM_MIXERS * n_models;
        Self {
            bank_weights: vec![W_INIT; total],
            bank_pos_weights: vec![[0i32; 8]; total],
            bank_lr_scales: vec![1.0; total],
            bank_lr: 0.02,
            global_mixer: LogisticMixer::new(n_models),
            master_mixer: LogisticMixer::new(3), // bank, global, lzp_conf
            n_models,
            asm: ByteAssembler::new(8),
        }
    }

    /// Set a per-model learning rate scale (index corresponds to model position).
    pub fn set_lr_scale(&mut self, idx: usize, scale: f32) {
        if idx < self.n_models {
            // Bank mixers: flat array at stride n_models
            for i in (idx..self.bank_lr_scales.len()).step_by(self.n_models) {
                self.bank_lr_scales[i] = scale;
            }
            self.global_mixer.set_lr_scale(idx, scale);
        }
    }

    /// Number of base models.
    #[must_use]
    pub fn n_models(&self) -> usize {
        self.n_models
    }

    /// Compute the bank ID for the current context.
    #[inline]
    fn current_bank_id(&self, bit_pos: u8) -> usize {
        let order1 = self.asm.prev_byte();
        let order2 = self.asm.prev_prev_byte();
        let byte_class = if self.asm.has_byte() {
            ByteClass::classify(self.asm.last_byte())
        } else {
            ByteClass::Other
        };
        mixer_id(byte_class, bit_pos, order1, order2, self.asm.word_hash())
    }

    /// Mix the model probabilities and return a fused 12-bit P(bit==1).
    ///
    /// `probs` — one probability per base model (in `[1,4095]`).
    /// `bit_pos` — MSB-first bit position within the current byte (0..7).
    /// `lzp_conf` — the LZP model's current prediction (high-confidence match
    ///   signal), used as the third input to the master mixer.
    #[must_use]
    #[inline(always)]
    pub fn mix(&self, probs: &[u16], bit_pos: u8, lzp_conf: u16) -> u16 {
        self.mix_acc(probs, bit_pos, lzp_conf).0
    }

    /// Single-pass mix: compute the master prediction AND the per-mixer
    /// accumulators for reuse by [`update_acc`](Self::update_acc).
    ///
    /// Returns `(master_prediction, acc)` where `acc` holds the bank/global/
    /// master dot products computed here with the *pre-update* weights. The
    /// caller runs its rANS step on `master_prediction`, then calls
    /// [`update_acc`](Self::update_acc) with `acc` — avoiding a second set of
    /// dot products per bit (previously `mix` then `update` recomputed them).
    #[must_use]
    #[inline(always)]
    pub fn mix_acc(&self, probs: &[u16], bit_pos: u8, lzp_conf: u16) -> (u16, MixerAcc) {
        let bank_id = self.current_bank_id(bit_pos);
        // SoA: bank_id * n_models gives the offset into the flat arrays
        let base = bank_id * self.n_models;

        // S11: prefetch the next bit-position's bank weights while we mix this bit.
        // bit_pos is MSB-first 7..0 within a byte; next is bit_pos-1 (or 7 after push).
        let next_bit_pos = if bit_pos > 0 { bit_pos - 1 } else { 7 };
        let next_id = self.current_bank_id(next_bit_pos);
        let next_base = next_id * self.n_models;
        prefetch_bank_weights(&self.bank_weights, next_base);

        let (acc_bank, q_bank, bank_stretches) = LogisticMixer::mix_acc_from_flat(
            probs, bit_pos, &self.bank_weights, &self.bank_pos_weights,
            &self.bank_lr_scales, self.n_models, base,
        );
        let (acc_global, q_global) = self.global_mixer.mix_acc(probs, bit_pos);
        let master_probs = [q_bank, q_global, lzp_conf];
        let (acc_master, p) = self.master_mixer.mix_acc(&master_probs, 0);
        (
            p,
            MixerAcc {
                acc_bank,
                acc_global,
                acc_master,
                master_probs,
                bank_stretches,
                n_stretches: probs.len().min(16) as u8,
            },
        )
    }

    /// Train the selected bank, the global mixer, and the master using the
    /// accumulators from [`mix_acc`](Self::mix_acc) with the same inputs.
    ///
    /// Identical to `update` but reuses the precomputed (`acc_bank`,
    /// `acc_global`, `acc_master`) and bank stretch LUT rather than recomputing.
    #[inline(always)]
    pub fn update_acc(&mut self, probs: &[u16], bit: bool, bit_pos: u8, acc: MixerAcc) {
        let bank_id = self.current_bank_id(bit_pos);
        let base = bank_id * self.n_models;
        let _ = LogisticMixer::update_from_acc_from_flat(
            probs, bit, bit_pos, acc.acc_bank, &mut self.bank_weights, &mut self.bank_pos_weights,
            &self.bank_lr_scales, self.n_models, base, &acc.bank_stretches,
        );
        let _ = acc.n_stretches; // documented for API clarity; stretches array is fixed-16
        self.global_mixer.update_from_acc(probs, bit, bit_pos, acc.acc_global);
        self.master_mixer
            .update_from_acc(&acc.master_probs, bit, 0, acc.acc_master);
    }

    /// Encoder-side single call: mix, run the encoded bit against the fused
    /// prediction via `encode`, then train all three mixers in one pass.
    ///
    /// The bit is known at encode time, so the acc can be fed straight through.
    pub fn mix_and_update(
        &mut self,
        probs: &[u16],
        bit: bool,
        bit_pos: u8,
        lzp_conf: u16,
        encode: &mut impl FnMut(bool, u16),
    ) {
        let (p, acc) = self.mix_acc(probs, bit_pos, lzp_conf);
        encode(bit, p);
        self.update_acc(probs, bit, bit_pos, acc);
    }

    /// Train the selected bank mixer, the global mixer, and the master mixer.
    ///
    /// Only three mixers are touched per bit: the context-selected bank, the
    /// global, and the master. Not all 16384. This is the key performance
    /// property of the two-level hierarchy.
    ///
    /// Returns the master mixer's predicted probability for this bit.
    #[inline]
    pub fn update(&mut self, probs: &[u16], bit: bool, bit_pos: u8, lzp_conf: u16) -> u16 {
        let bank_id = self.current_bank_id(bit_pos);
        let base = bank_id * self.n_models;
        let q_bank = LogisticMixer::update_from_flat(
            &mut self.bank_weights, &mut self.bank_pos_weights,
            &self.bank_lr_scales, self.n_models, base,
            probs, bit, bit_pos,
        );
        let q_global = self.global_mixer.update(probs, bit, bit_pos);
        self.master_mixer.update(&[q_bank, q_global, lzp_conf], bit, 0)
    }

    /// Feed a completed byte so byte-history context advances.
    #[inline]
    pub fn push_byte(&mut self, byte: u8) {
        self.asm.push_byte(byte);
    }

    /// Decay all mixer weights toward init (1.0) by `factor`.
    pub fn decay(&mut self, factor: f32) {
        for w in &mut self.bank_weights {
            let nf = *w as f32 * factor + W_INIT as f32 * (1.0 - factor);
            *w = nf.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        }
        for pw in &mut self.bank_pos_weights {
            for dw in pw.iter_mut() {
                let nf = *dw as f32 * factor;
                *dw = nf.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            }
        }
        self.global_mixer.decay(factor);
        self.master_mixer.decay(factor);
    }

    /// Hard reset (kept for API compatibility, but `decay` is preferred).
    pub fn reset(&mut self) {
        for w in &mut self.bank_weights {
            *w = W_INIT;
        }
        for pw in &mut self.bank_pos_weights {
            pw.fill(0);
        }
        for ls in &mut self.bank_lr_scales {
            *ls = 1.0;
        }
        self.bank_lr = 0.02;
        self.global_mixer.reset();
        self.master_mixer.reset();
        self.asm.reset();
    }

    /// Approximate memory footprint in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.bank_weights.len() * std::mem::size_of::<i32>()
            + self.bank_pos_weights.len() * std::mem::size_of::<[i32; 8]>()
            + self.bank_lr_scales.len() * std::mem::size_of::<f32>()
            + 2 * std::mem::size_of::<LogisticMixer>()
    }
}

impl Default for MixerBank {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_id_in_range() {
        let id = mixer_id(ByteClass::Vowel, 3, b'a', b'b', 42);
        assert!(id < NUM_MIXERS, "mixer_id {} out of range", id);
    }

    #[test]
    fn mixer_id_deterministic() {
        let a = mixer_id(ByteClass::Space, 5, b'x', b'y', 99);
        let b = mixer_id(ByteClass::Space, 5, b'x', b'y', 99);
        assert_eq!(a, b);
    }

    #[test]
    fn bank_mix_returns_valid_prob() {
        let bank = MixerBank::new(2);
        let p = bank.mix(&[2000, 3000], 0, 2048);
        assert!((1..=4095).contains(&p), "probability out of range: {}", p);
    }

    #[test]
    fn bank_learns() {
        let mut bank = MixerBank::new(2);
        let probs = [2000u16, 3000u16];
        for _ in 0..200 {
            bank.update(&probs, true, 0, 2048);
        }
        let p = bank.mix(&probs, 0, 2048);
        // Master should have learned toward 1 after 200 ones.
        assert!(p > 2048, "master mixer should learn toward 1, got {}", p);
    }

    #[test]
    fn bank_decay_preserves_structure() {
        let mut bank = MixerBank::new(2);
        let probs = [2000u16, 3000u16];
        for _ in 0..200 {
            bank.update(&probs, true, 0, 2048);
        }
        let before = bank.mix(&probs, 0, 2048);
        bank.decay(0.99);
        let after = bank.mix(&probs, 0, 2048);
        // After decay, the prediction should still be biased (not reset to 2048).
        assert!(
            after > 2048,
            "decay should preserve learned structure: got {} (before {})",
            after,
            before
        );
    }

    #[test]
    fn bank_decay_zero_matches_reset() {
        let mut bank = MixerBank::new(2);
        let probs = [2000u16, 3000u16];
        for _ in 0..100 {
            bank.update(&probs, true, 0, 2048);
        }
        bank.decay(0.0);
        let p_decayed = bank.mix(&probs, 0, 2048);
        let fresh = MixerBank::new(2);
        let p_fresh = fresh.mix(&probs, 0, 2048);
        assert_eq!(
            p_decayed, p_fresh,
            "decay(0.0) should match fresh mixer: got {}, fresh {}",
            p_decayed, p_fresh
        );
    }

    #[test]
    fn push_byte_advances_context() {
        let mut bank = MixerBank::new(2);
        bank.push_byte(b'a');
        bank.push_byte(b'b');
        assert_eq!(bank.asm.last_byte(), b'b');
        assert_eq!(bank.asm.prev_byte(), b'a');
    }

    #[test]
    fn context_changes_bank_selection() {
        let bank = MixerBank::new(2);
        // Same bit_pos but different preceding byte → different bank_id.
        let id1 = bank.current_bank_id(0);
        let mut bank2 = MixerBank::new(2);
        bank2.push_byte(b'x');
        let id2 = bank2.current_bank_id(0);
        assert_ne!(
            id1, id2,
            "different byte context should select different bank"
        );
    }

    #[test]
    fn memory_under_6mb() {
        let bank = MixerBank::new(8);
        let bytes = bank.memory_bytes();
        // SoA layout: 16384 banks × 8 models × (4 + 32 + 4) ≈ 5.2MB for banks,
        // plus global/master mixers.
        assert!(
            bytes < 6_000_000,
            "MixerBank with 8 models should be <6 MB: got {} bytes",
            bytes
        );
    }
}
