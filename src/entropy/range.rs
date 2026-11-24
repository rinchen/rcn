//! Entropy backend for `rcn`: a **bit coder** built on the audited [`ans`] rANS crate.
//!
//! `rcn` models data bit-by-bit (the logistic mixer emits a probability per bit), so
//! this module exposes a stream of bits where each bit is entropy-coded against a 2-symbol
//! rANS frequency table derived from the predicted probability. Using `ans` for the
//! primitive keeps the entropy stage correct and battle-tested; the novel work lives in
//! the models + mixer + classifier, not in re-deriving rANS.
//!
//! Wire format (per `ans::RansEncoder`/`RansDecoder`): encoder feeds symbols in
//! **reverse** order, decoder reads forward, final state is the 4-byte LE tail.

use ans::{AnsError, FrequencyTable, RansDecoder, RansEncoder};
use std::sync::OnceLock;

const PRECISION: u32 = 12; // total = 1<<12 = 4096 frequency mass, 12-bit probabilities
const TOTAL: u32 = 1 << PRECISION;

/// Lazily-built cache of every possible 2-symbol rANS table (one per `p` in `[1, TOTAL-1]`).
/// Building a `FrequencyTable` per bit was the dominant cost (heap alloc + normalization on
/// every one of millions of bits); caching makes `bit_table` O(1).
fn table_cache() -> &'static Vec<FrequencyTable> {
    static CACHE: OnceLock<Vec<FrequencyTable>> = OnceLock::new();
    CACHE.get_or_init(|| {
        (1..TOTAL)
            .map(|p1| {
                let p0 = TOTAL - p1;
                FrequencyTable::from_counts(&[p0, p1], PRECISION).expect("bit table always valid")
            })
            .collect()
    })
}

/// Look up the precomputed 2-symbol table for a predicted P(bit==1) in `[1, TOTAL-1]`.
///
/// Symbol 1 (bit set) gets frequency `p`; symbol 0 gets the complement `TOTAL - p`.
#[inline]
#[must_use]
fn bit_table(p1: u16) -> &'static FrequencyTable {
    let idx = (p1.clamp(1, (TOTAL - 1) as u16) as usize) - 1;
    &table_cache()[idx]
}

/// Bit-level arithmetic encoder over rANS.
pub struct BitEncoder {
    pending_bits: Vec<u8>,    // packed bits (LSB first within each byte)
    pending_probs: Vec<u16>,  // corresponding probabilities
    bit_count: usize,
}

impl BitEncoder {
    /// Create a new bit encoder with capacity for `cap` bits.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        // Pack 8 bits per byte for bits, 1 u16 per bit for probs
        let bytes_needed = (cap + 7) / 8;
        Self {
            pending_bits: Vec::with_capacity(bytes_needed),
            pending_probs: Vec::with_capacity(cap),
            bit_count: 0,
        }
    }

    /// Create a new bit encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending_bits: Vec::new(),
            pending_probs: Vec::new(),
            bit_count: 0,
        }
    }

    /// Buffer one bit for encoding.
    ///
    /// `p` is the predicted probability of bit==1, in `[1, 4095]` (12-bit). Bits are
    /// fed to the underlying rANS encoder in reverse order, so they are buffered here
    /// and flushed by [`finish`](Self::finish).
    #[inline]
    pub fn encode_bit(&mut self, bit: bool, p: u16) {
        // Pack bit into pending_bits
        let byte_idx = self.bit_count >> 3;
        let bit_idx = self.bit_count & 7;
        if byte_idx >= self.pending_bits.len() {
            self.pending_bits.push(0);
        }
        if bit {
            self.pending_bits[byte_idx] |= 1 << bit_idx;
        }
        self.pending_probs.push(p);
        self.bit_count += 1;
    }

    /// Finalize, returning the entropy-coded byte stream.
    ///
    /// # Panics
    ///
    /// Panics only if the underlying rANS primitive fails on a valid 2-symbol table,
    /// which cannot happen for the tables constructed here.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        let mut enc = RansEncoder::new();
        // Feed in reverse: the last buffered bit is encoded first.
        for i in (0..self.bit_count).rev() {
            let byte_idx = i >> 3;
            let bit_idx = i & 7;
            let bit = (self.pending_bits[byte_idx] >> bit_idx) & 1 != 0;
            let p = self.pending_probs[i];
            let table = bit_table(p);
            enc.put(u32::from(bit), table)
                .expect("rANS put cannot fail for a valid 2-symbol table");
        }
        enc.finish()
    }
}

impl Default for BitEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Bit-level decoder (forward order) over rANS.
pub struct BitDecoder<'a> {
    dec: RansDecoder<'a>,
}

impl<'a> BitDecoder<'a> {
    /// Create a decoder over `data`, the byte stream from [`BitEncoder::finish`].
    ///
    /// # Errors
    ///
    /// Returns [`AnsError`] if the stream is too short (< 4 bytes) or its trailing
    /// final-state word is below the rANS lower bound.
    pub fn new(data: &'a [u8]) -> Result<Self, AnsError> {
        Ok(BitDecoder {
            dec: RansDecoder::new(data)?,
        })
    }

    /// Decode one bit given the predicted P(bit==1) `p` in `[1, 4095]`.
    ///
    /// Must be called in the same forward order as encoding.
    ///
    /// # Errors
    ///
    /// Returns [`AnsError`] on a corrupt or truncated stream.
    pub fn decode_bit(&mut self, p: u16) -> Result<bool, AnsError> {
        let table = bit_table(p);
        let sym = self.dec.get(table)?; // 0 or 1
        Ok(sym == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bits() {
        // Sequence of (bit, p) pairs. p is the predicted prob of bit==1.
        let pairs: Vec<(bool, u16)> = (0..256)
            .map(|i| (i % 3 == 0, if i % 3 == 0 { 3000 } else { 1095 }))
            .collect();
        let mut e = BitEncoder::new();
        for &(b, p) in &pairs {
            e.encode_bit(b, p);
        }
        let buf = e.finish();
        let mut d = BitDecoder::new(&buf).expect("decode init");
        for &(b, p) in &pairs {
            assert_eq!(d.decode_bit(p).expect("decode bit"), b, "bit mismatch");
        }
    }
}
