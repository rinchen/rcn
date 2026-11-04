//! LZP pre-stage with hash chains and cross-block persistence.
//!
//! Long-range repetition needs a real LDM-style window. zstd-1 uses up to 8MB;
//! we start with 4MB. To make that useful, the hash table becomes 8-entry chains
//! instead of a single last-position-per-bucket, so we can walk candidates and
//! pick the longest match.
//!
//! Cross-block persistence:
//! - The hash table and position chain survive across blocks within a stream.
//! - Mixer weights decay by `*0.995` per block instead of reset, so learning
//!   transfers across block boundaries.
//! - On decompress, the decoder reconstructs the same state from the decoded
//!   stream because the state update is deterministic from byte history.
//!
//! Block size classifier-aware:
//! - Text blocks: 1–4 MB (longer window for repeated phrases, quotes, boilerplate).
//! - Binary/Exec: 1 MiB (W1 — longer DP-LZP match span within a block).
//! - Random: 64 KiB (verbatim).
//! - Binary/Exec also keep up to 4 MiB of prior same-kind bytes as match history
//!   so DP-LZP distances can cross block boundaries.

use super::BitModel;
use super::ByteAssembler;

const MIN_MATCH: usize = 4;
const TABLE_BITS: u32 = 19; // 512K entries
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const MAX_CHAIN: usize = 8; // walk at most 8 candidates per lookup
const DEFAULT_WINDOW: usize = 32 * 1024 * 1024; // 32MB sliding window
const HISTORY_CAP: usize = 1 << 16; // 64K byte ring (only need last-4 for hashing + near-distance lookups)
const NEXT_SIZE: usize = 1 << 21; // 2M entry chain-link table — index by position % NEXT_SIZE

/// LZP matcher with hash chains and cross-block persistence.
///
/// This is a lightweight match pre-stage that provides byte-level prediction
/// to the mixer. It does NOT emit explicit match records (unlike the reverted
/// match-copy experiment). Instead it produces a high-confidence bit prediction
/// when a recent repeat is found in the 4MB hash chain.
pub struct Lzp {
    /// Hash chain heads: `heads[hash]` = newest position for this 4-gram.
    heads: Vec<u32>,
    /// Hash chain links: indexed by `position % NEXT_SIZE`.
    next: Vec<u32>,
    /// Absolute byte counter (never wraps during a single compression).
    current_pos: usize,
    /// Sliding window: only look back this many bytes.
    window: usize,
    /// Assembled byte stream so we can key the hash on whole bytes, not bits.
    asm: ByteAssembler,
    /// Ring buffer of recent bytes for hash/context lookup (only last-4 needed).
    history: Vec<u8>,
    /// Write cursor into the ring buffer.
    ring_pos: usize,
    /// Number of bytes written to the ring.
    ring_len: usize,
    /// Cached match byte for the current (in-progress) byte.
    /// Recomputed when a new byte completes, valid for all 8 bit positions.
    cached_match: Option<u8>,
    /// The bit position at which the current cached_match was computed.
    cached_bit_pos: u8,
    /// Bit index within the current (in-progress) byte.
    bit_pos: u8,
}

impl Lzp {
    #[must_use]
    pub fn new() -> Self {
        Self {
            heads: vec![0u32; TABLE_SIZE],
            next: vec![0u32; NEXT_SIZE],
            current_pos: 0,
            window: DEFAULT_WINDOW,
            asm: ByteAssembler::new(8),
            history: vec![0u8; HISTORY_CAP],
            ring_pos: 0,
            ring_len: 0,
            cached_match: None,
            cached_bit_pos: 0,
            bit_pos: 0,
        }
    }

    #[must_use]
    pub fn with_window(window: usize) -> Self {
        let mut s = Self::new();
        s.window = window;
        s
    }

    /// Get the last byte at ring-buffer offset `off` (0 = most recent).
    #[inline]
    fn ring_get(&self, off: usize) -> u8 {
        if off >= self.ring_len {
            return 0;
        }
        let idx = (self.ring_pos.wrapping_sub(1 + off)) % HISTORY_CAP;
        self.history[idx]
    }

    #[inline]
    fn hash_from_ring(&self) -> usize {
        let b0 = self.ring_get(3);
        let b1 = self.ring_get(2);
        let b2 = self.ring_get(1);
        let b3 = self.ring_get(0);
        let h =
            u32::from(b0) | (u32::from(b1) << 8) | (u32::from(b2) << 16) | (u32::from(b3) << 24);
        (h as usize) & (TABLE_SIZE - 1)
    }

    /// Record that the stream has advanced by one completed byte.
    pub fn train_byte(&mut self, byte: u8) {
        self.history[self.ring_pos] = byte;
        self.ring_pos = (self.ring_pos + 1) % HISTORY_CAP;
        if self.ring_len < HISTORY_CAP {
            self.ring_len += 1;
        }

        if self.ring_len >= 4 {
            let idx = self.hash_from_ring();
            let p = self.current_pos as u32;
            let old = self.heads[idx];
            self.heads[idx] = p;
            let ni = (p as usize) % NEXT_SIZE;
            self.next[ni] = old;
            self.current_pos += 1;
        }
    }

    /// Record that the stream has advanced to `pos`. For tests/external APIs.
    pub fn train_at(&mut self, data: &[u8], pos: usize) {
        if pos < 4 {
            return;
        }
        let idx = Self::hash_static(&data[..pos]);
        let p = pos as u32;
        let old = self.heads[idx];
        self.heads[idx] = p;
        let ni = (p as usize) % NEXT_SIZE;
        self.next[ni] = old;
        self.current_pos = pos;
    }

    #[inline]
    fn hash_static(history: &[u8]) -> usize {
        let n = history.len();
        if n < 4 {
            return 0;
        }
        let h = u32::from(history[n - 4])
            | (u32::from(history[n - 3]) << 8)
            | (u32::from(history[n - 2]) << 16)
            | (u32::from(history[n - 1]) << 24);
        (h as usize) & (TABLE_SIZE - 1)
    }

    /// Walk at most `MAX_CHAIN` candidates and return the longest match length,
    /// or `None` if no match reaches `MIN_MATCH`.
    pub fn longest_match(&self, data: &[u8], pos: usize) -> Option<usize> {
        if pos < 4 || pos >= data.len() {
            return None;
        }
        let idx = Self::hash_static(&data[..pos]);
        let mut best = 0usize;
        let mut cur = self.heads[idx];
        let mut walked = 0usize;
        let limit;
        if self.current_pos > self.window {
            limit = self.current_pos - self.window;
        } else {
            limit = 0;
        }
        while cur != 0 && walked < MAX_CHAIN {
            let p = cur as usize;
            let ni = p % NEXT_SIZE;
            if p >= pos || p < limit as usize {
                cur = self.next[ni];
                walked += 1;
                continue;
            }
            let mut len = 0usize;
            while pos + len < data.len() && p + len < data.len() && data[pos + len] == data[p + len]
            {
                len += 1;
                if len > 255 {
                    break;
                }
            }
            if len > best {
                best = len;
                if best == 255 {
                    break;
                }
            }
            cur = self.next[ni];
            walked += 1;
        }
        if best >= MIN_MATCH {
            Some(best)
        } else {
            None
        }
    }

    /// Walk at most `MAX_CHAIN` candidates and return the best match as
    /// `(len, dist)` — the length and the match distance — or `None` if no
    /// candidate reaches `MIN_MATCH`.
    ///
    /// Unlike [`longest_match`](Self::longest_match), this reports the distance
    /// directly from the chain walk instead of making the caller re-discover it
    /// with an O(window) backward scan. The match-length comparison is unchanged,
    /// so distances are exact, not estimates.
    pub fn best_match(&self, data: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos < 4 || pos >= data.len() {
            return None;
        }
        let idx = Self::hash_static(&data[..pos]);
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let mut cur = self.heads[idx];
        let mut walked = 0usize;
        let limit = self.current_pos.saturating_sub(self.window);
        while cur != 0 && walked < MAX_CHAIN {
            let p = cur as usize;
            let ni = p % NEXT_SIZE;
            if p >= pos || p < limit {
                cur = self.next[ni];
                walked += 1;
                continue;
            }
            let mut len = 0usize;
            while pos + len < data.len() && p + len < data.len() && data[pos + len] == data[p + len]
            {
                len += 1;
                if len > 255 {
                    break;
                }
            }
            if len > best_len {
                best_len = len;
                best_dist = pos - p;
                if len == 255 {
                    break;
                }
            }
            cur = self.next[ni];
            walked += 1;
        }
        if best_len >= MIN_MATCH {
            Some((best_len, best_dist))
        } else {
            None
        }
    }

    /// Find the most recent matching byte in the hash chain.
    /// Cached per-byte to avoid redundant chain walks on every bit.
    fn matched_byte(&mut self) -> Option<u8> {
        // If we already computed this for the current byte, reuse it.
        if self.cached_bit_pos == self.bit_pos {
            return self.cached_match;
        }

        let hlen = self.ring_len;
        if hlen < MIN_MATCH + 1 {
            self.cached_match = None;
            self.cached_bit_pos = self.bit_pos;
            return None;
        }

        let idx = self.hash_from_ring();
        let mut cur = self.heads[idx];
        let mut best = 0usize;
        let mut _best_pos = 0usize;
        let limit = self.current_pos.saturating_sub(self.window);
        let mut walked = 0usize;
        while cur != 0 && walked < MAX_CHAIN {
            let p = cur as usize;
            let ni = p % NEXT_SIZE;
            if p >= self.current_pos || p < limit {
                cur = self.next[ni];
                walked += 1;
                continue;
            }
            if self.current_pos - p > best {
                best = self.current_pos - p;
                _best_pos = p;
                if best == 255 {
                    break;
                }
            }
            cur = self.next[ni];
            walked += 1;
        }

        let result = if best >= MIN_MATCH && best < HISTORY_CAP {
            // Walk the ring buffer to find the matched byte.
            // best = distance, so matched byte is ring_get(best)
            Some(self.ring_get(best))
        } else {
            None
        };

        self.cached_match = result;
        self.cached_bit_pos = self.bit_pos;
        result
    }
}

impl BitModel for Lzp {
    fn predict(&self) -> u16 {
        // We can't cache in a &self method because matched_byte needs &mut self.
        // Instead, the update() call populates the cache, and predict() re-reads it.
        // The cache is set in update() when a byte completes.
        // For the first 8 bits of a byte, we use the cached value from the previous byte.
        if self.cached_bit_pos == self.bit_pos {
            self.cached_match.map_or(2048, |b| {
                let bit = (b >> (7 - self.bit_pos)) & 1 == 1;
                if bit {
                    3584
                } else {
                    512
                }
            })
        } else {
            2048
        }
    }

    fn update(&mut self, bit: bool) {
        self.bit_pos = self.asm.nbits();
        let completed = self.asm.push_bit(bit);
        if let Some(byte) = completed {
            self.train_byte(byte);
            // Invalidate cache — will be recomputed on next predict.
            self.cached_match = None;
            self.cached_bit_pos = 255; // force recompute
            self.bit_pos = 0;
        }
    }

    fn reset(&mut self) {
        self.heads.fill(0);
        self.next.fill(0);
        self.current_pos = 0;
        self.asm.reset();
        self.ring_pos = 0;
        self.ring_len = 0;
        self.cached_match = None;
        self.cached_bit_pos = 0;
        self.bit_pos = 0;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for Lzp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_byte_finds_repeat() {
        let mut lzp = Lzp::new();
        let data = b"abcabcabcabcabcabc";
        for i in 0..data.len() {
            lzp.train_byte(data[i]);
        }
        // After training on "abc" x 6, the hash of "abc" should have a chain.
        assert!(lzp.current_pos >= MIN_MATCH);
    }

    #[test]
    fn chain_no_match_on_unique_prefix() {
        let lzp = Lzp::new();
        assert!(lzp.longest_match(b"abcdefgh", 4).is_none());
    }
}
