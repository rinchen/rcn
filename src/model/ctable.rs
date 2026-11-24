//! Fast direct-addressed context counter table for the bit models.
//!
//! Originally each context model keyed its `[count0, count1]` tallies in a
//! `HashMap<(u64, u8), [u32; 2]>` and looked it up **per bit** (≈6 models × ~8 lookups
//! per byte × millions of bytes). That was both the dominant speed sink and the reason
//! orders above 2 were too expensive to use.
//!
//! This table is a power-of-two bucket array addressed by a multiplicative hash of the
//! context. Lookups are O(1) with no heap allocation. Reset is O(1): a global generation
//! counter is bumped, and a bucket is lazily re-seeded with `INIT` counts the first time
//! it is touched in a new block (detected when its stored generation differs). A hash
//! collision merely merges two contexts (a small ratio cost) — never a correctness bug,
//! because the codec is causal and the decoder mirrors the encoder exactly.

use std::cell::Cell;

const INIT: u32 = 32; // initial count per side so a fresh context predicts ~50/50

/// Direct-addressed per-context count table.
pub struct CtxTable {
    mask: usize,
    gen: Vec<Cell<u32>>,
    cur: Cell<u32>,
    c0: Vec<Cell<u32>>,
    c1: Vec<Cell<u32>>,
}

impl CtxTable {
    /// Create a table with `bits` address bits (2^`bits` buckets).
    #[must_use]
    pub fn new(bits: u32) -> Self {
        let n = 1usize << bits;
        Self {
            mask: n - 1,
            gen: (0..n).map(|_| Cell::new(0)).collect(),
            cur: Cell::new(1),
            c0: (0..n).map(|_| Cell::new(INIT)).collect(),
            c1: (0..n).map(|_| Cell::new(INIT)).collect(),
        }
    }

    #[inline]
    fn slot(&self, ctx: u64) -> usize {
        // 64-bit multiplicative hash (Knuth / splitmix style). Low bits index the bucket.
        let h = ctx.wrapping_mul(0x9E37_79B9_1AB3_57C5u64);
        let idx = (h as usize) & self.mask;
        let cur = self.cur.get();
        if self.gen[idx].get() != cur {
            self.gen[idx].set(cur);
            self.c0[idx].set(INIT);
            self.c1[idx].set(INIT);
        }
        idx
    }

    /// `[count0, count1]` for `ctx`, re-seeding the bucket if it is stale for this block.
    #[inline]
    pub fn get(&self, ctx: u64) -> [u32; 2] {
        let idx = self.slot(ctx);
        [self.c0[idx].get(), self.c1[idx].get()]
    }

    /// Increment the count for `bit` under `ctx`.
    #[inline]
    pub fn update(&self, ctx: u64, bit: bool) {
        let idx = self.slot(ctx);
        if bit {
            self.c1[idx].set(self.c1[idx].get() + 1);
        } else {
            self.c0[idx].set(self.c0[idx].get() + 1);
        }
    }

    /// Begin a fresh block. O(1): bump the generation; on full wrap, clear generations.
    pub fn reset(&self) {
        let nxt = self.cur.get().wrapping_add(1);
        if nxt == 0 {
            for g in &self.gen {
                g.set(0);
            }
        }
        self.cur.set(nxt);
    }
}
