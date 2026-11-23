//! Byte-level PPM-style entropy codec (the "fast" path).
//!
//! The bit path (see [`crate::codec`]) runs eight model-predict + mixer-mix +
//! eight model-update passes *per bit* (~16M bit-steps per 2MB block, with
//! text-trained models that are poorly calibrated on transformed streams).
//! This module is a true byte-level coder:
//!
//!   per byte = deterministic context selection → one 256-way distribution
//!              built from the winning count row → one rANS symbol.
//!
//! No `[u16; 256]` distribution array is materialized.  The 256-symbol
//! distribution walk is fused directly with the encode cumulative capture
//! (encode) or the decode cumulative search (decode), so both sides are
//! single-pass ALU-only per byte:
//!
//!   per byte = order selection (2 integer comparisons) + 256× (load + 2 mul +
//!              2 shift + fixed-point cumulative rounding + add) + 1 division
//!              (rANS state step) + model update (~20 ops).
//!
//! Contexts:
//!
//! * **order-0** (1 context), **order-1** (256 contexts), **order-2** (hashed
//!   4 096 / 1 024 contexts) — `u16` count tables with row-halving when any
//!   count exceeds 256 (keeps row totals ≤ 65 536).
//! * A **deterministic single-context selector** (no escapes, no softmax
//!   mixing): pick the order whose context is best concentrated, so both
//!   encode and decode agree on the distribution without any decoding of the
//!   symbol itself. Selection uses only context statistics (`total` and
//!   `distinct`), maintained incrementally.
//!
//! The encoder and decoder run byte-identical deterministic code, so the
//! distribution fed to [`RansByteEncoder::encode_fc`] on the encode side
//! equals the one the decoder walks in its cumulative search on the decode side.
//!
//! The block framing is handled by the caller ([`crate::codec`]): this module
//! only produces/consumes the rANS byte stream for one block. Uncompressed
//! length travels in the container `BlockEntry`.
//!
//! ## AVX2 / NEON `walk_dist`
//!
//! On AVX2-capable x86_64 CPUs the 256-symbol cumulative walk is vectorized
//! (unless `no_avx2`). On aarch64, a NEON path is used by default. Both are
//! bit-identical to the scalar path (parity tests in this module).

use crate::entropy::byterans::{RansByteDecoder32, RansByteEncoder32, BYTE_SCALE};
use crate::error::{RcnError, Result};

/// Hashed order-2 contexts. 4 096 rows × 256 × u16 = 2 MB.
const ORDER2_CTX: usize = 1 << 12;
const ORDER2_MASK: usize = ORDER2_CTX - 1;
/// Hashed order-2 contexts for small (binary/exec) blocks.
const ORDER2_CTX_SMALL: usize = 1 << 10;
/// Hashed order-3 contexts for deep Binary/Exec Fast path (W2).
const ORDER3_CTX: usize = 1 << 12;
const ORDER3_MASK: usize = ORDER3_CTX - 1;

/// Halve a row when any count exceeds this → row total ≤ 256 · 256 = 65 536.
const COUNT_CAP: u16 = 256;
const MAX_TOTAL: usize = COUNT_CAP as usize * 256;

/// Minimum distinct-probability mass before an order is eligible, so fresh
/// contexts don't win by accident. order-1: 2 bytes seen, order-2: 4 bytes.
const MIN_T1: u32 = 2;
const MIN_T2: u32 = 4;
const MIN_T3: u32 = 6;

/// Distribution scale of the normalized count row (movable mass, before the
/// base-1 per symbol).
const BUDGET: u32 = BYTE_SCALE - 256;

/// Fixed-point fractional bits for cumulative rounding (guarantees the 256-symbol
/// sum = SCALE-1 or SCALE exactly, at most one unit of deficit patched to
/// symbol 255). `idx` is a 12-bit probability, so `exact = idx·BUDGET/4096`
/// accumulates directly as `idx·BUDGET` fixed-point at 2^12.
const FRAC_BITS: u32 = 12;
const FRAC_ROUND: u64 = 1 << (FRAC_BITS - 1);

/// `floor((1 << 32) / t)` for `t` in 1..=MAX_TOTAL.
fn reciprocal_table() -> &'static [u32; MAX_TOTAL + 1] {
    static T: std::sync::OnceLock<[u32; MAX_TOTAL + 1]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; MAX_TOTAL + 1];
        for (i, v) in t.iter_mut().enumerate().skip(1) {
            *v = ((1u64 << 32) / i as u64) as u32;
        }
        t
    })
}

fn hash2(a: u8, b: u8) -> usize {
    let x = u32::from(a) | (u32::from(b) << 8);
    (x.wrapping_mul(0x9E37_79B1) >> 20) as usize
}

fn hash3(a: u8, b: u8, c: u8) -> usize {
    let x = u32::from(a) | (u32::from(b) << 8) | (u32::from(c) << 16);
    (x.wrapping_mul(0x9E37_79B1) >> 20) as usize
}

/// AVX2-accelerated inclusive prefix sum of 8 u32 elements.
///
/// Input:  [a0, a1, a2, a3, a4, a5, a6, a7]
/// Output: [a0, a0+a1, ..., a0+..+a3, a4+a0..a3, ..., a4+..+a7+a0..a3]
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn avx2_prefix_sum_epi32(v: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    // Step 1: pairwise sums. `_mm256_slli_si256` shifts each 128-bit lane
    // toward higher indices (left), so each element gets the previous one.
    let s1 = std::arch::x86_64::_mm256_add_epi32(v, std::arch::x86_64::_mm256_slli_si256::<4>(v));
    // Step 2: quad sums. `_mm256_slli_si256` shifts each 128-bit lane
    // toward higher indices, so each lane holds the correct 4-element
    // prefix *relative to itself*; the upper lane is still missing the lower
    // lane's total.
    let s2 = std::arch::x86_64::_mm256_add_epi32(s1, std::arch::x86_64::_mm256_slli_si256::<8>(s1));
    // Step 3: propagate the lower 128-bit lane's total into the upper lane.
    // The lower lane's total is s2[3] (the last element of the lower lane).
    // Broadcasting it only to the upper lane keeps the lower lane intact.
    let lower_sum = std::arch::x86_64::_mm256_extract_epi32::<3>(s2);
    let upper_sum = std::arch::x86_64::_mm256_set_epi32(0, 0, 0, 0, lower_sum, lower_sum, lower_sum, lower_sum);
    std::arch::x86_64::_mm256_add_epi32(s2, upper_sum)
}

/// AVX2-accelerated `walk_dist` — bit-identical to the scalar path.
///
/// Processes 32 symbols per outer iteration (8 AVX2 lanes × 4 chunks).
/// Each chunk: loads 8 u16 counts → widens to u32 → `idx = (cnt*inv)>>20`
/// → `weighted = idx*BUDGET` → prefix sum → chain offset from previous
/// chunks → extract to scalar for the threshold/target search.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn walk_dist_avx2(
    counts: *const u16,
    base: usize,
    inv: u32,
    target: Option<u8>,
    cdf: Option<u32>,
) -> WalkResult {
    use std::arch::x86_64::*;

    let budget = BUDGET as u32;
    let inv_v = _mm256_set1_epi32(inv as i32);
    let budget_v = _mm256_set1_epi32(budget as i32);
    let mut prev_r: u64 = 0;
    let mut cum: u32 = 0;
    let mut running_total: u32 = 0;
    let mut cum_before_255 = 0u32;
    let mut f255 = 0u32;
    let mut found_sym = -1i32;
    let mut found_freq = 0u32;
    let mut found_cum = 0u32;

    for chunk_start in (0..256).step_by(32) {
        // Load 32 u16 counts, widen to u32
        let c0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(
            counts.add(base + chunk_start) as *const __m128i,
        ));
        let c1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(
            counts.add(base + chunk_start + 8) as *const __m128i,
        ));
        let c2 = _mm256_cvtepu16_epi32(_mm_loadu_si128(
            counts.add(base + chunk_start + 16) as *const __m128i,
        ));
        let c3 = _mm256_cvtepu16_epi32(_mm_loadu_si128(
            counts.add(base + chunk_start + 24) as *const __m128i,
        ));

        /// [x86_64] Multiply u16 counts (zero-extended to u32) by u32 reciprocal,
/// produce u32 index = (cnt * inv) >> 20 (low 32 bits only).
/// Safe for non-power-of-two totals t where cnt * inv < 2^32 for all cnt ≤ t.
#[inline(always)]
unsafe fn avx2_mul_epu32_to_i32(a: std::arch::x86_64::__m256i, inv: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    let inv_u32 = _mm256_castps_si256(_mm256_castsi256_ps(inv));
    _mm256_mullo_epi32(a, inv_u32)
}

// idx = (cnt * inv) >> 20 where cnt is u16 zero-extended to u32, inv is u32
let idx0 = _mm256_srli_epi32(avx2_mul_epu32_to_i32(c0, inv_v), 20);
let idx1 = _mm256_srli_epi32(avx2_mul_epu32_to_i32(c1, inv_v), 20);
let idx2 = _mm256_srli_epi32(avx2_mul_epu32_to_i32(c2, inv_v), 20);
let idx3 = _mm256_srli_epi32(avx2_mul_epu32_to_i32(c3, inv_v), 20);

        // weighted = idx * BUDGET
        let w0 = _mm256_mullo_epi32(idx0, budget_v);
        let w1 = _mm256_mullo_epi32(idx1, budget_v);
        let w2 = _mm256_mullo_epi32(idx2, budget_v);
        let w3 = _mm256_mullo_epi32(idx3, budget_v);

        // Prefix sums
        let p0 = avx2_prefix_sum_epi32(w0);
        // Continue the cumulative from the previous 32-symbol outer iteration so
        // `r - prev_r` never goes negative across chunk_start boundaries.
        let p0 = _mm256_add_epi32(p0, _mm256_set1_epi32(running_total as i32));
        let p1 = avx2_prefix_sum_epi32(w1);
        let p2 = avx2_prefix_sum_epi32(w2);
        let p3 = avx2_prefix_sum_epi32(w3);

        // Chain prefix sums across chunks
        let p0_final = _mm256_extract_epi32::<7>(p0) as u32;
        let p1 = _mm256_add_epi32(p1, _mm256_set1_epi32(p0_final as i32));
        let p1_final = _mm256_extract_epi32::<7>(p1) as u32;
        let p2 = _mm256_add_epi32(p2, _mm256_set1_epi32(p1_final as i32));
        let p2_final = _mm256_extract_epi32::<7>(p2) as u32;
        let p3 = _mm256_add_epi32(p3, _mm256_set1_epi32(p2_final as i32));
        let p3_final = _mm256_extract_epi32::<7>(p3) as u32;
        running_total = p3_final;

        // Extract to scalar arrays (u32: cumulative values can exceed 2^31)
        let mut buf0 = [0u32; 8];
        let mut buf1 = [0u32; 8];
        let mut buf2 = [0u32; 8];
        let mut buf3 = [0u32; 8];
        _mm256_storeu_si256(buf0.as_mut_ptr() as *mut __m256i, p0);
        _mm256_storeu_si256(buf1.as_mut_ptr() as *mut __m256i, p1);
        _mm256_storeu_si256(buf2.as_mut_ptr() as *mut __m256i, p2);
        _mm256_storeu_si256(buf3.as_mut_ptr() as *mut __m256i, p3);

        for (chunk_idx, &_offset) in [0u32, p0_final, p1_final, p2_final].iter().enumerate() {
            let buf = match chunk_idx {
                0 => &buf0,
                1 => &buf1,
                2 => &buf2,
                _ => &buf3,
            };
            let start = chunk_start + chunk_idx * 8;
            for (i, &cum_val) in buf.iter().enumerate() {
                let s = start + i;
                let r = ((cum_val as u64 + FRAC_ROUND) >> FRAC_BITS);
                let base_s = (r as i64 - prev_r as i64) as u64;
                prev_r = r;
                let f = 1u32 + base_s as u32;

                if s == 255 {
                    f255 = f;
                    cum_before_255 = cum;
                }

                if let Some(b) = target {
                    if s == usize::from(b) {
                        found_cum = cum;
                        found_freq = f;
                    }
                }
                cum += f;

                if let Some(c) = cdf {
                    if found_sym < 0 && cum > c {
                        found_sym = s as i32;
                        found_freq = f;
                        found_cum = cum - f;
                    }
                }
            }
        }
    }

    let total = cum;
    let sym = if found_sym >= 0 {
        found_sym as u8
    } else if cdf.is_some() {
        255
    } else {
        target.unwrap_or(0)
    };
    if sym == 255 {
        found_freq = f255 + (BYTE_SCALE - total);
        found_cum = cum_before_255;
    }

    WalkResult {
        sym,
        freq: found_freq,
        cum: found_cum,
    }
}

/// NEON-accelerated `walk_dist` — bit-identical to the scalar path.
///
/// Processes 4 symbols per iteration with `uint32x4_t`: load u16 counts, widen,
/// `(cnt * inv) >> 20` via 32×32→64 widening mul (always safe — no power-of-two
/// guard), accumulate `idx * BUDGET`, then scalar finalize for target/cdf search.
#[cfg(target_arch = "aarch64")]
unsafe fn walk_dist_neon(
    counts: *const u16,
    base: usize,
    inv: u32,
    target: Option<u8>,
    cdf: Option<u32>,
) -> WalkResult {
    use std::arch::aarch64::*;

    let inv_v = vdupq_n_u32(inv);
    let budget_v = vdupq_n_u32(BUDGET);
    let mut prev_r: u64 = 0;
    let mut cum: u32 = 0;
    let mut cum_before_255 = 0u32;
    let mut f255 = 0u32;
    let mut found_sym = -1i32;
    let mut found_freq = 0u32;
    let mut found_cum = 0u32;
    let mut acc_hi = 0u64;

    for chunk_start in (0..256).step_by(4) {
        let cptr = counts.add(base + chunk_start);
        let c16 = vld1_u16(cptr);
        let c32 = vmovl_u16(c16);

        let lo = vmull_u32(vget_low_u32(c32), vget_low_u32(inv_v));
        let hi = vmull_u32(vget_high_u32(c32), vget_high_u32(inv_v));
        let idx_lo = vshrn_n_u64::<20>(lo);
        let idx_hi = vshrn_n_u64::<20>(hi);
        let idx = vcombine_u32(idx_lo, idx_hi);
        let weighted = vmulq_u32(idx, budget_v);

        let mut w = [0u32; 4];
        vst1q_u32(w.as_mut_ptr(), weighted);

        for (i, &w_i) in w.iter().enumerate() {
            let s = chunk_start + i;
            acc_hi = acc_hi.wrapping_add(u64::from(w_i));
            let r = (acc_hi + FRAC_ROUND) >> FRAC_BITS;
            let base_s = r - prev_r;
            prev_r = r;
            let f = 1u32 + base_s as u32;

            if s == 255 {
                f255 = f;
                cum_before_255 = cum;
            }

            if let Some(b) = target {
                if s == usize::from(b) {
                    found_cum = cum;
                    found_freq = f;
                }
            }
            cum += f;

            if let Some(c) = cdf {
                if found_sym < 0 && cum > c {
                    found_sym = s as i32;
                    found_freq = f;
                    found_cum = cum - f;
                }
            }
        }
    }

    let total = cum;
    let sym = if found_sym >= 0 {
        found_sym as u8
    } else if cdf.is_some() {
        255
    } else {
        target.unwrap_or(0)
    };
    if sym == 255 {
        found_freq = f255 + (BYTE_SCALE - total);
        found_cum = cum_before_255;
    }

    WalkResult {
        sym,
        freq: found_freq,
        cum: found_cum,
    }
}

/// Count tables with incremental totals/distinct for the order selector.
///
/// `totals`/`distinct` are maintained incrementally and recomputed only on
/// row-halving, so [`ByteCountModel::total`] and [`ByteCountModel::distinct`]
/// are O(1) for the context selector.
struct ByteCountModel {
    counts: Vec<u16>,
    num_ctx: usize,
    totals: Vec<u32>,
    distinct: Vec<u32>,
}

impl ByteCountModel {
    fn new(num_ctx: usize) -> Self {
        Self {
            counts: vec![0; num_ctx * 256],
            num_ctx,
            totals: vec![0; num_ctx],
            distinct: vec![0; num_ctx],
        }
    }

    #[inline]
    fn total(&self, ctx: usize) -> u32 {
        self.totals[ctx]
    }

    #[inline]
    fn distinct(&self, ctx: usize) -> u32 {
        self.distinct[ctx]
    }

    /// Walk the count row for `ctx` and, in a single pass:
    ///
    /// - **encode** (`target = Some(byte)`): capture `(freq, cum)` for that byte.
    /// - **decode** (`cdf = Some(state_cdf)`): find the byte whose cumulative
    ///   range contains `cdf`, and return it.
    ///
    /// The walk builds the distribution on-the-fly using cumulative fixed-point
    /// rounding (no sort, no extra materialisation). The sum is at most SCALE;
    /// symbol 255 absorbs any leftover mass so encode and decode agree on
    /// range [cum_before_255, SCALE) for it.
    #[inline(always)]
    fn walk_dist(
        &self,
        ctx: usize,
        target: Option<u8>,
        cdf: Option<u32>,
    ) -> WalkResult {
        let base = ctx * 256;
        let t = u64::from(self.totals[ctx].max(1));
        let inv = u32::from(reciprocal_table()[t as usize]);

        #[cfg(all(target_arch = "x86_64", not(feature = "no_avx2")))]
        {
            // AVX2 path is bit-identical to scalar only when `cnt * inv < 2^32`.
            // inv = floor(2^32/t). Wrap occurs iff t is a power of two and cnt == t
            // (single symbol dominates the context). Guard: disable AVX2 for power-of-two totals.
            let use_avx2 = is_x86_feature_detected!("avx2") && !t.is_power_of_two();
            if use_avx2 {
                return unsafe { walk_dist_avx2(self.counts.as_ptr(), base, inv, target, cdf) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { walk_dist_neon(self.counts.as_ptr(), base, inv, target, cdf) };
        }

        #[allow(unreachable_code)]
        walk_dist_scalar(&self.counts, base, inv, target, cdf)
    }

    /// Force the scalar `walk_dist` implementation (for AVX2↔scalar parity tests).
    #[cfg(test)]
    fn walk_dist_scalar_only(
        &self,
        ctx: usize,
        target: Option<u8>,
        cdf: Option<u32>,
    ) -> WalkResult {
        let base = ctx * 256;
        let t = u64::from(self.totals[ctx].max(1));
        let inv = u32::from(reciprocal_table()[t as usize]);
        walk_dist_scalar(&self.counts, base, inv, target, cdf)
    }

    /// Call the AVX2 path directly when available and safe for this total.
    #[cfg(all(test, target_arch = "x86_64", not(feature = "no_avx2")))]
    fn walk_dist_avx2_only(
        &self,
        ctx: usize,
        target: Option<u8>,
        cdf: Option<u32>,
    ) -> Option<WalkResult> {
        let t = u64::from(self.totals[ctx].max(1));
        if !is_x86_feature_detected!("avx2") || t.is_power_of_two() {
            return None;
        }
        let base = ctx * 256;
        let inv = u32::from(reciprocal_table()[t as usize]);
        Some(unsafe { walk_dist_avx2(self.counts.as_ptr(), base, inv, target, cdf) })
    }

    /// Call the NEON path directly (aarch64).
    #[cfg(all(test, target_arch = "aarch64"))]
    fn walk_dist_neon_only(
        &self,
        ctx: usize,
        target: Option<u8>,
        cdf: Option<u32>,
    ) -> WalkResult {
        let base = ctx * 256;
        let t = u64::from(self.totals[ctx].max(1));
        let inv = u32::from(reciprocal_table()[t as usize]);
        unsafe { walk_dist_neon(self.counts.as_ptr(), base, inv, target, cdf) }
    }

    fn update(&mut self, ctx: usize, byte: u8) {
        let base = ctx * 256;
        let s = usize::from(byte);
        let c = &mut self.counts[base + s];
        *c = c.saturating_add(1);
        if *c == 1 {
            self.distinct[ctx] += 1;
        }
        self.totals[ctx] += 1;
        if *c > COUNT_CAP {
            let mut t = 0u32;
            let mut d = 0u32;
            for v in &mut self.counts[base..base + 256] {
                *v /= 2;
                t += u32::from(*v);
                if *v > 0 {
                    d += 1;
                }
            }
            self.totals[ctx] = t;
            self.distinct[ctx] = d;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WalkResult {
    sym: u8,
    freq: u32,
    cum: u32,
}

/// Scalar `walk_dist` implementation shared by the dispatcher and tests.
#[inline(always)]
fn walk_dist_scalar(
    counts: &[u16],
    base: usize,
    inv: u32,
    target: Option<u8>,
    cdf: Option<u32>,
) -> WalkResult {
    let mut acc = 0u64;
    let mut prev = 0u64;
    let mut cum = 0u32;
    let mut cum_before_255 = 0u32;
    let mut f255 = 0u32;
    let mut found_sym = -1i32;
    let mut found_freq = 0u32;
    let mut found_cum = 0u32;

    for s in 0..256 {
        let cnt = u64::from(counts[base + s]);
        let idx = (cnt * u64::from(inv)) >> 20;
        acc += idx * u64::from(BUDGET);
        let r = (acc + FRAC_ROUND) >> FRAC_BITS;
        let base_s = r - prev;
        prev = r;
        let f = 1u32 + base_s as u32;

        if s == 255 {
            f255 = f;
            cum_before_255 = cum;
        }

        if let Some(b) = target {
            if s == usize::from(b) {
                found_cum = cum;
                found_freq = f;
            }
        }
        cum += f;

        if let Some(c) = cdf {
            if found_sym < 0 && cum > c {
                found_sym = s as i32;
                found_freq = f;
                found_cum = cum - f;
            }
        }
    }

    let total = cum;
    let sym = if found_sym >= 0 {
        found_sym as u8
    } else if cdf.is_some() {
        255
    } else {
        target.unwrap_or(0)
    };
    if sym == 255 {
        found_freq = f255 + (BYTE_SCALE - total);
        found_cum = cum_before_255;
    }

    WalkResult {
        sym,
        freq: found_freq,
        cum: found_cum,
    }
}

/// PPM-style deterministic order selection: pick the highest-order context
/// whose concentration (`total/(distinct+1)`-style score) beats the lower
/// orders. Only context statistics are consulted — never the symbol being
/// encoded — so encode and decode agree without any escape mechanism.
#[inline(always)]
fn pick_order(t1: u32, d1: u32, t2: u32, d2: u32) -> Order {
    if t2 >= MIN_T2
        && (t1 < MIN_T1
            || u64::from(t2) * u64::from(d1 + 1) >= u64::from(t1) * u64::from(d2 + 1))
    {
        Order::Order2
    } else if t1 >= MIN_T1 {
        Order::Order1
    } else {
        Order::Order0
    }
}

/// Deep Binary/Exec selector (W2): also considers hashed order-3.
#[inline(always)]
fn pick_order_deep(t1: u32, d1: u32, t2: u32, d2: u32, t3: u32, d3: u32) -> Order {
    if t3 >= MIN_T3
        && (t2 < MIN_T2
            || u64::from(t3) * u64::from(d2 + 1) >= u64::from(t2) * u64::from(d3 + 1))
    {
        // Prefer o3 over o2 when concentrated; still fall through vs o1/o0.
        if t1 < MIN_T1
            || u64::from(t3) * u64::from(d1 + 1) >= u64::from(t1) * u64::from(d3 + 1)
        {
            return Order::Order3;
        }
    }
    pick_order(t1, d1, t2, d2)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Order0,
    Order1,
    Order2,
    Order3,
}

/// Compress one block's worth of bytes into an rANS byte stream.
///
/// `data.len()` must be ≥ 1; empty input should be stored via the container's
/// copy method instead.
#[must_use]
pub fn compress_block(data: &[u8]) -> Vec<u8> {
    compress_block_inner(data, &[], false)
}

/// One DP-LZP match run: start at `pos`, copy `len` bytes from `dist` back.
#[derive(Debug, Clone, Copy)]
pub struct ByteMatchRun {
    pub pos: usize,
    pub len: usize,
    pub dist: usize,
}

/// Compress `data`, skipping rANS for matched regions (models still updated).
///
/// Framing: `[num_runs:u32][flag:u8][payload_len:u32][side…][rANS…]` — same
/// side-stream layout as the bit path so Hybrid/Fast can share match records.
///
/// `deep` enables hashed order-3 + large o2 tables (W2 Binary/Exec Fast path).
#[must_use]
pub fn compress_block_with_matches(data: &[u8], runs: &[ByteMatchRun], deep: bool) -> Vec<u8> {
    use crate::entropy::side_fse;

    let mut raw_side = Vec::new();
    let mut prev_pos: usize = 0;
    for r in runs {
        let delta_pos = r.pos.wrapping_sub(prev_pos);
        write_varint(&mut raw_side, delta_pos);
        write_varint(&mut raw_side, r.len);
        write_varint(&mut raw_side, r.dist);
        prev_pos = r.pos;
    }
    let (flag, payload) = side_fse::pack_side_stream(&raw_side);
    let mut out = Vec::new();
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    out.push(flag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend(compress_block_inner(data, runs, deep));
    out
}

fn write_varint(out: &mut Vec<u8>, mut v: usize) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

fn read_varint(data: &[u8], mut offset: usize) -> Option<(usize, usize)> {
    let mut result = 0usize;
    let mut shift = 0u32;
    loop {
        if offset >= data.len() || shift > 63 {
            return None;
        }
        let b = data[offset];
        offset += 1;
        result |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            return Some((result, offset));
        }
        shift += 7;
    }
}

fn compress_block_inner(data: &[u8], runs: &[ByteMatchRun], deep: bool) -> Vec<u8> {
    let big = deep || data.len() >= 256 * 1024;
    let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
    let order2_mask = order2_ctx - 1;
    let mut o0 = ByteCountModel::new(1);
    let mut o1 = ByteCountModel::new(256);
    let mut o2 = ByteCountModel::new(order2_ctx);
    let mut o3 = if deep {
        Some(ByteCountModel::new(ORDER3_CTX))
    } else {
        None
    };
    let mut enc = RansByteEncoder32::new();

    let mut match_len_at: Vec<usize> = vec![0; data.len()];
    for r in runs {
        if r.pos < data.len() && match_len_at[r.pos] < r.len {
            match_len_at[r.pos] = r.len;
        }
    }

    let mut p_3 = 0u8;
    let mut p_2 = 0u8;
    let mut p_1 = 0u8;
    let mut i = 0usize;
    while i < data.len() {
        if match_len_at[i] > 0 {
            let len = match_len_at[i];
            for j in 0..len {
                let b = data[i + j];
                let c1 = usize::from(p_1);
                let c2 = hash2(p_1, p_2) & order2_mask;
                o0.update(0, b);
                o1.update(c1, b);
                o2.update(c2, b);
                if let Some(ref mut o3m) = o3 {
                    let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                    o3m.update(c3, b);
                }
                p_3 = p_2;
                p_2 = p_1;
                p_1 = b;
            }
            i += len;
        } else {
            let b = data[i];
            let c1 = usize::from(p_1);
            let c2 = hash2(p_1, p_2) & order2_mask;
            let ord = if let Some(ref o3m) = o3 {
                let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                pick_order_deep(
                    o1.total(c1),
                    o1.distinct(c1),
                    o2.total(c2),
                    o2.distinct(c2),
                    o3m.total(c3),
                    o3m.distinct(c3),
                )
            } else {
                pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2))
            };
            let wr = match ord {
                Order::Order0 => o0.walk_dist(0, Some(b), None),
                Order::Order1 => o1.walk_dist(c1, Some(b), None),
                Order::Order2 => o2.walk_dist(c2, Some(b), None),
                Order::Order3 => {
                    let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                    o3.as_ref()
                        .expect("order-3 model exists when Order3 is selected")
                        .walk_dist(c3, Some(b), None)
                }
            };
            enc.encode_fc(wr.freq, wr.cum);

            o0.update(0, b);
            o1.update(c1, b);
            o2.update(c2, b);
            if let Some(ref mut o3m) = o3 {
                let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                o3m.update(c3, b);
            }

            p_3 = p_2;
            p_2 = p_1;
            p_1 = b;
            i += 1;
        }
    }
    enc.finish()
}

/// Decompress a block stream from [`compress_block`].
///
/// # Errors
///
/// Returns [`RcnError::CorruptBlock`] if the stream is truncated (fewer than
/// the 4-byte rANS state tail).
pub fn decompress_block(comp: &[u8], orig_len: usize) -> Result<Vec<u8>> {
    decompress_block_plain(comp, orig_len, false)
}

/// Decompress a block produced by [`compress_block_with_matches`].
///
/// `match_hist` is prior same-kind Binary/Exec bytes (may be empty) so match
/// distances can reach across block boundaries (W1).
/// `deep` must match the encoder (W2 Binary/Exec hashed order-3).
///
/// # Errors
///
/// Returns [`RcnError`] on truncated side-stream or corrupt rANS payload.
pub fn decompress_block_with_matches(
    comp: &[u8],
    orig_len: usize,
    match_hist: &[u8],
    deep: bool,
) -> Result<Vec<u8>> {
    use crate::entropy::side_fse;

    if comp.len() < 9 {
        return Err(RcnError::CorruptBlock(
            "byte match side-stream too short".into(),
        ));
    }
    let num_runs = u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]) as usize;
    let flag = comp[4];
    let payload_len = u32::from_le_bytes([comp[5], comp[6], comp[7], comp[8]]) as usize;
    let payload_start: usize = 9;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| RcnError::CorruptBlock("byte match side-stream length overflow".into()))?;
    if payload_end > comp.len() {
        return Err(RcnError::CorruptBlock(
            "byte match side-stream truncated".into(),
        ));
    }
    let raw_side = side_fse::unpack_side_stream(flag, &comp[payload_start..payload_end])
        .map_err(|()| RcnError::CorruptBlock("corrupt byte match side-stream entropy".into()))?;

    let mut runs: Vec<ByteMatchRun> = Vec::with_capacity(num_runs);
    let mut prev_pos: usize = 0;
    let mut offset = 0usize;
    for _ in 0..num_runs {
        let (delta_pos, new_offset) = read_varint(&raw_side, offset).ok_or_else(|| {
            RcnError::CorruptBlock("truncated byte match record (delta_pos)".into())
        })?;
        offset = new_offset;
        let (len, new_offset) = read_varint(&raw_side, offset)
            .ok_or_else(|| RcnError::CorruptBlock("truncated byte match record (len)".into()))?;
        offset = new_offset;
        let (dist, new_offset) = read_varint(&raw_side, offset)
            .ok_or_else(|| RcnError::CorruptBlock("truncated byte match record (dist)".into()))?;
        offset = new_offset;
        let pos = prev_pos.wrapping_add(delta_pos);
        prev_pos = pos;
        runs.push(ByteMatchRun { pos, len, dist });
    }

    if runs.is_empty() {
        return decompress_block_plain(&comp[payload_end..], orig_len, deep);
    }

    let mut match_len_at: Vec<usize> = vec![0; orig_len];
    let mut match_dist_at: Vec<usize> = vec![0; orig_len];
    for r in &runs {
        if r.pos < orig_len && match_len_at[r.pos] < r.len {
            match_len_at[r.pos] = r.len;
            match_dist_at[r.pos] = r.dist;
        }
    }

    let n_lits = {
        let mut matched = 0usize;
        let mut i = 0usize;
        while i < orig_len {
            if match_len_at[i] > 0 {
                let len = match_len_at[i];
                matched += len;
                i += len;
            } else {
                i += 1;
            }
        }
        orig_len.saturating_sub(matched)
    };

    let big = deep || orig_len >= 256 * 1024;
    let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
    let order2_mask = order2_ctx - 1;
    let mut o0 = ByteCountModel::new(1);
    let mut o1 = ByteCountModel::new(256);
    let mut o2 = ByteCountModel::new(order2_ctx);
    let mut o3 = if deep {
        Some(ByteCountModel::new(ORDER3_CTX))
    } else {
        None
    };
    let mut dec = RansByteDecoder32::new(&comp[payload_end..], n_lits)
        .map_err(|_| RcnError::CorruptBlock("short rANS stream".into()))?;

    let mut p_3 = 0u8;
    let mut p_2 = 0u8;
    let mut p_1 = 0u8;
    let mut out = Vec::with_capacity(orig_len);
    let mut i = 0usize;
    let mut lane_cdfs = [0u32; 32];
    let mut lane_i = 32usize;
    let mut lanes_in_group = 0usize;

    while out.len() < orig_len {
        if i < orig_len && match_len_at[i] > 0 {
            let len = match_len_at[i];
            let dist = match_dist_at[i];
            for _ in 0..len {
                let byte = if dist == 0 {
                    0u8
                } else if dist <= out.len() {
                    out[out.len() - dist]
                } else {
                    let into_hist = dist - out.len();
                    if into_hist <= match_hist.len() {
                        match_hist[match_hist.len() - into_hist]
                    } else {
                        0u8
                    }
                };
                let c1 = usize::from(p_1);
                let c2 = hash2(p_1, p_2) & order2_mask;
                o0.update(0, byte);
                o1.update(c1, byte);
                o2.update(c2, byte);
                if let Some(ref mut o3m) = o3 {
                    let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                    o3m.update(c3, byte);
                }
                out.push(byte);
                p_3 = p_2;
                p_2 = p_1;
                p_1 = byte;
            }
            i += len;
        } else {
            if lane_i >= lanes_in_group {
                if dec.remaining() == 0 {
                    return Err(RcnError::CorruptBlock(
                        "byte match decode: ran out of literals".into(),
                    ));
                }
                lane_cdfs = dec.cdf_batch();
                lanes_in_group = dec.remaining().min(32);
                lane_i = 0;
            }
            let c1 = usize::from(p_1);
            let c2 = hash2(p_1, p_2) & order2_mask;
            let ord = if let Some(ref o3m) = o3 {
                let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                pick_order_deep(
                    o1.total(c1),
                    o1.distinct(c1),
                    o2.total(c2),
                    o2.distinct(c2),
                    o3m.total(c3),
                    o3m.distinct(c3),
                )
            } else {
                pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2))
            };
            let cdf = lane_cdfs[lane_i];
            let wr = match ord {
                Order::Order0 => o0.walk_dist(0, None, Some(cdf)),
                Order::Order1 => o1.walk_dist(c1, None, Some(cdf)),
                Order::Order2 => o2.walk_dist(c2, None, Some(cdf)),
                Order::Order3 => {
                    let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                    o3.as_ref()
                        .expect("order-3 model exists when Order3 is selected")
                        .walk_dist(c3, None, Some(cdf))
                }
            };
            dec.lane_advance(lane_i, wr.freq, wr.cum);
            lane_i += 1;

            o0.update(0, wr.sym);
            o1.update(c1, wr.sym);
            o2.update(c2, wr.sym);
            if let Some(ref mut o3m) = o3 {
                let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                o3m.update(c3, wr.sym);
            }
            out.push(wr.sym);
            p_3 = p_2;
            p_2 = p_1;
            p_1 = wr.sym;
            i += 1;
        }
    }
    Ok(out)
}

fn decompress_block_plain(comp: &[u8], orig_len: usize, deep: bool) -> Result<Vec<u8>> {
    let big = deep || orig_len >= 256 * 1024;
    let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
    let order2_mask = order2_ctx - 1;
    let mut o0 = ByteCountModel::new(1);
    let mut o1 = ByteCountModel::new(256);
    let mut o2 = ByteCountModel::new(order2_ctx);
    let mut o3 = if deep {
        Some(ByteCountModel::new(ORDER3_CTX))
    } else {
        None
    };
    let mut dec = RansByteDecoder32::new(comp, orig_len)
        .map_err(|_| RcnError::CorruptBlock("short rANS stream".into()))?;

    let mut p_3 = 0u8;
    let mut p_2 = 0u8;
    let mut p_1 = 0u8;

    let mut out = Vec::with_capacity(orig_len);
    let mut lane_cdfs = [0u32; 32];
    let mut lane_i = 32usize;
    let mut lanes_in_group = 0usize;

    while out.len() < orig_len {
        if lane_i >= lanes_in_group {
            if dec.remaining() == 0 {
                return Err(RcnError::CorruptBlock(
                    "byte plain decode: ran out of symbols".into(),
                ));
            }
            lane_cdfs = dec.cdf_batch();
            lanes_in_group = dec.remaining().min(32);
            lane_i = 0;
        }
        let c1 = usize::from(p_1);
        let c2 = hash2(p_1, p_2) & order2_mask;
        let ord = if let Some(ref o3m) = o3 {
            let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
            pick_order_deep(
                o1.total(c1),
                o1.distinct(c1),
                o2.total(c2),
                o2.distinct(c2),
                o3m.total(c3),
                o3m.distinct(c3),
            )
        } else {
            pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2))
        };
        let cdf = lane_cdfs[lane_i];
        let wr = match ord {
            Order::Order0 => o0.walk_dist(0, None, Some(cdf)),
            Order::Order1 => o1.walk_dist(c1, None, Some(cdf)),
            Order::Order2 => o2.walk_dist(c2, None, Some(cdf)),
            Order::Order3 => {
                let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
                o3.as_ref()
                    .expect("order-3 model exists when Order3 is selected")
                    .walk_dist(c3, None, Some(cdf))
            }
        };
        dec.lane_advance(lane_i, wr.freq, wr.cum);
        lane_i += 1;

        o0.update(0, wr.sym);
        o1.update(c1, wr.sym);
        o2.update(c2, wr.sym);
        if let Some(ref mut o3m) = o3 {
            let c3 = hash3(p_1, p_2, p_3) & ORDER3_MASK;
            o3m.update(c3, wr.sym);
        }
        out.push(wr.sym);
        p_3 = p_2;
        p_2 = p_1;
        p_1 = wr.sym;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::byterans::{RansByteDecoder, RansByteEncoder};

    fn roundtrip(data: &[u8]) {
        let comp = compress_block(data);
        let back = decompress_block(&comp, data.len()).expect("decompress");
        assert_eq!(back, data, "byte-codec round-trip mismatch");
    }

    #[test]
    fn walk_roundtrip_exact() {
        let mut v = vec![0u8; 8192];
        let mut x = 0x9E37_79B9u32;
        for b in &mut v {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let n = 256;
        let big = n >= 256 * 1024;
        let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
        let order2_mask = order2_ctx - 1;
        let mut o0 = ByteCountModel::new(1);
        let mut o1 = ByteCountModel::new(256);
        let mut o2 = ByteCountModel::new(order2_ctx);
        let mut enc = RansByteEncoder::new();
        let mut p_2 = 0u8;
        let mut p_1 = 0u8;
        let mut e_triples = Vec::new();
        for &b in &v[..n] {
            let c1 = usize::from(p_1);
            let c2 = hash2(p_1, p_2) & order2_mask;
            let ord = pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2));
            let wr = match ord {
                Order::Order0 => o0.walk_dist(0, Some(b), None),
                Order::Order1 => o1.walk_dist(c1, Some(b), None),
                Order::Order2 => o2.walk_dist(c2, Some(b), None),
                Order::Order3 => unreachable!("pick_order never returns Order3"),
            };
            enc.encode_fc(wr.freq, wr.cum);
            e_triples.push((b, wr.freq, wr.cum));
            o0.update(0, b);
            o1.update(c1, b);
            o2.update(c2, b);
            p_2 = p_1;
            p_1 = b;
        }
        let comp = enc.finish();

        let mut d0 = ByteCountModel::new(1);
        let mut d1 = ByteCountModel::new(256);
        let mut d2 = ByteCountModel::new(order2_ctx);
        let mut dec = RansByteDecoder::new(&comp).unwrap();
        let mut q_2 = 0u8;
        let mut q_1 = 0u8;
        for i in 0..n {
            let c1 = usize::from(q_1);
            let c2 = hash2(q_1, q_2) & order2_mask;
            let ord = pick_order(d1.total(c1), d1.distinct(c1), d2.total(c2), d2.distinct(c2));
            let cdf = dec.cdf();
            let wr = match ord {
                Order::Order0 => d0.walk_dist(0, None, Some(cdf)),
                Order::Order1 => d1.walk_dist(c1, None, Some(cdf)),
                Order::Order2 => d2.walk_dist(c2, None, Some(cdf)),
                Order::Order3 => unreachable!("pick_order never returns Order3"),
            };
            let (eb, ef, ec) = e_triples[i];
            assert_eq!(
                (wr.sym, wr.freq, wr.cum),
                (eb, ef, ec),
                "walk divergence at i={i} (ord={ord:?}, cdf={cdf})"
            );
            dec.advance(wr.freq, wr.cum);
            d0.update(0, wr.sym);
            d1.update(c1, wr.sym);
            d2.update(c2, wr.sym);
            q_2 = q_1;
            q_1 = wr.sym;
        }
    }

    #[test]
    fn roundtrip_deterministic_streams() {
        roundtrip(b"");
        roundtrip(b"a");
        roundtrip(&vec![0x41; 64]);
        roundtrip(&(0u8..=255).collect::<Vec<_>>());
        roundtrip(&(0u8..=255).cycle().take(4096).collect::<Vec<_>>());
    }

    #[test]
    fn roundtrip_text_like() {
        let text = b"the quick brown fox jumps over the lazy dog. ";
        let mut v = Vec::new();
        for _ in 0..50 {
            v.extend_from_slice(text);
        }
        roundtrip(&v);
    }

    #[test]
    fn roundtrip_random() {
        let mut v = vec![0u8; 8192];
        let mut x = 0x9E37_79B9u32;
        for b in &mut v {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        roundtrip(&v);
    }

    #[test]
    fn compress_is_smaller_on_redundant() {
        let data: Vec<u8> = std::iter::repeat(b'x').take(4096).collect();
        let comp = compress_block(&data);
        assert!(comp.len() < data.len(), "redundant data should shrink");
    }

    #[test]
    fn order_selection_uses_concentration() {
        assert_eq!(pick_order(0, 0, 0, 0), Order::Order0);
        assert_eq!(pick_order(1, 1, 0, 0), Order::Order0, "t1 below MIN_T1");
        assert_eq!(pick_order(3, 2, 4, 4), Order::Order1, "order-2 too scattered");
        assert_eq!(
            pick_order(3, 2, 4, 1),
            Order::Order2,
            "concentrated order-2 wins at equal scale"
        );
    }

    #[test]
    fn walk_dist_avx2_matches_scalar_encode_target() {
        let mut m = ByteCountModel::new(1);
        // Non-power-of-two total so AVX2 may run.
        for b in [1u8, 2, 3, 5, 8, 13, 21, 34, 55, 89] {
            for _ in 0..3 {
                m.update(0, b);
            }
        }
        assert!(!u64::from(m.total(0)).is_power_of_two());
        for b in 0..=255u8 {
            let scalar = m.walk_dist_scalar_only(0, Some(b), None);
            let dispatched = m.walk_dist(0, Some(b), None);
            assert_eq!(
                (dispatched.sym, dispatched.freq, dispatched.cum),
                (scalar.sym, scalar.freq, scalar.cum),
                "dispatch≠scalar for target={b}"
            );
            #[cfg(all(target_arch = "x86_64", not(feature = "no_avx2")))]
            if let Some(avx) = m.walk_dist_avx2_only(0, Some(b), None) {
                assert_eq!(
                    (avx.sym, avx.freq, avx.cum),
                    (scalar.sym, scalar.freq, scalar.cum),
                    "avx2≠scalar for target={b}"
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                let neon = m.walk_dist_neon_only(0, Some(b), None);
                assert_eq!(
                    (neon.sym, neon.freq, neon.cum),
                    (scalar.sym, scalar.freq, scalar.cum),
                    "neon≠scalar for target={b}"
                );
            }
        }
    }

    #[test]
    fn walk_dist_avx2_matches_scalar_decode_cdf() {
        let mut m = ByteCountModel::new(1);
        for b in 0..=255u8 {
            m.update(0, b);
            m.update(0, b.wrapping_mul(3));
        }
        if u64::from(m.total(0)).is_power_of_two() {
            m.update(0, 7);
        }
        assert!(!u64::from(m.total(0)).is_power_of_two());
        for cdf in (0..BYTE_SCALE).step_by(17) {
            let scalar = m.walk_dist_scalar_only(0, None, Some(cdf));
            let dispatched = m.walk_dist(0, None, Some(cdf));
            assert_eq!(
                (dispatched.sym, dispatched.freq, dispatched.cum),
                (scalar.sym, scalar.freq, scalar.cum),
                "dispatch≠scalar for cdf={cdf}"
            );
            #[cfg(all(target_arch = "x86_64", not(feature = "no_avx2")))]
            if let Some(avx) = m.walk_dist_avx2_only(0, None, Some(cdf)) {
                assert_eq!(
                    (avx.sym, avx.freq, avx.cum),
                    (scalar.sym, scalar.freq, scalar.cum),
                    "avx2≠scalar for cdf={cdf}"
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                let neon = m.walk_dist_neon_only(0, None, Some(cdf));
                assert_eq!(
                    (neon.sym, neon.freq, neon.cum),
                    (scalar.sym, scalar.freq, scalar.cum),
                    "neon≠scalar for cdf={cdf}"
                );
            }
        }
    }

    #[test]
    fn walk_dist_avx2_disabled_for_power_of_two_total() {
        let mut m = ByteCountModel::new(1);
        // Exactly 8 updates of one symbol → total 8 = 2^3.
        for _ in 0..8 {
            m.update(0, b'A');
        }
        assert!(u64::from(m.total(0)).is_power_of_two());
        #[cfg(all(target_arch = "x86_64", not(feature = "no_avx2")))]
        assert!(
            m.walk_dist_avx2_only(0, Some(b'A'), None).is_none(),
            "AVX2 must refuse power-of-two totals"
        );
        let scalar = m.walk_dist_scalar_only(0, Some(b'A'), None);
        let dispatched = m.walk_dist(0, Some(b'A'), None);
        assert_eq!(
            (dispatched.sym, dispatched.freq, dispatched.cum),
            (scalar.sym, scalar.freq, scalar.cum)
        );
    }

    #[test]
    fn walk_dist_avx2_matches_scalar_after_many_updates() {
        let mut m = ByteCountModel::new(1);
        let mut x = 0xC0FF_EE00u32;
        for _ in 0..4096 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            m.update(0, x as u8);
        }
        // Force a non-power-of-two total if needed.
        if u64::from(m.total(0)).is_power_of_two() {
            m.update(0, 7);
        }
        for b in [0u8, 1, 127, 128, 255] {
            let scalar = m.walk_dist_scalar_only(0, Some(b), None);
            let dispatched = m.walk_dist(0, Some(b), None);
            assert_eq!(
                (dispatched.freq, dispatched.cum),
                (scalar.freq, scalar.cum),
                "after many updates target={b}"
            );
        }
    }

    #[test]
    fn walk_dist_parity_under_feature_no_avx2() {
        // Smoke: round-trips still work when this crate is built with no_avx2
        // (CI) or without it (local). The scalar path must always succeed.
        roundtrip(b"parity-smoke-abcdefghijklmnopqrstuvwxyz0123456789");
    }
}