//! Level `-1` / `-1` wire engine: hash-chain LZ77 + optional order-0 rANS.
//!
//! Token stream is LZ4-style (`token`, extra lengths, literals, `u16` offset).
//! [`crate::entropy::side_fse::pack_side_stream`] wraps the tokens when that
//! shrinks the payload. The encoder falls back to a raw copy at a higher layer
//! when the whole block does not shrink.

use crate::entropy::side_fse;
use crate::error::{RcnError, Result};

const HASH_LOG: usize = 16;
const HASH_SIZE: usize = 1 << HASH_LOG;
const MIN_MATCH: usize = 4;
const WINDOW: usize = 65_535;
const MAX_MATCH: usize = 65_535;
const CHAIN: usize = 4;

/// Payload flag: raw LZ tokens.
const FLAG_RAW: u8 = 0;
/// Payload flag: order-0 rANS over the token stream.
const FLAG_FSE: u8 = 1;

#[inline]
fn hash4(data: &[u8], i: usize) -> usize {
    let v = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_LOG)) as usize
}

#[inline]
fn match_len(data: &[u8], a: usize, b: usize, max: usize) -> usize {
    let mut n = 0;
    while n < max && data[a + n] == data[b + n] {
        n += 1;
    }
    n
}

/// Compress `data` to a METHOD_WIRE payload (never larger-path; caller may copy).
#[must_use]
pub fn compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![FLAG_RAW];
    }
    let tokens = lz_encode(data);
    let (flag, packed) = side_fse::pack_side_stream(&tokens);
    let mut out = Vec::with_capacity(1 + packed.len());
    if flag == side_fse::FLAG_FSE {
        out.push(FLAG_FSE);
    } else {
        out.push(FLAG_RAW);
    }
    out.extend_from_slice(&packed);
    out
}

/// Inverse of [`compress`]. `orig_len` is the original block length.
///
/// # Errors
///
/// Returns [`RcnError::CorruptBlock`] on a truncated or inconsistent stream.
pub fn decompress(src: &[u8], orig_len: usize) -> Result<Vec<u8>> {
    if orig_len == 0 {
        return Ok(Vec::new());
    }
    if src.is_empty() {
        return Err(RcnError::corrupt_block("wire: empty payload"));
    }
    let tokens = match src[0] {
        FLAG_RAW | FLAG_FSE => {
            let flag = if src[0] == FLAG_FSE {
                side_fse::FLAG_FSE
            } else {
                side_fse::FLAG_RAW
            };
            side_fse::unpack_side_stream(flag, &src[1..])
                .map_err(|()| RcnError::corrupt_block("wire: token entropy"))?
        }
        other => {
            return Err(RcnError::corrupt_block(format!(
                "wire: unknown flag {other}"
            )))
        }
    };
    lz_decode(&tokens, orig_len)
}

fn lz_encode(data: &[u8]) -> Vec<u8> {
    let n = data.len();
    let mut out = Vec::with_capacity(n / 2 + 16);
    let mut table = vec![0u32; HASH_SIZE];
    let mut chain = vec![0u32; HASH_SIZE];
    let mut lit_start = 0usize;
    let mut i = 0usize;

    while i + MIN_MATCH + 1 < n {
        let (best_len, best_off) = find_match(data, i, &table, &chain);
        if best_len < MIN_MATCH {
            insert(data, i, &mut table, &mut chain);
            i += 1;
            continue;
        }

        // One-step lazy: a longer match at i+1 wins.
        let mut take_len = best_len;
        let mut take_off = best_off;
        let mut take_at = i;
        if i + 1 + MIN_MATCH < n {
            let (lazy_len, lazy_off) = find_match(data, i + 1, &table, &chain);
            if lazy_len > best_len {
                take_len = lazy_len;
                take_off = lazy_off;
                take_at = i + 1;
            }
        }

        emit_seq(&mut out, &data[lit_start..take_at], take_off, take_len);
        let match_end = take_at + take_len;
        while i < match_end && i + MIN_MATCH <= n {
            insert(data, i, &mut table, &mut chain);
            i += 1;
        }
        lit_start = match_end;
        i = match_end;
    }

    emit_literals_only(&mut out, &data[lit_start..]);
    out
}

fn find_match(data: &[u8], i: usize, table: &[u32], chain_tbl: &[u32]) -> (usize, u16) {
    let n = data.len();
    if i + MIN_MATCH > n {
        return (0, 0);
    }
    let h = hash4(data, i);
    // `table[h]` is the latest position at this hash; `chain_tbl[h]` is the
    // previous occupant (one extra probe). CHAIN=3 also retries the chain slot
    // once after a failed latest so a stale table entry still yields a match.
    let probes = [table[h], chain_tbl[h], chain_tbl[h]];
    let mut best_len = 0usize;
    let mut best_off = 0u16;
    let mut seen = 0u32;
    for cand in probes.iter().take(CHAIN) {
        if *cand == 0 || *cand == seen {
            continue;
        }
        seen = *cand;
        let j = *cand as usize;
        if j >= i {
            continue;
        }
        let dist = i - j;
        if dist == 0 || dist > WINDOW {
            continue;
        }
        let max = (n - i).min(MAX_MATCH);
        let ml = match_len(data, j, i, max);
        if ml >= MIN_MATCH && ml > best_len {
            best_len = ml;
            best_off = dist as u16;
        }
    }
    (best_len, best_off)
}

fn insert(data: &[u8], i: usize, table: &mut [u32], chain: &mut [u32]) {
    if i + MIN_MATCH > data.len() {
        return;
    }
    let h = hash4(data, i);
    chain[h] = table[h];
    table[h] = i as u32;
}

fn emit_seq(out: &mut Vec<u8>, lits: &[u8], offset: u16, match_len: usize) {
    debug_assert!(offset > 0 && match_len >= MIN_MATCH);
    let extra_match = match_len - MIN_MATCH;
    let lit_code = lits.len().min(15) as u8;
    let match_code = extra_match.min(15) as u8;
    out.push((lit_code << 4) | match_code);
    write_extra_len(out, lits.len());
    out.extend_from_slice(lits);
    out.extend_from_slice(&offset.to_le_bytes());
    write_extra_len(out, extra_match);
}

fn emit_literals_only(out: &mut Vec<u8>, lits: &[u8]) {
    if lits.is_empty() && !out.is_empty() {
        return;
    }
    let lit_code = lits.len().min(15) as u8;
    out.push(lit_code << 4);
    write_extra_len(out, lits.len());
    out.extend_from_slice(lits);
}

fn write_extra_len(out: &mut Vec<u8>, len: usize) {
    if len < 15 {
        return;
    }
    let mut rem = len - 15;
    while rem >= 255 {
        out.push(255);
        rem -= 255;
    }
    out.push(rem as u8);
}

fn lz_decode(src: &[u8], orig_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(orig_len);
    let mut p = 0usize;
    while out.len() < orig_len {
        if p >= src.len() {
            return Err(RcnError::corrupt_block("wire: truncated token"));
        }
        let token = src[p];
        p += 1;
        let mut lit_len = usize::from(token >> 4);
        p = read_extra_len(src, p, &mut lit_len)?;
        if p + lit_len > src.len() {
            return Err(RcnError::corrupt_block("wire: truncated literals"));
        }
        if out.len() + lit_len > orig_len {
            return Err(RcnError::corrupt_block("wire: literal overrun"));
        }
        out.extend_from_slice(&src[p..p + lit_len]);
        p += lit_len;
        if out.len() == orig_len {
            break;
        }
        if p + 2 > src.len() {
            return Err(RcnError::corrupt_block("wire: truncated offset"));
        }
        let offset = u16::from_le_bytes([src[p], src[p + 1]]);
        p += 2;
        if offset == 0 {
            return Err(RcnError::corrupt_block("wire: zero offset"));
        }
        let mut extra = usize::from(token & 0x0F);
        p = read_extra_len(src, p, &mut extra)?;
        let ml = extra + MIN_MATCH;
        let off = usize::from(offset);
        if off > out.len() || out.len() + ml > orig_len {
            return Err(RcnError::corrupt_block("wire: bad match"));
        }
        out.reserve(ml);
        for _ in 0..ml {
            let b = out[out.len() - off];
            out.push(b);
        }
    }
    if out.len() != orig_len {
        return Err(RcnError::corrupt_block("wire: length mismatch"));
    }
    Ok(out)
}

fn read_extra_len(src: &[u8], mut p: usize, len: &mut usize) -> Result<usize> {
    if *len < 15 {
        return Ok(p);
    }
    loop {
        if p >= src.len() {
            return Err(RcnError::corrupt_block("wire: truncated extra length"));
        }
        let b = src[p];
        p += 1;
        *len += usize::from(b);
        if b != 255 {
            return Ok(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(data: &[u8]) {
        let c = compress(data);
        let back = decompress(&c, data.len()).expect("decompress");
        assert_eq!(back, data);
    }

    #[test]
    fn empty_and_tiny() {
        rt(b"");
        rt(b"a");
        rt(b"abcd");
        rt(b"abcde");
    }

    #[test]
    fn redundant_round_trip() {
        let data = b"rcnrcnrcn".repeat(8_000);
        rt(&data);
        let c = compress(&data);
        assert!(c.len() < data.len() / 4, "got {} vs {}", c.len(), data.len());
    }

    #[test]
    fn mixed_round_trip() {
        let mut v = Vec::new();
        for i in 0..4000u32 {
            v.extend_from_slice(format!("the quick brown fox {i}\n").as_bytes());
        }
        rt(&v);
    }

    #[test]
    fn overlapping_match() {
        let data = b"aaaa".repeat(200);
        rt(&data);
    }

    #[test]
    fn incompressible_round_trip() {
        let mut buf = vec![0u8; 4096];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        rt(&buf);
    }

    #[test]
    fn corrupt_flag_errors() {
        let err = decompress(&[99], 8).unwrap_err();
        assert!(matches!(err, RcnError::CorruptBlock(_)));
    }
}
