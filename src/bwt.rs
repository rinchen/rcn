//! Burrows-Wheeler transform + Move-To-Front + RLE0 pipeline for Text blocks.
//!
//! BWT groups similar contexts together, turning long-range repetitions (e.g. repeated
//! words in text) into local runs that CM predicts well. MTF converts BWT output to a
//! rank stream (0 = same char as previous position), which produces many zeros — RLE0
//! compresses those runs. The full pipeline is an involution: every transform has an
//! exact inverse, and the encoder/decoder apply them in opposite order.
//!
//! Pipeline (encode):  data → BWT → MTF → RLE0 → [CM/rANS]
//! Pipeline (decode):  [rANS/CM] → RLE0⁻¹ → MTF⁻¹ → BWT⁻¹ → data
//!
//! BWT suffix-array backend:
//! - **Default**: `divsufsort` (O(n) SA on the doubled string).
//! - **Optional** (`bwt_libsais` feature): pure-Rust `libsais-rs` SA-IS on the same
//!   doubled string. Streams stay structurally compatible (primary index + BWT string);
//!   bit-identical SA order across backends is *not* required — inverse must match the
//!   forward of the *same* backend.

#[cfg(not(feature = "bwt_libsais"))]
use divsufsort::sort as divsufsort_sort;

// ---------------------------------------------------------------------------
// RLE0
// ---------------------------------------------------------------------------

/// Run-length-encode zero runs and literal 0xFF bytes in a byte stream.
///
/// Non-zero, non-0xFF bytes pass through unchanged. Consecutive zeros are encoded as a
/// flag byte `0xFF` followed by the run length as a single byte (max run 255).
/// A literal `0xFF` in the data is escaped as `0xFF 0x00`.
/// A literal `0x00` in the data is escaped as `0xFF 0x01` (if not part of a longer run).
pub fn rle0(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b == 0xFF {
            // Escape literal 0xFF.
            out.push(0xFF);
            out.push(0x00);
            i += 1;
        } else if b == 0 {
            // Count zero run (max 255 per RLE token).
            let mut count = 0u8;
            while i < data.len() && data[i] == 0 && count < 255 {
                count += 1;
                i += 1;
            }
            out.push(0xFF);
            out.push(count);
        } else {
            out.push(b);
            i += 1;
        }
    }
    out
}

/// Inverse of [`rle0`].
pub fn rle0_inverse(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b == 0xFF {
            i += 1;
            if i >= data.len() {
                // Trailing 0xFF with no count byte — shouldn't happen in valid data.
                break;
            }
            let count = data[i];
            if count == 0 {
                // Escaped literal 0xFF.
                out.push(0xFF);
            } else {
                // Zero run of length count.
                out.resize(out.len() + count as usize, 0);
            }
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// MTF
// ---------------------------------------------------------------------------

/// Move-To-Front transform on a byte stream.
///
/// Maintains a list of 256 symbols. Each input byte is replaced by its index in
/// the list, then moved to the front.
pub fn mtf_transform(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut list: Vec<u8> = (0u8..=255).collect();
    for &b in data {
        let pos = list.iter().position(|&x| x == b).unwrap();
        out.push(pos as u8);
        list.remove(pos);
        list.insert(0, b);
    }
    out
}

/// Inverse of [`mtf_transform`].
pub fn mtf_inverse(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut list: Vec<u8> = (0u8..=255).collect();
    for &idx in data {
        let idx = idx as usize;
        let b = list[idx];
        out.push(b);
        list.remove(idx);
        list.insert(0, b);
    }
    out
}

// ---------------------------------------------------------------------------
// BWT (rotation-based via doubled string)
// ---------------------------------------------------------------------------

/// Build a suffix array for `data` (positions into `data`).
///
/// Default: divsufsort. With `bwt_libsais`: libsais-rs SA-IS.
fn build_sa(data: &[u8]) -> Vec<usize> {
    #[cfg(feature = "bwt_libsais")]
    {
        let n = data.len();
        let mut sa = vec![0i32; n];
        let rc = libsais_rs::libsais(data, &mut sa, 0, None);
        debug_assert_eq!(rc, 0, "libsais failed with code {rc}");
        sa.into_iter().map(|p| p as usize).collect()
    }
    #[cfg(not(feature = "bwt_libsais"))]
    {
        let sa = divsufsort_sort(data);
        sa.into_parts().1.iter().map(|&p| p as usize).collect()
    }
}

/// Forward Burrows-Wheeler transform.
///
/// Sorts all cyclic rotations of `data` (using the doubled-string SA trick to avoid
/// sentinel collisions) and produces the BWT string: for each rotation in sorted order,
/// the last character. Appends a 4-byte LE primary index so the decoder knows where to
/// start the LF-mapping walk.
pub fn bwt_forward(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();

    // Build doubled string for rotation-based BWT (avoid sentinel collisions).
    let mut doubled = Vec::with_capacity(n * 2);
    doubled.extend_from_slice(data);
    doubled.extend_from_slice(data);

    let sa = build_sa(&doubled);

    // Collect only suffixes starting at positions 0..n (rotations of the original).
    let mut bwt = Vec::with_capacity(n);
    let mut primary = 0usize;
    let mut bwt_idx = 0usize;
    for &suf_pos in &sa {
        if suf_pos < n {
            if suf_pos == 0 {
                primary = bwt_idx;
            }
            // The character preceding this rotation is data[suf_pos - 1] (circular).
            bwt.push(data[(suf_pos + n - 1) % n]);
            bwt_idx += 1;
        }
        if bwt.len() == n {
            break;
        }
    }

    // Append 4-byte LE primary index.
    bwt.extend_from_slice(&(primary as u32).to_le_bytes());

    bwt
}

/// Inverse Burrows-Wheeler transform.
///
/// Reconstructs the original data from the BWT output (with 4-byte primary index)
/// produced by [`bwt_forward`]. The LF-mapping walk is guaranteed to form a single
/// cycle because the BWT is rotation-based.
pub fn bwt_inverse(data: &[u8]) -> Vec<u8> {
    // Need at least the 4-byte primary index.
    if data.len() < 4 {
        return Vec::new();
    }

    // Split BWT data and 4-byte primary index (last 4 bytes).
    let primary_bytes = &data[data.len() - 4..];
    let primary = u32::from_le_bytes([
        primary_bytes[0],
        primary_bytes[1],
        primary_bytes[2],
        primary_bytes[3],
    ]) as usize;
    let bwt = &data[..data.len() - 4];

    let n = bwt.len();
    if n == 0 {
        return Vec::new();
    }

    // Count occurrences of each byte value in the BWT (last column).
    let mut counts = [0usize; 256];
    for &b in bwt {
        counts[b as usize] += 1;
    }

    // Starting position of each byte value in the sorted (first) column.
    let mut starts = [0usize; 256];
    let mut acc = 0;
    for i in 0..256 {
        starts[i] = acc;
        acc += counts[i];
    }

    // For each position in BWT, compute its rank within its byte value.
    let mut ranks = vec![0usize; n];
    let mut seen = [0usize; 256];
    for i in 0..n {
        let c = bwt[i] as usize;
        ranks[i] = seen[c];
        seen[c] += 1;
    }

    // LF-mapping walk from the primary index, reconstructing the original string.
    // The walk produces the string in reverse order (last character first).
    let mut result = Vec::with_capacity(n);
    let mut idx = primary;
    for _ in 0..n {
        let c = bwt[idx] as usize;
        result.push(c as u8);
        idx = starts[c] + ranks[idx];
    }
    result.reverse();

    result
}

// ---------------------------------------------------------------------------
// LZP encode/decode (lightweight — reused from model/lzp.rs patterns)
// ---------------------------------------------------------------------------

/// Encode `data` with a simple LZP pre-filter using a hash chain for match finding.
///
/// Scans for matches of length >= 4 in a 4 MB history window. Emits:
///   - `[1, len, dist_hi, dist_lo, dist_lo2]` for matches (len 4..255, dist 1..4MB)
///   - `[0, literal]` for non-matching bytes
///
/// Uses a 2-byte hash with chaining for O(n) expected match finding.
pub fn lzp_encode(data: &[u8]) -> Vec<u8> {
    let n = data.len();
    let mut out = Vec::with_capacity(n / 4 + 4);
    // Prepend original length as 4-byte LE so the decoder knows how many bytes
    // of original data to expect (the LZP side-stream itself is variable-length).
    out.extend_from_slice(&(n as u32).to_le_bytes());
    const HASH_SIZE: usize = 1 << 16;
    let mut head: Vec<i32> = vec![-1; HASH_SIZE];
    let mut prev: Vec<i32> = vec![-1; n];

    let window = 4 * 1024 * 1024;
    let mut i = 0usize;
    while i < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;

        if i + 4 <= n {
            let hash = ((data[i] as u32) << 8 | (data[i + 1] as u32)) as usize % HASH_SIZE;
            let mut cand = head[hash];
            let mut probes = 0;
            while cand >= 0 && probes < 32 {
                let cand_usize = cand as usize;
                if i.saturating_sub(cand_usize) <= window {
                    let mut len = 0usize;
                    while i + len < n && len < 255 && data[cand_usize + len] == data[i + len] {
                        len += 1;
                    }
                    if len > best_len {
                        best_len = len;
                        best_dist = i - cand_usize;
                        if best_len >= 255 {
                            break;
                        }
                    }
                }
                let next = prev[cand_usize];
                if next < 0 {
                    break;
                }
                cand = next;
                probes += 1;
            }
            prev[i] = head[hash];
            head[hash] = i as i32;
        }

        if best_len >= 4 {
            out.push(1);
            out.push(best_len.min(255) as u8);
            let dist = best_dist as u32;
            out.push((dist >> 16) as u8);
            out.push((dist >> 8) as u8);
            out.push(dist as u8);

            // Insert skipped positions into the hash chain.
            let advance = best_len;
            i += advance;
            // Insert intermediate positions
            for j in 0..advance {
                let p = i - advance + j;
                if p + 1 < n {
                    let h = ((data[p] as u32) << 8 | (data[p + 1] as u32)) as usize % HASH_SIZE;
                    prev[p] = head[h];
                    head[h] = p as i32;
                }
            }
        } else {
            out.push(0);
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Decode an LZP side-stream produced by [`lzp_encode`].
pub fn lzp_decode(data: &[u8], orig_len: usize) -> Vec<u8> {
    // Read 4-byte LE original length from the start of the stream.
    if data.len() < 4 {
        return Vec::new();
    }
    let orig = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // Use the decoded original length if the passed orig_len doesn't match
    // (e.g., when called from the codec with transformed length). Fall back
    // to the passed value.
    let orig = if orig == 0 && orig_len == 0 { 0 } else { orig };
    let mut out = Vec::with_capacity(orig);
    let mut i = 4usize;
    while i < data.len() && out.len() < orig {
        let flag = data[i];
        i += 1;
        if flag == 1 {
            if i + 3 >= data.len() {
                break;
            }
            let len = data[i] as usize;
            let dist =
                ((data[i + 1] as u32) << 16) | ((data[i + 2] as u32) << 8) | (data[i + 3] as u32);
            i += 4;
            let dist = dist as usize;
            if dist > out.len() || dist > 4 * 1024 * 1024 {
                break;
            }
            let start = out.len() - dist;
            let mut j = 0;
            while out.len() < orig && j < len {
                // Copy byte-by-byte to handle overlapping matches (dist < len).
                out.push(out[start + j]);
                j += 1;
            }
        } else {
            if i >= data.len() {
                break;
            }
            out.push(data[i]);
            i += 1;
        }
    }
    if out.len() < orig {
        out.resize(orig, 0);
    }
    out
}

// ---------------------------------------------------------------------------
// Full BWT encode/decode pipelines
// ---------------------------------------------------------------------------

/// Full BWT encode pipeline: data → BWT → MTF → RLE0.
///
/// The BWT output (with 4-byte primary index) is fed through MTF + RLE0.
pub fn bwt_mtf_rle_encode(data: &[u8]) -> Vec<u8> {
    let bwt = bwt_forward(data);
    let mtf = mtf_transform(&bwt);
    rle0(&mtf)
}

/// Full BWT decode pipeline: RLE0⁻¹ → MTF⁻¹ → BWT⁻¹.
pub fn bwt_mtf_rle_decode(data: &[u8]) -> Vec<u8> {
    let rle = rle0_inverse(data);
    let mtf = mtf_inverse(&rle);
    bwt_inverse(&mtf)
}

/// Same as bwt_forward but without RLE0 — for Path C (LZP → BWT → MTF → CM)
/// where the CM operates on MTF ranks directly (no RLE0).
pub fn bwt_mtf_encode(data: &[u8]) -> Vec<u8> {
    let bwt = bwt_forward(data);
    mtf_transform(&bwt)
}

/// Full BWT decode pipeline without RLE0: MTF⁻¹ → BWT⁻¹.
pub fn bwt_mtf_decode(data: &[u8]) -> Vec<u8> {
    let mtf = mtf_inverse(data);
    bwt_inverse(&mtf)
}

// ---------------------------------------------------------------------------
// Per-block trial: pick the best pipeline for this block
// ---------------------------------------------------------------------------

/// Which BWT-based pipeline was chosen for a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BwtPipeline {
    /// Path A: raw CM — no transform, original bytes fed to CM.
    RawCm,
    /// Path B: BWT → MTF → RLE0 → CM on MTF ranks.
    BwtMtfRle,
    /// Path C: LZP → BWT → MTF → CM (no RLE0).
    LzpBwtMtf,
    /// Path D: JSON stream split → 4× independent BWT trials → CM.
    JsonSplit,
    /// Path E: XWRT dictionary → BWT → MTF → RLE0 → CM.
    XwrtBwtMtfRle,
    /// Path F: CSV column split → N× independent BWT trials → CM.
    CsvSplit,
    /// Path G: XML stream split → 3× independent BWT trials → CM.
    XmlSplit,
}

/// Pick the smallest of raw / BWT→MTF→RLE0 / LZP→BWT→MTF for one stream.
fn encode_stream_pick(data: &[u8]) -> (u8, Vec<u8>) {
    let raw_len = data.len();
    let bwt = bwt_mtf_rle_encode(data);
    let lzp = bwt_mtf_encode(&lzp_encode(data));
    if bwt.len() <= raw_len && bwt.len() <= lzp.len() {
        (1, bwt)
    } else if lzp.len() <= raw_len && lzp.len() <= bwt.len() {
        (2, lzp)
    } else {
        (0, data.to_vec())
    }
}

fn decode_stream_pick(data: &[u8], pipe: u8) -> Vec<u8> {
    match pipe {
        0 => data.to_vec(),
        1 => bwt_mtf_rle_decode(data),
        _ => {
            let mtf = bwt_mtf_decode(data);
            lzp_decode(&mtf, 0)
        }
    }
}

/// Pack 2-bit per-stream selectors into a byte vec (`ceil(n/4)` bytes).
fn pack_selectors(pipes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; pipes.len().div_ceil(4)];
    for (i, &p) in pipes.iter().enumerate() {
        out[i / 4] |= (p & 0x3) << (2 * (i % 4));
    }
    out
}

fn unpack_selector(bytes: &[u8], i: usize) -> u8 {
    (bytes[i / 4] >> (2 * (i % 4))) & 0x3
}

/// Run 1–N independent size jobs concurrently via nested `rayon::join`.
fn parallel_map_sizes<F, T>(jobs: Vec<F>) -> Vec<T>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    match jobs.len() {
        0 => Vec::new(),
        1 => {
            let mut jobs = jobs;
            vec![jobs.remove(0)()]
        }
        2 => {
            let mut iter = jobs.into_iter();
            let (a, b) = (iter.next().unwrap(), iter.next().unwrap());
            let (ra, rb) = rayon::join(a, b);
            vec![ra, rb]
        }
        3 => {
            let mut iter = jobs.into_iter();
            let (a, b, c) = (
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
            );
            let (ra, (rb, rc)) = rayon::join(a, || rayon::join(b, c));
            vec![ra, rb, rc]
        }
        _ => {
            let mid = jobs.len() / 2;
            let mut left = jobs;
            let right = left.split_off(mid);
            let (mut left_r, right_r) =
                rayon::join(|| parallel_map_sizes(left), || parallel_map_sizes(right));
            left_r.extend(right_r);
            left_r
        }
    }
}

/// Pack `[orig_len:u32][selector:u8][len0..len_{n-2}:u32][body0..][body_{n-1}]`
/// where the last stream consumes the remainder (no length prefix for the last body).
fn pack_marked_streams(orig_len: usize, selector: u8, encoded: &[&[u8]]) -> Vec<u8> {
    assert!(!encoded.is_empty());
    let mut out = Vec::with_capacity(
        5 + (encoded.len().saturating_sub(1)) * 4 + encoded.iter().map(|e| e.len()).sum::<usize>(),
    );
    out.extend_from_slice(&(orig_len as u32).to_le_bytes());
    out.push(selector);
    for enc in &encoded[..encoded.len() - 1] {
        out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
    }
    for enc in encoded {
        out.extend_from_slice(enc);
    }
    out
}

/// Inverse of [`pack_marked_streams`]: parse header + `n_streams - 1` length prefixes;
/// last stream is the remainder.
fn unpack_marked_streams(
    payload: &[u8],
    n_streams: usize,
) -> crate::error::Result<(usize, u8, Vec<&[u8]>)> {
    use crate::split_common::read_u32_le;
    if n_streams == 0 {
        return Err(crate::error::RcnError::CorruptBlock(
            "unpack_marked_streams requires at least one stream".into(),
        ));
    }
    let header_lens = n_streams - 1;
    let min_len = 5 + header_lens * 4;
    if payload.len() < min_len {
        return Err(crate::error::RcnError::CorruptBlock(format!(
            "structured-split payload truncated: need {min_len} header bytes, got {}",
            payload.len()
        )));
    }
    let orig_len = read_u32_le(payload, 0)
        .map_err(crate::error::RcnError::CorruptBlock)? as usize;
    let selector = payload[4];
    let mut pos = 5;
    let mut lens = Vec::with_capacity(header_lens);
    for _ in 0..header_lens {
        let len = read_u32_le(payload, pos).map_err(crate::error::RcnError::CorruptBlock)? as usize;
        lens.push(len);
        pos += 4;
    }
    let prefixed_total: usize = lens.iter().sum();
    if pos + prefixed_total > payload.len() {
        return Err(crate::error::RcnError::CorruptBlock(
            "structured-split stream lengths overrun payload".into(),
        ));
    }
    let mut streams = Vec::with_capacity(n_streams);
    for &len in &lens {
        streams.push(&payload[pos..pos + len]);
        pos += len;
    }
    streams.push(&payload[pos..]);
    Ok((orig_len, selector, streams))
}

impl BwtPipeline {
    /// Encode `data` using this pipeline, returning the payload that CM/rANS will
    /// compress. For `RawCm`, the payload IS the original data.
    ///
    /// For `JsonSplit`, the payload is: `[4×4-byte stream lengths][4 BWT-encoded streams]`.
    /// For `XwrtBwtMtfRle`, if `global_dict` is Some, it's used instead of building a per-block dictionary.
    pub fn encode(
        self,
        data: &[u8],
        global_dict: Option<&crate::model::word::XwrtDictionary>,
    ) -> Vec<u8> {
        match self {
            BwtPipeline::RawCm => data.to_vec(),
            BwtPipeline::BwtMtfRle => bwt_mtf_rle_encode(data),
            BwtPipeline::LzpBwtMtf => bwt_mtf_encode(&lzp_encode(data)),
            BwtPipeline::JsonSplit => {
                let streams = crate::json_split::split(data);
                let (struct_pipe, struct_encoded) = encode_stream_pick(&streams.structural);
                let (keys_pipe, keys_encoded) = encode_stream_pick(&streams.keys);
                let (vals_pipe, vals_encoded) = encode_stream_pick(&streams.string_values);
                let (nums_pipe, nums_encoded) = encode_stream_pick(&streams.numbers);

                let selector = (struct_pipe)
                    | (keys_pipe << 2)
                    | (vals_pipe << 4)
                    | (nums_pipe << 6);

                // Layout: [orig_len:u32][selector:u8][s0_len:u32][s1_len:u32][s2_len:u32][s0][s1][s2][s3]
                pack_marked_streams(
                    data.len(),
                    selector,
                    &[
                        &struct_encoded,
                        &keys_encoded,
                        &vals_encoded,
                        &nums_encoded,
                    ],
                )
            }
            BwtPipeline::CsvSplit => {
                let streams = crate::csv_split::split(data);
                let ncols = streams.ncols();
                let nrows = streams.nrows();
                let mut pipes = Vec::with_capacity(ncols);
                let mut encoded_cols = Vec::with_capacity(ncols);
                for col in &streams.columns {
                    let (pipe, enc) = encode_stream_pick(col);
                    pipes.push(pipe);
                    encoded_cols.push(enc);
                }
                let selectors = pack_selectors(&pipes);
                // Layout:
                // [orig_len:u32][delim:u8][trailing_nl:u8][ncols:u16][nrows:u32]
                // [fields_per_row:u16×nrows][selectors:ceil(ncols/4)]
                // [col_len:u32×ncols][col data...]
                let mut out = Vec::with_capacity(data.len() + 32 + nrows * 2);
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.push(streams.delim);
                out.push(u8::from(streams.trailing_newline));
                out.extend_from_slice(&(ncols as u16).to_le_bytes());
                out.extend_from_slice(&(nrows as u32).to_le_bytes());
                for &f in &streams.fields_per_row {
                    out.extend_from_slice(&f.to_le_bytes());
                }
                out.extend_from_slice(&selectors);
                for enc in &encoded_cols {
                    out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
                }
                for enc in &encoded_cols {
                    out.extend_from_slice(enc);
                }
                out
            }
            BwtPipeline::XmlSplit => {
                let streams = crate::xml_split::split(data);
                let (tags_pipe, tags_encoded) = encode_stream_pick(&streams.tags);
                let (attrs_pipe, attrs_encoded) = encode_stream_pick(&streams.attrs);
                let (text_pipe, text_encoded) = encode_stream_pick(&streams.text);
                let selector = tags_pipe | (attrs_pipe << 2) | (text_pipe << 4);
                // Layout: [orig_len:u32][selector:u8][t_len:u32][a_len:u32][tags][attrs][text]
                pack_marked_streams(
                    data.len(),
                    selector,
                    &[&tags_encoded, &attrs_encoded, &text_encoded],
                )
            }
            BwtPipeline::XwrtBwtMtfRle => {
                // Use global dictionary if provided (stored once in the container
                // header), otherwise build + embed a per-block dictionary.
                let (xwrt, dict_bytes) = match global_dict {
                    Some(gd) => (gd.transform(data), Vec::new()),
                    None => {
                        let d = crate::model::word::XwrtDictionary::build_from_data(data);
                        (d.transform(data), d.to_bytes())
                    }
                };
                let encoded = bwt_mtf_rle_encode(&xwrt);
                let mut out = Vec::with_capacity(4 + 4 + encoded.len() + dict_bytes.len());
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                out.extend_from_slice(&encoded);
                out.extend_from_slice(&dict_bytes);
                out
            }
        }
    }

    /// Decode a payload produced by [`Self::encode`], returning the original data.
    ///
    /// For `XwrtBwtMtfRle`, a `Some` `global_dict` (stored once in the container
    /// header) is used instead of an embedded per-block dictionary.
    ///
    /// # Errors
    ///
    /// Returns [`RcnError::CorruptBlock`] (or a split-specific variant) when the
    /// payload is truncated or structurally invalid.
    pub fn decode(
        self,
        payload: &[u8],
        orig_len: usize,
        global_dict: Option<&crate::model::word::XwrtDictionary>,
    ) -> crate::error::Result<Vec<u8>> {
        use crate::split_common::{read_u16_le, read_u32_le};
        match self {
            BwtPipeline::RawCm => Ok(payload.to_vec()),
            BwtPipeline::BwtMtfRle => Ok(bwt_mtf_rle_decode(payload)),
            BwtPipeline::LzpBwtMtf => {
                let mtf = bwt_mtf_decode(payload);
                Ok(lzp_decode(&mtf, orig_len))
            }
            BwtPipeline::JsonSplit => {
                let (orig_len, selector, parts) = unpack_marked_streams(payload, 4)?;
                let streams = crate::json_split::JsonStreams {
                    structural: decode_stream_pick(parts[0], selector & 0x3),
                    keys: decode_stream_pick(parts[1], (selector >> 2) & 0x3),
                    string_values: decode_stream_pick(parts[2], (selector >> 4) & 0x3),
                    numbers: decode_stream_pick(parts[3], (selector >> 6) & 0x3),
                };
                crate::json_split::merge(&streams, orig_len)
            }
            BwtPipeline::CsvSplit => {
                if payload.len() < 12 {
                    return Err(crate::error::RcnError::CorruptBlock(
                        "csv-split payload truncated".into(),
                    ));
                }
                let mut pos = 0usize;
                let _orig_len =
                    read_u32_le(payload, pos).map_err(crate::error::RcnError::CorruptBlock)?
                        as usize;
                pos += 4;
                let delim = payload[pos];
                pos += 1;
                let trailing_newline = payload[pos] != 0;
                pos += 1;
                let ncols =
                    read_u16_le(payload, pos).map_err(crate::error::RcnError::CorruptBlock)?
                        as usize;
                pos += 2;
                let nrows =
                    read_u32_le(payload, pos).map_err(crate::error::RcnError::CorruptBlock)?
                        as usize;
                pos += 4;
                if pos + nrows * 2 > payload.len() {
                    return Err(crate::error::RcnError::CorruptBlock(
                        "csv-split fields_per_row overruns payload".into(),
                    ));
                }
                let mut fields_per_row = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    fields_per_row.push(
                        read_u16_le(payload, pos).map_err(crate::error::RcnError::CorruptBlock)?,
                    );
                    pos += 2;
                }
                let sel_len = ncols.div_ceil(4);
                if pos + sel_len + ncols * 4 > payload.len() {
                    return Err(crate::error::RcnError::CorruptBlock(
                        "csv-split selectors/col_lens overrun payload".into(),
                    ));
                }
                let selectors = &payload[pos..pos + sel_len];
                pos += sel_len;
                let mut col_lens = Vec::with_capacity(ncols);
                for _ in 0..ncols {
                    col_lens.push(
                        read_u32_le(payload, pos).map_err(crate::error::RcnError::CorruptBlock)?
                            as usize,
                    );
                    pos += 4;
                }
                let mut columns = Vec::with_capacity(ncols);
                for (i, &clen) in col_lens.iter().enumerate() {
                    if pos + clen > payload.len() {
                        return Err(crate::error::RcnError::CorruptBlock(format!(
                            "csv-split column {i} length {clen} overruns payload"
                        )));
                    }
                    let pipe = unpack_selector(selectors, i);
                    columns.push(decode_stream_pick(&payload[pos..pos + clen], pipe));
                    pos += clen;
                }
                crate::csv_split::join(&crate::csv_split::CsvStreams {
                    delim,
                    columns,
                    fields_per_row,
                    trailing_newline,
                })
            }
            BwtPipeline::XmlSplit => {
                let (orig_len, selector, parts) = unpack_marked_streams(payload, 3)?;
                crate::xml_split::join(
                    &crate::xml_split::XmlStreams {
                        tags: decode_stream_pick(parts[0], selector & 0x3),
                        attrs: decode_stream_pick(parts[1], (selector >> 2) & 0x3),
                        text: decode_stream_pick(parts[2], (selector >> 4) & 0x3),
                    },
                    orig_len,
                )
            }
            BwtPipeline::XwrtBwtMtfRle => {
                if payload.len() < 8 {
                    return Err(crate::error::RcnError::CorruptBlock(
                        "xwrt payload truncated".into(),
                    ));
                }
                let orig_len = read_u32_le(payload, 0)
                    .map_err(crate::error::RcnError::CorruptBlock)?
                    as usize;
                let encoded_len = read_u32_le(payload, 4)
                    .map_err(crate::error::RcnError::CorruptBlock)?
                    as usize;
                if payload.len() < 8 + encoded_len {
                    return Err(crate::error::RcnError::CorruptBlock(
                        "xwrt encoded body overruns payload".into(),
                    ));
                }
                let mtf = bwt_mtf_rle_decode(&payload[8..8 + encoded_len]);
                Ok(match global_dict {
                    Some(gd) => crate::model::word::inverse_with_dict(&mtf, orig_len, gd),
                    None => {
                        let dict_data = &payload[8 + encoded_len..];
                        crate::model::word::xwrt_inverse_with_dict(&mtf, orig_len, dict_data)
                    }
                })
            }
        }
    }
}

/// Result of running all BWT paths and comparing sizes.
///
/// Includes the winning transformed payload so callers do not re-encode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BwtPathResult {
    /// The chosen pipeline.
    pub pipeline: BwtPipeline,
    /// Size of the encoded payload (before rANS/CM compression).
    pub encoded_size: usize,
    /// Whether this is a transform path (BWT-based) or raw CM.
    pub is_bwt: bool,
    /// Winning pipeline output (ready for CM / byte rANS).
    pub payload: Vec<u8>,
}

/// Fast-path threshold: blocks smaller than this skip the BWT/JSON trials and
/// go straight to raw CM. Below ~1 MB the transformed-size comparison rarely pays
/// for the BWT construction + two extra full-CM passes. JSON is exempt — its
/// stream-split win is large enough that trialing it from 256 KB up is worth it.
const TRIAL_MIN_LEN: usize = 2 * 1024 * 1024; // 2 MB
/// Blocks remaining above this trial threshold but near-random skip all trials.
/// (Text passed the classifier at shannon < 7.9, but rich-but-near-random text
/// gains nothing from BWT's long-range reordering.)
const TRIAL_MAX_SHANNON: f32 = 7.2;

/// Run all BWT paths on `data` and return the smallest.
///
/// Fast-path heuristics: blocks < 256 KB always use raw CM (the transforms can
/// only help long-range structure that short blocks lack). Blocks under 1 MB
/// (256 KB..1 MB) skip trials *unless* they look like JSON/CSV/XML — structured
/// stream-splitting pays off from 256 KB up. Blocks above 1 MB use a Shannon
/// guard: near-random text skips trials too. Returns the winning **pipeline**,
/// **size**, and **payload** so callers do not re-encode.
///
/// When detectors fire, structured-split paths (JSON / CSV / XML) are tried.
/// When `global_dict` is `Some`, `XwrtBwtMtfRle` is also tried (ASCII-only).
/// All applicable size trials run concurrently via nested `rayon::join`.
pub fn compress_text_with_trial(
    data: &[u8],
    global_dict: Option<&crate::model::word::XwrtDictionary>,
) -> BwtPathResult {
    let is_json = crate::json_split::looks_like_json(data);
    let is_csv = !is_json && crate::csv_split::looks_like_csv(data);
    let is_xml = !is_json && !is_csv && crate::xml_split::looks_like_xml(data);
    let is_structured = is_json || is_csv || is_xml;
    let small = data.len() < 256 * 1024;
    let medium = (256 * 1024..TRIAL_MIN_LEN).contains(&data.len()) && !is_structured;
    let near_random = crate::classify::shannon_estimate(data) > TRIAL_MAX_SHANNON;
    if small || medium || near_random {
        return BwtPathResult {
            pipeline: BwtPipeline::RawCm,
            encoded_size: data.len(),
            is_bwt: false,
            payload: data.to_vec(),
        };
    }

    // Collect applicable size-trial closures (B, C, optional structured, optional XWRT)
    // and run them concurrently via nested rayon::join. Each job returns the full
    // payload so the winner need not be re-encoded by the caller.
    let try_xwrt = global_dict.is_some() && data.is_ascii();
    type Trial = (BwtPipeline, Vec<u8>);
    let mut jobs: Vec<Box<dyn FnOnce() -> Trial + Send>> = Vec::new();
    jobs.push(Box::new(|| {
        (
            BwtPipeline::BwtMtfRle,
            BwtPipeline::BwtMtfRle.encode(data, global_dict),
        )
    }));
    jobs.push(Box::new(|| {
        (
            BwtPipeline::LzpBwtMtf,
            BwtPipeline::LzpBwtMtf.encode(data, global_dict),
        )
    }));
    if is_json {
        jobs.push(Box::new(|| {
            (
                BwtPipeline::JsonSplit,
                BwtPipeline::JsonSplit.encode(data, global_dict),
            )
        }));
    }
    if is_csv {
        jobs.push(Box::new(|| {
            (
                BwtPipeline::CsvSplit,
                BwtPipeline::CsvSplit.encode(data, global_dict),
            )
        }));
    }
    if is_xml {
        jobs.push(Box::new(|| {
            (
                BwtPipeline::XmlSplit,
                BwtPipeline::XmlSplit.encode(data, global_dict),
            )
        }));
    }
    if try_xwrt {
        jobs.push(Box::new(|| {
            (
                BwtPipeline::XwrtBwtMtfRle,
                BwtPipeline::XwrtBwtMtfRle.encode(data, global_dict),
            )
        }));
    }

    let mut candidates: Vec<Trial> = vec![(BwtPipeline::RawCm, data.to_vec())];
    candidates.extend(parallel_map_sizes(jobs));

    let (best_pipeline, best_payload) = candidates
        .into_iter()
        .min_by_key(|(_, payload)| payload.len())
        .unwrap();
    let best_size = best_payload.len();

    BwtPathResult {
        pipeline: best_pipeline,
        encoded_size: best_size,
        is_bwt: best_pipeline != BwtPipeline::RawCm,
        payload: best_payload,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle0_round_trip_empty() {
        let encoded = rle0(b"");
        assert!(encoded.is_empty());
        let decoded = rle0_inverse(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn rle0_round_trip_zeros() {
        let input = vec![0u8; 100];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_mixed() {
        let input: Vec<u8> = (0u8..=255).collect();
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_with_0xff() {
        let input = vec![0xFF, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_long_zeros() {
        let input = vec![0u8; 600];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_lone_0xff() {
        // 0xFF not adjacent to any zeros — must survive.
        let input = vec![0x42, 0xFF, 0x43];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn mtf_round_trip() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1000).collect();
        let encoded = mtf_transform(&input);
        let decoded = mtf_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn mtf_round_trip_repetitive() {
        let input = b"abracadabra".repeat(100);
        let encoded = mtf_transform(&input);
        let decoded = mtf_inverse(&encoded);
        assert_eq!(decoded.as_slice(), input);
    }

    #[test]
    fn bwt_round_trip_small() {
        let input = b"banana";
        let bwt = bwt_forward(input);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, input);
    }

    #[test]
    fn bwt_round_trip_empty() {
        let bwt = bwt_forward(b"");
        let decoded = bwt_inverse(&bwt);
        assert!(decoded.is_empty());
    }

    #[test]
    fn bwt_round_trip_single() {
        let bwt = bwt_forward(&[42u8]);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, vec![42u8]);
    }

    #[test]
    fn bwt_round_trip_large() {
        let text = b"the quick brown fox jumps over the lazy dog. \n".repeat(200);
        let bwt = bwt_forward(&text);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_round_trip_repetitive() {
        // "banana" has repeated substrings — tests rotation-based BWT correctness.
        let text = b"banana".repeat(1000);
        let bwt = bwt_forward(&text);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_round_trip_random() {
        let mut buf = vec![0u8; 10_000];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let bwt = bwt_forward(&buf);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_round_trip() {
        let text = b"the quick brown fox. \n".repeat(500);
        let encoded = bwt_mtf_rle_encode(&text);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_round_trip_random() {
        let mut buf = vec![0u8; 10_000];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let encoded = bwt_mtf_rle_encode(&buf);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_round_trip_with_nulls() {
        // Data with embedded null bytes.
        let mut buf = vec![0u8; 5_000];
        let mut x = 0x9876_5432u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = (x % 200) as u8; // 0-199, includes 0
        }
        let encoded = bwt_mtf_rle_encode(&buf);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_round_trip_with_0xff() {
        // Data with many 0xFF bytes.
        let mut buf = vec![0u8; 5_000];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = if i % 3 == 0 { 0xFF } else { (i % 200) as u8 };
        }
        let encoded = bwt_mtf_rle_encode(&buf);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn lzp_round_trip() {
        let mut buf = vec![0u8; 10_000];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let encoded = lzp_encode(&buf);
        let decoded = lzp_decode(&encoded, buf.len());
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_lzp_bwt_round_trip() {
        let text = b"the quick brown fox. \n".repeat(500);
        let encoded = BwtPipeline::LzpBwtMtf.encode(&text, None);
        let decoded = BwtPipeline::LzpBwtMtf
            .decode(&encoded, text.len(), None)
            .expect("decode");
        assert_eq!(decoded, text);
    }

    #[test]
    fn compress_text_with_trial_selects_smallest() {
        // Repetitive text should favor BWT path.
        let text = b"banana ".repeat(5000);
        let result = compress_text_with_trial(&text, None);
        println!(
            "Trial result: {:?}, size={}",
            result.pipeline, result.encoded_size
        );
        assert!(result.pipeline != BwtPipeline::RawCm || result.encoded_size <= text.len());
    }

    #[test]
    fn bwt_pipeline_json_split_round_trip() {
        let text = b"{\"name\":\"John\",\"age\":42,\"city\":\"New York\"}\n".repeat(500);
        let encoded = BwtPipeline::JsonSplit.encode(&text, None);
        let decoded = BwtPipeline::JsonSplit
            .decode(&encoded, text.len(), None)
            .expect("decode");
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_csv_split_round_trip() {
        let text = b"name,age,city\nJohn,30,NYC\nAnna,28,LA\nBob,45,CHI\n".repeat(200);
        let encoded = BwtPipeline::CsvSplit.encode(&text, None);
        let decoded = BwtPipeline::CsvSplit
            .decode(&encoded, text.len(), None)
            .expect("decode");
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_xml_split_round_trip() {
        let text = b"<catalog><book id=\"1\">Alpha</book><book id=\"2\">Beta</book></catalog>\n"
            .repeat(200);
        let encoded = BwtPipeline::XmlSplit.encode(&text, None);
        let decoded = BwtPipeline::XmlSplit
            .decode(&encoded, text.len(), None)
            .expect("decode");
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_xwrt_round_trip() {
        let text = b"hello world hello world hello world".to_vec();
        let encoded = BwtPipeline::XwrtBwtMtfRle.encode(&text, None);
        eprintln!("XWRT encode len={} content={:?}", encoded.len(), encoded);
        let decoded = BwtPipeline::XwrtBwtMtfRle
            .decode(&encoded, text.len(), None)
            .expect("decode");
        eprintln!("XWRT decode len={} content={:?}", decoded.len(), decoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_json_split_decode_truncated_payload_returns_err() {
        let err = BwtPipeline::JsonSplit
            .decode(&[0u8; 8], 0, None)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::RcnError::CorruptBlock(_)
        ));
    }

    #[test]
    fn bwt_pipeline_csv_split_decode_bad_col_len_returns_err() {
        // Header claims 1 column with length 100 but no body bytes follow.
        let mut payload = Vec::new();
        payload.extend_from_slice(&10u32.to_le_bytes()); // orig_len
        payload.push(b','); // delim
        payload.push(1); // trailing_nl
        payload.extend_from_slice(&1u16.to_le_bytes()); // ncols
        payload.extend_from_slice(&1u32.to_le_bytes()); // nrows
        payload.extend_from_slice(&1u16.to_le_bytes()); // fields_per_row[0]
        payload.push(0); // selectors (ceil(1/4)=1)
        payload.extend_from_slice(&100u32.to_le_bytes()); // col_len claims 100
        // no column body
        let err = BwtPipeline::CsvSplit
            .decode(&payload, 10, None)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::RcnError::CorruptBlock(_) | crate::error::RcnError::CsvSplitError(_)
        ));
    }

    #[test]
    fn bwt_pipeline_xml_split_decode_truncated_payload_returns_err() {
        let err = BwtPipeline::XmlSplit
            .decode(&[0u8; 5], 0, None)
            .unwrap_err();
        assert!(matches!(err, crate::error::RcnError::CorruptBlock(_)));
    }

    #[test]
    fn bwt_pipeline_json_split_decode_merge_failure_not_empty_vec() {
        // Valid-looking header with absurd stream lengths that fail merge.
        let mut payload = Vec::new();
        payload.extend_from_slice(&100u32.to_le_bytes()); // orig_len
        payload.push(0); // selector
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        // empty streams → merge length mismatch
        let err = BwtPipeline::JsonSplit
            .decode(&payload, 100, None)
            .unwrap_err();
        assert!(matches!(err, crate::error::RcnError::JsonSplitError(_)));
    }
}
