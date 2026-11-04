//! Core codec: glue the classifier, bit models, two-level logistic mixer, rANS backend,
//! and the `RCN1` container into `compress` / `decompress`.
//!
//! ## Two-level mixer hierarchy
//!
//! The mixer uses a CMIX/PAQ8-style three-layer stack:
//!
//! 1. **Bank mixers** (16384 instances): selected by a context hash of
//!    order-1 / order-2 / word-context + bit position. Each bank specializes
//!    its weights to a specific context, avoiding the ~50% saturation that a
//!    single logistic mixer hits on repetitive corpora like dickens.
//! 2. **Global mixer**: a single context-agnostic mixer over the same models.
//! 3. **Master mixer**: blends `[p_bank, p_global, p_lzp_conf]` in logistic space.
//!
//! Only the selected bank + global + master are trained per bit — never all 16384.
//! At block boundaries, weights are **decayed** (not reset), preserving learned
//! structure across the stream.
//!
//! Strategy per block:
//! - `Random` blocks are stored verbatim (copy record, method 0).
//! - `Text` blocks use a text-optimized stack: orders 0–2, Sparse, Exec, Lzp,
//!   PpmdSsm (order-8), WordModel. (IndirectModel is in-tree but not default.)
//! - `Binary` blocks use orders 0–2, Sparse, Exec, LZP, PPM order-3.
//! - `Exec` blocks use orders 0–2, Sparse, LZP, PPM order-3 (no Exec model).
//! - Fallback / unknown (method 1) is the Binary-like heterogeneous stack.
//!   This is also the decoder default for any future method value, so old
//!   streams remain valid.
//!
//! Default compress mode is **Hybrid** (level `-9`): Text/Random use the Fast
//! byte path; Binary/Exec use the Slow bit path above. Level `-1` is the
//! [`crate::wire`] LZ engine; level `-3` is byte CM without BWT trials.
//!
//! ## DP-optimal LZP match pre-pass (default)
//!
//! rcn runs a forward LZP match pre-pass
//! with **DP optimal parsing** and emits explicit `(len, dist)` records for long
//! matches (adaptive min length: Text ≥12, Binary/Exec ≥16).
//! Matched bytes are **skipped** in the rANS stream —
//! only literals (non-matched bytes) are CM-encoded. The decoder reconstructs
//! matched bytes by copying from history.
//!
//! Match selection cost = bits(match_flag) + bits(len) + bits(dist) + residual_cost,
//! where residual_cost is estimated as len × AVG_BITS_PER_BYTE (CM cost per byte).
//! A match is taken when its overhead (33 bits) is less than the CM cost of the
//! matched bytes (len × 4 bits), i.e., len > 8.25. With min length ≥12, matches
//! always save net bits.
//!
//! Method values:
//!   0 = copy, 1 = cm (full stack), 2 = text, 3 = binary, 4 = exec.

use crate::bwt::{self};
use crate::container::{
    read_global_dict, read_members, write_global_dict, write_members, BlockEntry, Header, Member,
    FLAG_GLOBAL_DICT, VERSION, VERSION_V2,
};
use crate::entropy::range::{BitDecoder, BitEncoder};
use crate::entropy::side_fse;
use crate::error::{RcnError, Result};
use crate::model::lzp::Lzp;
use crate::model::mixer_bank::MixerBank;
use crate::model::sse_apm::SseApmCascade;
use crate::model::word::XwrtDictionary;

/// Varint encoding: read/write unsigned LEB128. Most values in the match
/// side-stream (pos deltas, lengths, distances) are small, so varint
/// encoding saves space over fixed-width fields.
fn read_varint(data: &[u8], mut offset: usize) -> Option<(usize, usize)> {
    let mut result: usize = 0;
    let mut shift: usize = 0;
    while offset < data.len() {
        let byte = data[offset];
        result |= ((byte & 0x7F) as usize) << shift;
        offset += 1;
        if (byte & 0x80) == 0 {
            return Some((result, offset));
        }
        shift += 7;
        // Safety: prevent infinite loop on malformed data.
        if shift > 35 {
            return None;
        }
    }
    None
}

fn write_varint(out: &mut Vec<u8>, mut value: usize) {
    loop {
        if value < 0x80 {
            out.push(value as u8);
            return;
        }
        out.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
}

/// Adaptive DP-LZP minimum match length by block kind.
/// Text uses a lower gate (more matches); Binary/Exec keep 16 after a ≥24
/// trial regressed `mr` (~20.6%→27.3%).
#[inline]
fn match_min_len(kind: crate::classify::BlockKind) -> usize {
    match kind {
        crate::classify::BlockKind::Text => 12,
        crate::classify::BlockKind::Binary
        | crate::classify::BlockKind::Exec
        | crate::classify::BlockKind::Random => 16,
    }
}

#[derive(Debug, Clone, Copy)]
struct MatchRun {
    pos: usize,
    len: usize,
    dist: usize,
}

/// Default block size: 64 KiB. `block_size_log = 16`.
pub const DEFAULT_BLOCK_SIZE_LOG: u8 = 16;

/// Container method constants (stored in `BlockEntry::method`).
pub const METHOD_COPY: u8 = 0;
/// Full heterogenous CM stack (legacy / fallback).
pub const METHOD_CM: u8 = 1;
/// Text-optimized CM stack (orders 0–2, Sparse, Exec, Lzp, PpmModel order-3, WordModel).
pub const METHOD_TEXT: u8 = 2;
/// Binary CM stack (orders 0–2, Sparse, Exec, LZP, PPM order-3).
pub const METHOD_BINARY: u8 = 3;
/// Exec-optimized CM stack (orders 0–2, Sparse, LZP, PPM order-3; no Exec model).
pub const METHOD_EXEC: u8 = 4;
/// Text BWT+B path: BWT(1MB) → MTF → RLE0 → CM on MTF ranks.
pub const METHOD_BWT_MTF_RLE: u8 = 5;
/// Text BWT path: LZP(4MB) → BWT → MTF → CM (no RLE0).
pub const METHOD_LZP_BWT_MTF: u8 = 6;
/// Text JSON split path: split JSON into 4 streams → 4× BWT → CM.
pub const METHOD_JSON_SPLIT: u8 = 7;
/// Text XWRT path: XWRT dictionary → BWT → MTF → RLE0 → CM.
pub const METHOD_XWRT_BWT_MTF_RLE: u8 = 13;
/// Text CSV column-split path: split CSV → N× BWT → CM.
pub const METHOD_CSV_SPLIT: u8 = 15;
/// Text XML stream-split path: split XML → 3× BWT → CM.
pub const METHOD_XML_SPLIT: u8 = 16;
/// Byte-level CM (fast path): orders 0–2 count models + byte rANS.
pub const METHOD_BYTE_CM: u8 = 8;
/// Text BWT+RLE0 path, byte-coded (fast).
pub const METHOD_BYTE_BWT_MTF_RLE: u8 = 9;
/// Text LZP→BWT→MTF path, byte-coded (fast).
pub const METHOD_BYTE_LZP_BWT_MTF: u8 = 10;
/// Text JSON split path, byte-coded (fast).
pub const METHOD_BYTE_JSON_SPLIT: u8 = 11;
/// Text XWRT→BWT→MTF→RLE0 path, byte-coded (fast).
pub const METHOD_BYTE_XWRT_BWT_MTF_RLE: u8 = 12;
/// Text CSV split path, byte-coded (fast).
pub const METHOD_BYTE_CSV_SPLIT: u8 = 17;
/// Text XML split path, byte-coded (fast).
pub const METHOD_BYTE_XML_SPLIT: u8 = 18;
/// Exec E8E9 transform, byte-coded (fast).
pub const METHOD_BYTE_EXEC_E8E9: u8 = 14;
/// Level `-1` wire engine: hash-chain LZ77 + optional order-0 rANS.
pub const METHOD_WIRE: u8 = 19;
/// Byte CM on Text without cross-block match history (level `-3`).
pub const METHOD_BYTE_TEXT: u8 = 20;

/// One block's compression summary, reported by [`compress_mode_diag`] for
/// `--verbose` progress output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDiag {
    /// Classifier label for the block.
    pub kind: String,
    /// Raw [`METHOD_*`] byte written to the container.
    pub method: u8,
    /// Uncompressed size of the block.
    pub size_in: usize,
    /// Compressed size of the block.
    pub size_out: usize,
}

/// Human-readable label for a [`METHOD_*`] byte.
#[must_use]
pub fn method_label(method: u8) -> &'static str {
    match method {
        METHOD_COPY => "COPY",
        METHOD_CM => "CM",
        METHOD_TEXT => "CM(text)",
        METHOD_BINARY => "CM(binary)",
        METHOD_EXEC => "CM(exec)",
        METHOD_BWT_MTF_RLE => "BWT→MTF→RLE0→CM",
        METHOD_LZP_BWT_MTF => "LZP→BWT→MTF→CM",
        METHOD_JSON_SPLIT => "JSON-split→BWT→CM",
        METHOD_XWRT_BWT_MTF_RLE => "XWRT→BWT→MTF→RLE0→CM",
        METHOD_CSV_SPLIT => "CSV-split→BWT→CM",
        METHOD_XML_SPLIT => "XML-split→BWT→CM",
        METHOD_BYTE_CM => "byte-CM",
        METHOD_BYTE_BWT_MTF_RLE => "byte-BWT→MTF→RLE0",
        METHOD_BYTE_LZP_BWT_MTF => "byte-LZP→BWT→MTF",
        METHOD_BYTE_JSON_SPLIT => "byte-JSON-split→BWT",
        METHOD_BYTE_XWRT_BWT_MTF_RLE => "byte-XWRT→BWT→MTF→RLE0",
        METHOD_BYTE_CSV_SPLIT => "byte-CSV-split→BWT",
        METHOD_BYTE_XML_SPLIT => "byte-XML-split→BWT",
        METHOD_BYTE_EXEC_E8E9 => "byte-E8E9→CM",
        METHOD_WIRE => "wire-LZ",
        METHOD_BYTE_TEXT => "byte-CM(text)",
        _ => "?",
    }
}

/// Map a chosen BWT pipeline to the container method byte for `mode`.
fn method_for_pipeline(pipeline: bwt::BwtPipeline, mode: CodecMode) -> u8 {
    // Hybrid Text uses Fast method bytes; Hybrid Binary/Exec never calls this.
    let mode = match mode {
        CodecMode::Hybrid | CodecMode::General | CodecMode::Wire => CodecMode::Fast,
        other => other,
    };
    match (mode, pipeline) {
        (CodecMode::Slow, bwt::BwtPipeline::RawCm) => METHOD_TEXT,
        (CodecMode::Slow, bwt::BwtPipeline::BwtMtfRle) => METHOD_BWT_MTF_RLE,
        (CodecMode::Slow, bwt::BwtPipeline::LzpBwtMtf) => METHOD_LZP_BWT_MTF,
        (CodecMode::Slow, bwt::BwtPipeline::JsonSplit) => METHOD_JSON_SPLIT,
        (CodecMode::Slow, bwt::BwtPipeline::XwrtBwtMtfRle) => METHOD_XWRT_BWT_MTF_RLE,
        (CodecMode::Slow, bwt::BwtPipeline::CsvSplit) => METHOD_CSV_SPLIT,
        (CodecMode::Slow, bwt::BwtPipeline::XmlSplit) => METHOD_XML_SPLIT,
        (CodecMode::Fast, bwt::BwtPipeline::RawCm) => METHOD_BYTE_CM,
        (CodecMode::Fast, bwt::BwtPipeline::BwtMtfRle) => METHOD_BYTE_BWT_MTF_RLE,
        (CodecMode::Fast, bwt::BwtPipeline::LzpBwtMtf) => METHOD_BYTE_LZP_BWT_MTF,
        (CodecMode::Fast, bwt::BwtPipeline::JsonSplit) => METHOD_BYTE_JSON_SPLIT,
        (CodecMode::Fast, bwt::BwtPipeline::XwrtBwtMtfRle) => METHOD_BYTE_XWRT_BWT_MTF_RLE,
        (CodecMode::Fast, bwt::BwtPipeline::CsvSplit) => METHOD_BYTE_CSV_SPLIT,
        (CodecMode::Fast, bwt::BwtPipeline::XmlSplit) => METHOD_BYTE_XML_SPLIT,
        (CodecMode::Hybrid | CodecMode::General | CodecMode::Wire, _) => {
            unreachable!("remapped to Fast above")
        }
    }
}

/// Reverse BWT / E8E9 / structured-split transforms after entropy decode.
fn inverse_transform_for_method(
    method: u8,
    decoded: Vec<u8>,
    orig_len: usize,
    global_dict: Option<&XwrtDictionary>,
) -> Result<Vec<u8>> {
    match method {
        METHOD_BYTE_BWT_MTF_RLE | METHOD_BWT_MTF_RLE => Ok(bwt::bwt_mtf_rle_decode(&decoded)),
        METHOD_BYTE_LZP_BWT_MTF | METHOD_LZP_BWT_MTF => {
            let mtf = bwt::bwt_mtf_decode(&decoded);
            Ok(bwt::lzp_decode(&mtf, orig_len))
        }
        METHOD_BYTE_JSON_SPLIT | METHOD_JSON_SPLIT => {
            bwt::BwtPipeline::JsonSplit.decode(&decoded, orig_len, global_dict)
        }
        METHOD_BYTE_XWRT_BWT_MTF_RLE | METHOD_XWRT_BWT_MTF_RLE => {
            bwt::BwtPipeline::XwrtBwtMtfRle.decode(&decoded, orig_len, global_dict)
        }
        METHOD_BYTE_CSV_SPLIT | METHOD_CSV_SPLIT => {
            bwt::BwtPipeline::CsvSplit.decode(&decoded, orig_len, global_dict)
        }
        METHOD_BYTE_XML_SPLIT | METHOD_XML_SPLIT => {
            bwt::BwtPipeline::XmlSplit.decode(&decoded, orig_len, global_dict)
        }
        METHOD_BYTE_EXEC_E8E9 | METHOD_EXEC => Ok(crate::model::e8e9::e8e9_inverse(&decoded)),
        _ => Ok(decoded),
    }
}

/// Encoding strategy for [`compress_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMode {
    /// Bit-level CM: 8–9 models + two-level bank mixer + bit rANS. Level `-19`.
    Slow,
    /// Byte-level CM with BWT trials. Not a numbered level (see [`General`]).
    Fast,
    /// Per-block adaptive: Fast entropy for Text/Random, Slow+DP-LZP for Binary/Exec.
    /// Level `-9` (default).
    Hybrid,
    /// Hash-chain LZ + optional order-0 rANS. Level `-1`.
    Wire,
    /// Byte CM + DP-LZP, no BWT trials. Level `-3`.
    General,
}

/// Decay factor for cross-block weight persistence. 0.995 keeps 99.5% of learned
/// weight structure per block boundary, smoothly transferring context without
/// hard-clearing (which would defeat the 8k-bank specialization).
const BLOCK_DECAY: f32 = 0.995;

/// Compress `buf` into a `RCN1` container using classifier-aware stacks.
///
/// Default mode is [`CodecMode::Hybrid`] (Fast Text, Slow Binary/Exec).
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::Hybrid)
}

/// Compress `buf` using the byte-level (fast) path.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_fast(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::Fast)
}

/// Compress `buf` using Hybrid mode: Fast for Text/Random, Slow for Binary/Exec.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_hybrid(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::Hybrid)
}

/// Compress `buf` using the level `-1` wire engine.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_wire(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::Wire)
}

/// Compress `buf` using the level `-3` general engine (byte CM, no BWT).
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_general(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::General)
}

/// Compress `buf` at a named [`crate::level::Level`].
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_level(buf: &[u8], level: crate::level::Level) -> Result<Vec<u8>> {
    compress_mode(buf, level.mode())
}

pub fn compress_with<F>(buf: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>),
{
    Ok(compress_impl(buf, CodecMode::Slow, build_stack)?.0)
}

/// Compress with an explicit [`CodecMode`]. Both encoder sides of a given
/// mode decode correctly from the method byte in the container.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_mode(buf: &[u8], mode: CodecMode) -> Result<Vec<u8>> {
    Ok(compress_mode_diag(buf, mode)?.0)
}

/// Compress with an explicit [`CodecMode`], also returning a per-block
/// [`BlockDiag`] summary suitable for `--verbose` progress output.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_mode_diag(buf: &[u8], mode: CodecMode) -> Result<(Vec<u8>, Vec<BlockDiag>)> {
    compress_impl(buf, mode, &mut build_stack_for_kind)
}

/// Compress named members into a v2 `RCN1` archive (Hybrid by default via `mode`).
///
/// # Errors
///
/// Returns [`RcnError`] if `members` is empty, a name is unsafe, or encoding fails.
pub fn compress_archive(members: &[(&str, &[u8])], mode: CodecMode) -> Result<Vec<u8>> {
    Ok(compress_archive_diag(members, mode)?.0)
}

/// Compress named members into a v2 archive, also returning per-block [`BlockDiag`].
///
/// # Errors
///
/// Returns [`RcnError`] if `members` is empty, a name is unsafe, or encoding fails.
pub fn compress_archive_diag(
    members: &[(&str, &[u8])],
    mode: CodecMode,
) -> Result<(Vec<u8>, Vec<BlockDiag>)> {
    if members.is_empty() {
        return Err(RcnError::InvalidContainer("nothing to compress".into()));
    }
    for (name, _) in members {
        crate::container::validate_member_name(name)?;
    }
    compress_archive_impl(members, mode, &mut build_stack_for_kind)
}

fn compress_impl<F>(
    buf: &[u8],
    mode: CodecMode,
    build_stack: &mut F,
) -> Result<(Vec<u8>, Vec<BlockDiag>)>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>),
{
    let encoded = encode_members(&[buf], mode, build_stack)?;
    Ok(assemble_container(encoded, VERSION, &[]))
}

fn compress_archive_impl<F>(
    members: &[(&str, &[u8])],
    mode: CodecMode,
    build_stack: &mut F,
) -> Result<(Vec<u8>, Vec<BlockDiag>)>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>),
{
    let bufs: Vec<&[u8]> = members.iter().map(|(_, b)| *b).collect();
    let encoded = encode_members(&bufs, mode, build_stack)?;
    let table: Vec<Member> = members
        .iter()
        .zip(encoded.ranges.iter())
        .map(|((name, _), &(first_block, num_blocks))| Member {
            name: (*name).to_string(),
            first_block,
            num_blocks,
        })
        .collect();
    Ok(assemble_container(encoded, VERSION_V2, &table))
}

struct EncodedArchive {
    global_dict_bytes: Option<Vec<u8>>,
    ranges: Vec<(u32, u32)>,
    entries: Vec<BlockEntry>,
    payloads: Vec<u8>,
    diags: Vec<BlockDiag>,
}

fn assemble_container(
    encoded: EncodedArchive,
    version: u8,
    members: &[Member],
) -> (Vec<u8>, Vec<BlockDiag>) {
    let mut out = Vec::new();
    let header = Header {
        version,
        flags: if encoded.global_dict_bytes.is_some() {
            FLAG_GLOBAL_DICT
        } else {
            0
        },
        block_size_log: DEFAULT_BLOCK_SIZE_LOG,
        num_blocks: encoded.entries.len() as u32,
    };
    header.write(&mut out);
    if let Some(ref dict) = encoded.global_dict_bytes {
        write_global_dict(&mut out, dict);
    }
    if version == VERSION_V2 {
        write_members(&mut out, members).expect("member names validated before encode");
    }
    for e in &encoded.entries {
        e.write(&mut out);
    }
    out.extend_from_slice(&encoded.payloads);
    (out, encoded.diags)
}

fn encode_members<F>(
    bufs: &[&[u8]],
    mode: CodecMode,
    build_stack: &mut F,
) -> Result<EncodedArchive>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>),
{
    use rayon::prelude::*;

    // Phase 1: classify each member independently (do not concatenate first).
    let mut segments: Vec<(usize, usize, usize, crate::classify::BlockKind)> = Vec::new();
    for (mi, buf) in bufs.iter().enumerate() {
        let mut offset = 0usize;
        while offset < buf.len() {
            let kind = crate::classify::classify(&buf[offset..]);
            let block_size = block_size_for_kind(kind, offset, buf.len(), mode);
            let end = (offset + block_size).min(buf.len());
            segments.push((mi, offset, end, kind));
            offset = end;
        }
    }

    let global_dict: Option<XwrtDictionary> = match mode {
        CodecMode::Wire | CodecMode::General => None,
        _ => {
            let mut concat = Vec::new();
            for buf in bufs {
                concat.extend_from_slice(buf);
            }
            Some(XwrtDictionary::build_from_data(&concat))
        }
    };

    let use_slow_for = |kind: crate::classify::BlockKind| match mode {
        CodecMode::Slow => true,
        CodecMode::Fast | CodecMode::General | CodecMode::Wire => false,
        CodecMode::Hybrid => matches!(
            kind,
            crate::classify::BlockKind::Binary | crate::classify::BlockKind::Exec
        ),
    };

    // W3: Fast Text/Random blocks are independent after the global dict (no
    // match hist, no mixer). Encode them in parallel; results stay ordered by
    // segment index for a deterministic container.
    let mut encoded: Vec<Option<(Vec<u8>, u8, usize)>> = vec![None; segments.len()];

    let parallel_idxs: Vec<usize> = segments
        .iter()
        .enumerate()
        .filter(|(_, (_, _, _, kind))| {
            if mode == CodecMode::Wire {
                return true;
            }
            !use_slow_for(*kind)
                && matches!(
                    kind,
                    crate::classify::BlockKind::Text | crate::classify::BlockKind::Random
                )
        })
        .map(|(i, _)| i)
        .collect();

    if parallel_idxs.len() >= 2 {
        let dict_ref = global_dict.as_ref();
        let parallel_out: Vec<(usize, (Vec<u8>, u8, usize))> = parallel_idxs
            .par_iter()
            .map(|&i| {
                let (mi, start, end, kind) = segments[i];
                let mut hist = Vec::new();
                let r = encode_block_fastish(
                    mode,
                    &bufs[mi][start..end],
                    kind,
                    dict_ref,
                    &mut hist,
                );
                (i, r)
            })
            .collect();
        for (i, r) in parallel_out {
            encoded[i] = Some(r);
        }
    } else if let Some(&i) = parallel_idxs.first() {
        let (mi, start, end, kind) = segments[i];
        let mut hist = Vec::new();
        encoded[i] = Some(encode_block_fastish(
            mode,
            &bufs[mi][start..end],
            kind,
            global_dict.as_ref(),
            &mut hist,
        ));
    }

    // Phase 2: serial encode for Slow blocks and Fast Binary/Exec (match hist).
    let mut last_kind: Option<crate::classify::BlockKind> = None;
    let mut last_member: Option<usize> = None;
    let mut models: Vec<crate::model::stack_enum::StackModel> = Vec::new();
    let mut mixer = MixerBank::new(0);
    let mut lzp_idx: Option<usize> = None;
    let mut match_hist: Vec<u8> = Vec::new();

    for (i, &(mi, start, end, kind)) in segments.iter().enumerate() {
        if last_member != Some(mi) {
            last_member = Some(mi);
            last_kind = None;
            match_hist.clear();
        }
        if encoded[i].is_some() {
            // Already done in the parallel Fast Text/Random pass.
            match_hist.clear();
            continue;
        }
        let block_data = &bufs[mi][start..end];
        let use_slow = use_slow_for(kind);

        if use_slow && last_kind != Some(kind) {
            let (new_models, new_mixer, new_lzp_idx) = build_stack(kind);
            models = new_models;
            mixer = new_mixer;
            lzp_idx = new_lzp_idx;
            last_kind = Some(kind);
            match_hist.clear();
        }

        let (comp, method, store_orig_len) = if use_slow {
            encode_block_slow(
                block_data,
                kind,
                &mut models,
                &mut mixer,
                lzp_idx,
                global_dict.as_ref(),
                &mut match_hist,
            )
        } else {
            encode_block_fastish(
                mode,
                block_data,
                kind,
                global_dict.as_ref(),
                &mut match_hist,
            )
        };

        if use_slow && method != METHOD_COPY {
            mixer.decay(BLOCK_DECAY);
        }
        encoded[i] = Some((comp, method, store_orig_len));
    }

    // Phase 3: assemble entries in segment order and record per-member ranges.
    let mut entries: Vec<BlockEntry> = Vec::new();
    let mut payloads: Vec<u8> = Vec::new();
    let mut diags: Vec<BlockDiag> = Vec::new();
    let mut first_block = vec![0u32; bufs.len()];
    let mut num_blocks = vec![0u32; bufs.len()];

    for (i, &(mi, start, end, kind)) in segments.iter().enumerate() {
        let block_data = &bufs[mi][start..end];
        let (comp, method, store_orig_len) = encoded[i].take().expect("block encoded");
        if num_blocks[mi] == 0 {
            first_block[mi] = entries.len() as u32;
        }
        num_blocks[mi] += 1;
        entries.push(BlockEntry {
            comp_len: comp.len() as u32,
            orig_len: store_orig_len as u32,
            method,
            crc32: crc32(block_data),
        });
        diags.push(BlockDiag {
            kind: format!("{:?}", kind),
            method,
            size_in: block_data.len(),
            size_out: comp.len(),
        });
        payloads.extend_from_slice(&comp);
    }

    let ranges: Vec<(u32, u32)> = first_block.into_iter().zip(num_blocks).collect();
    let global_dict_bytes = global_dict.as_ref().map(XwrtDictionary::to_bytes);
    Ok(EncodedArchive {
        global_dict_bytes,
        ranges,
        entries,
        payloads,
        diags,
    })
}

/// Dispatch Fast / General / Wire per-block encode (no Slow bit CM).
fn encode_block_fastish(
    mode: CodecMode,
    block_data: &[u8],
    kind: crate::classify::BlockKind,
    global_dict: Option<&crate::model::word::XwrtDictionary>,
    match_hist: &mut Vec<u8>,
) -> (Vec<u8>, u8, usize) {
    match mode {
        CodecMode::Wire => encode_block_wire(block_data),
        CodecMode::General => encode_block_general(block_data, kind, match_hist),
        _ => encode_block_fast(block_data, kind, global_dict, match_hist),
    }
}

/// Level `-1`: hash-chain LZ. Copy when the wire payload does not shrink.
fn encode_block_wire(block_data: &[u8]) -> (Vec<u8>, u8, usize) {
    if block_data.is_empty() {
        return (Vec::new(), METHOD_COPY, 0);
    }
    let comp = crate::wire::compress(block_data);
    if comp.len() >= block_data.len() {
        (block_data.to_vec(), METHOD_COPY, block_data.len())
    } else {
        (comp, METHOD_WIRE, block_data.len())
    }
}

/// Level `-3`: byte CM + DP-LZP, no BWT / XWRT trials.
fn encode_block_general(
    block_data: &[u8],
    kind: crate::classify::BlockKind,
    match_hist: &mut Vec<u8>,
) -> (Vec<u8>, u8, usize) {
    if kind == crate::classify::BlockKind::Random {
        match_hist.clear();
        return (block_data.to_vec(), METHOD_COPY, block_data.len());
    }
    if kind == crate::classify::BlockKind::Exec {
        let transformed = crate::model::e8e9::e8e9_transform(block_data);
        let comp = compress_byte_with_matches(&transformed, kind, match_hist);
        append_match_hist(match_hist, &transformed);
        return (comp, METHOD_BYTE_EXEC_E8E9, transformed.len());
    }
    if kind == crate::classify::BlockKind::Binary {
        let comp = compress_byte_with_matches(block_data, kind, match_hist);
        append_match_hist(match_hist, block_data);
        (comp, METHOD_BYTE_CM, block_data.len())
    } else {
        // Text: run a cheap BWT/XWRT trial on blocks ≥256KB to close `-3`
        // vs gzip on ratio. Falls back to METHOD_BYTE_TEXT (plain byte CM)
        // when no transform wins.
        match_hist.clear();
        let trial = if block_data.len() >= 256 * 1024 {
            bwt::compress_text_with_trial(block_data, None)
        } else {
            bwt::BwtPathResult {
                pipeline: bwt::BwtPipeline::RawCm,
                encoded_size: block_data.len(),
                is_bwt: false,
                payload: block_data.to_vec(),
            }
        };
        let transformed = trial.payload;
        let store_len = transformed.len();
        let comp = compress_byte_with_matches(&transformed, kind, &[]);
        let method = if trial.is_bwt {
            method_for_pipeline(trial.pipeline, CodecMode::General)
        } else {
            METHOD_BYTE_TEXT
        };
        (comp, method, store_len)
    }
}

/// Fast-mode per-block encode: byte-level CM, or copy for random blocks.
///
/// `match_hist` mirrors the Slow Binary/Exec path (W1 cross-block matches).
fn encode_block_fast(
    block_data: &[u8],
    kind: crate::classify::BlockKind,
    global_dict: Option<&crate::model::word::XwrtDictionary>,
    match_hist: &mut Vec<u8>,
) -> (Vec<u8>, u8, usize) {
    if kind == crate::classify::BlockKind::Random {
        match_hist.clear();
        (block_data.to_vec(), METHOD_COPY, block_data.len())
    } else if kind == crate::classify::BlockKind::Text {
        match_hist.clear();
        // Same BWT trial as the bit path; the chosen pipeline's payload is then
        // byte-coded with DP-LZP literal-skip (R2).
        let trial = bwt::compress_text_with_trial(block_data, global_dict);
        let method = method_for_pipeline(trial.pipeline, CodecMode::Fast);
        let transformed = trial.payload;
        let store_len = transformed.len();
        let comp = compress_byte_with_matches(&transformed, kind, &[]);
        (comp, method, store_len)
    } else if kind == crate::classify::BlockKind::Exec {
        // Exec: apply E8E9 transform to convert x86 relative offsets to absolute,
        // making them much more compressible.
        let transformed = crate::model::e8e9::e8e9_transform(block_data);
        let comp = compress_byte_with_matches(&transformed, kind, match_hist);
        append_match_hist(match_hist, &transformed);
        (comp, METHOD_BYTE_EXEC_E8E9, transformed.len())
    } else {
        // Binary: raw byte CM + DP-LZP.
        let comp = compress_byte_with_matches(block_data, kind, match_hist);
        append_match_hist(match_hist, block_data);
        (comp, METHOD_BYTE_CM, block_data.len())
    }
}

/// Byte CM with DP-LZP match side-stream (same framing as the bit path).
fn compress_byte_with_matches(
    data: &[u8],
    kind: crate::classify::BlockKind,
    hist: &[u8],
) -> Vec<u8> {
    let runs = scan_matches(data, kind, hist);
    let byte_runs: Vec<crate::bytecodec::ByteMatchRun> = runs
        .iter()
        .map(|r| crate::bytecodec::ByteMatchRun {
            pos: r.pos,
            len: r.len,
            dist: r.dist,
        })
        .collect();
    let deep = matches!(
        kind,
        crate::classify::BlockKind::Binary | crate::classify::BlockKind::Exec
    );
    crate::bytecodec::compress_block_with_matches(data, &byte_runs, deep)
}

/// Slow-mode per-block encode: bit-level CM with the classifier-aware stacks.
///
/// `match_hist` is prior same-kind Binary/Exec bytes (empty for Text/Random).
/// Updated in place after Binary/Exec blocks so the next block can match across
/// the boundary (W1).
fn encode_block_slow(
    block_data: &[u8],
    kind: crate::classify::BlockKind,
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    global_dict: Option<&crate::model::word::XwrtDictionary>,
    match_hist: &mut Vec<u8>,
) -> (Vec<u8>, u8, usize) {
    // Random blocks: store verbatim (method COPY). Don't run CM on
    // entropy-poor data — the rANS path would inflate and the decoder
    // treats METHOD_COPY as a passthrough anyway.
    if kind == crate::classify::BlockKind::Random {
        match_hist.clear();
        (block_data.to_vec(), METHOD_COPY, block_data.len())
    } else if kind == crate::classify::BlockKind::Text {
        match_hist.clear();
        // Per-block trial: pick the best BWT pipeline for this Text block.
        let trial = bwt::compress_text_with_trial(block_data, global_dict);
        let method = method_for_pipeline(trial.pipeline, CodecMode::Slow);
        // Winning payload is returned by the trial — do not re-encode.
        let transformed = trial.payload;
        // For BWT paths, `orig_len` stores the *transformed* length (what the
        // decoder must decode from rANS). The original length is recovered
        // during BWT reversal; correctness is verified by CRC.
        let store_len = transformed.len();
        let comp = compress_block(models, mixer, lzp_idx, &transformed, kind, &[]);
        (comp, method, store_len)
    } else if kind == crate::classify::BlockKind::Exec {
        // Exec: apply E8E9 transform to convert x86 relative offsets to absolute,
        // making them much more compressible.
        let transformed = crate::model::e8e9::e8e9_transform(block_data);
        let comp = compress_block(models, mixer, lzp_idx, &transformed, kind, match_hist);
        append_match_hist(match_hist, &transformed);
        (comp, METHOD_EXEC, transformed.len())
    } else {
        // Binary: raw CM with the existing stack.
        let comp = compress_block(models, mixer, lzp_idx, block_data, kind, match_hist);
        append_match_hist(match_hist, block_data);
        (comp, method_for_kind(kind), block_data.len())
    }
}

/// Binary/Exec block size (W1): 1 MiB so DP-LZP matches span farther within a
/// block. Random stays 64 KiB (verbatim copy — no match gain).
const BINARY_BLOCK_SIZE: usize = 1024 * 1024;

/// Cross-block match history retained for Binary/Exec DP-LZP (W1). Caps lookback
/// so dist in the side-stream can reference prior same-kind blocks.
/// Bumped 4 MiB → 8 MiB (T2) to let the DP-LZP find longer-distance matches
/// on binary/`mr` streams; the match window in `scan_matches` moves with it.
const MATCH_HIST_CAP: usize = 8 * 1024 * 1024;

fn append_match_hist(hist: &mut Vec<u8>, data: &[u8]) {
    hist.extend_from_slice(data);
    if hist.len() > MATCH_HIST_CAP {
        let drop = hist.len() - MATCH_HIST_CAP;
        hist.drain(..drop);
    }
}

fn block_size_for_kind(
    kind: crate::classify::BlockKind,
    offset: usize,
    total: usize,
    mode: CodecMode,
) -> usize {
    if mode == CodecMode::Wire {
        return (4 * 1024 * 1024).min(total.saturating_sub(offset)).max(1);
    }
    match kind {
        crate::classify::BlockKind::Text => {
            let max_text = 4 * 1024 * 1024;
            let size = (total - offset).min(max_text);
            size.max(64 * 1024)
        }
        crate::classify::BlockKind::Binary | crate::classify::BlockKind::Exec => BINARY_BLOCK_SIZE,
        crate::classify::BlockKind::Random => 64 * 1024,
    }
}

/// Map a `BlockKind` to the container method byte the decoder uses to pick a stack.
#[must_use]
pub const fn method_for_kind(kind: crate::classify::BlockKind) -> u8 {
    match kind {
        crate::classify::BlockKind::Random => METHOD_COPY,
        crate::classify::BlockKind::Text => METHOD_TEXT,
        crate::classify::BlockKind::Binary => METHOD_BINARY,
        crate::classify::BlockKind::Exec => METHOD_EXEC,
    }
}

/// Choose the model stack for a block, based on the classifier's `BlockKind`.
/// Both encode and decode paths call this, so the stacks are guaranteed to be in sync.
///
/// Returns `(models, mixer, lzp_idx)` where `lzp_idx` is the index of the LZP
/// model in `models` (if present).
#[must_use]
pub fn build_stack_for_kind(
    kind: crate::classify::BlockKind,
) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>) {
    match kind {
        // Random: copy, no models needed (encoder won't call compress_block).
        crate::classify::BlockKind::Random => {
            let models: Vec<crate::model::stack_enum::StackModel> = vec![];
            (models, MixerBank::new(0), None)
        }
        crate::classify::BlockKind::Text => {
            // Text stack: orders 0–2, Sparse, Exec, LZP, PPMd+SSM, Word.
            // IndirectModel was tried with 16k banks (C8); kept in-tree but not
            // in the default stack after prior dickens regressions.
            use crate::model::stack_enum::StackModel;
            let n = 8;
            let models: Vec<StackModel> = vec![
                StackModel::Order(crate::model::order::OrderN::new(0)),
                StackModel::Order(crate::model::order::OrderN::new(1)),
                StackModel::Order(crate::model::order::OrderN::new(2)),
                StackModel::Sparse(crate::model::sparse::Sparse::new()),
                StackModel::Exec(crate::model::exec::Exec::new()),
                StackModel::Lzp(crate::model::lzp::Lzp::new()),
                // C10 Order-12 measured ~0pt on text (2MB dickens −0.003pt); keep default order-8.
                StackModel::Ppmd(crate::model::ppmd_ssm::PpmdSsm::new()),
                StackModel::Word(crate::model::word::WordModel::new()),
            ];
            (models, MixerBank::new(n), Some(5))
        }
        crate::classify::BlockKind::Binary => {
            // Binary stack (best configuration, no SSM).
            // Indirect (W5) A/B'd on headline mr and regressed ~0.1pt → conditional on binary.
            use crate::model::stack_enum::StackModel;
            let n = 8;
            let models: Vec<StackModel> = vec![
                StackModel::Order(crate::model::order::OrderN::new(0)),
                StackModel::Order(crate::model::order::OrderN::new(1)),
                StackModel::Order(crate::model::order::OrderN::new(2)),
                StackModel::Sparse(crate::model::sparse::Sparse::new()),
                StackModel::Exec(crate::model::exec::Exec::new()),
                StackModel::Lzp(crate::model::lzp::Lzp::new()),
                StackModel::Ppm(crate::model::ppm::PpmModel::new(3)),
                StackModel::Indirect(crate::model::indirect_dmc::IndirectModel::default()),
            ];
            (models, MixerBank::new(n), Some(5))
        }
        crate::classify::BlockKind::Exec => {
            // Exec stack (best configuration, no SSM).
            // with orders 0-2, Sparse, LZP, PPM order-3; no Exec model
            // Conditional Indirect (W5) for binary/exec mode
            use crate::model::stack_enum::StackModel;
            let n = 7;
            let models: Vec<StackModel> = vec![
                StackModel::Order(crate::model::order::OrderN::new(0)),
                StackModel::Order(crate::model::order::OrderN::new(1)),
                StackModel::Order(crate::model::order::OrderN::new(2)),
                StackModel::Sparse(crate::model::sparse::Sparse::new()),
                StackModel::Lzp(crate::model::lzp::Lzp::new()),
                StackModel::Ppm(crate::model::ppm::PpmModel::new(3)),
                StackModel::Indirect(crate::model::indirect_dmc::IndirectModel::default()),
            ];
            (models, MixerBank::new(n), Some(4))
        }
    }
}

/// Legacy alias kept for benchmark tooling (`src/stacks.rs`).
#[must_use]
pub fn build_full_stack() -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>) {
    build_stack_for_kind(crate::classify::BlockKind::Binary)
}

/// Compress one block.
///
/// Runs DP-optimal LZP match pre-pass, emits (len, dist, pos)
/// side-stream records, then rANS-encodes only **literal** (non-matched) bytes.
/// Matched bytes are reconstructed by the decoder from the side-stream.
fn compress_block(
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    block: &[u8],
    kind: crate::classify::BlockKind,
    match_hist: &[u8],
) -> Vec<u8> {
    let runs = scan_matches(block, kind, match_hist);
    encode_block_with_matches(models, mixer, lzp_idx, block, &runs)
}

/// Plain CM encoding (no match side-stream).
fn encode_block_plain(
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    block: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cascade = SseApmCascade::new();

    // Pre-build per-block dictionaries.
    for m in models.iter_mut() {
        m.prepare_block(block);
    }

    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;

    let mut prev_byte = 0u8;
    for &byte in block {
        cascade.set_context(prev_byte); // Use PREVIOUS byte as context (decoder also sees prev)
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1u8 == 1u8;
            let bit_pos = bit_idx as u8;
            for (j, m) in models.iter().enumerate() {
                probs[j] = m.predict();
            }
            let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);

            mixer.mix_and_update(
                &probs[..n],
                bit,
                bit_pos,
                lzp_conf,
                &mut |encoded_bit, p_mixer| {
                    let p_refined = cascade.refine(p_mixer, bit_pos);
                    enc.encode_bit(encoded_bit, p_refined);
                    cascade.update(encoded_bit, p_mixer, bit_pos);
                },
            );
            for m in models.iter_mut() {
                m.update(bit);
            }
        }
        // Feed completed byte to the mixer bank's byte assembler for context.
        mixer.push_byte(byte);
        prev_byte = byte; // Update for next iteration
    }
    out.extend(enc.finish());
    out
}

fn encode_block_with_matches(
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    block: &[u8],
    runs: &[MatchRun],
) -> Vec<u8> {
    let mut out = Vec::new();

    for m in models.iter_mut() {
        m.prepare_block(block);
    }

    // Match side-stream: [num_runs:u32][flag:u8][payload_len:u32][payload...]
    // where payload is varint records (delta_pos, len, dist), optionally
    // order-0 FSE/rANS-compressed (flag=1) when that shrinks the blob.
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
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    out.push(flag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);

    // Build a position→length map for matched regions.
    let mut match_len_at: Vec<usize> = vec![0; block.len()];
    for r in runs {
        if r.pos < block.len() && match_len_at[r.pos] < r.len {
            match_len_at[r.pos] = r.len;
        }
    }

    // rANS-encode only literal bytes (skip matched regions).
    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;
    let mut enc_bit = |bit: bool, p: u16| enc.encode_bit(bit, p);

    // Helper closure: feed a byte through models+mixer WITHOUT rANS encoding.
    // Used for matched bytes — they must update context state but not produce
    // rANS bits (the decoder reconstructs them from match records).
    macro_rules! skip_byte {
        ($byte:expr) => {{
            let byte = $byte;
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1u8 == 1u8;
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
                mixer.update(&probs[..n], bit, bit_pos, lzp_conf);
                for m in models.iter_mut() {
                    m.update(bit);
                }
            }
            mixer.push_byte(byte);
        }};
    }

    let mut i = 0usize;
    while i < block.len() {
        if match_len_at[i] > 0 {
            // Matched region: feed bytes for context, skip rANS encoding.
            let len = match_len_at[i];
            for j in 0..len {
                skip_byte!(block[i + j]);
            }
            i += len;
        } else {
            // Literal: encode through rANS.
            let byte = block[i];
            let mut cascade = SseApmCascade::new();
            cascade.set_context(byte); // Set byte context for SSE/APM cascade
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1u8 == 1u8;
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);

                mixer.mix_and_update(
                    &probs[..n],
                    bit,
                    bit_pos,
                    lzp_conf,
                    &mut |encoded_bit, p_mixer| {
                        let p_refined = cascade.refine(p_mixer, bit_pos);
                        enc.encode_bit(encoded_bit, p_refined);
                        cascade.update(encoded_bit, p_mixer, bit_pos);
                    },
                );
                for m in models.iter_mut() {
                    m.update(bit);
                }
            }
            mixer.push_byte(byte);
            i += 1;
        }
    }
    out.extend(enc.finish());
    out
}

fn scan_matches(block: &[u8], kind: crate::classify::BlockKind, hist: &[u8]) -> Vec<MatchRun> {
    let mut lzp = Lzp::new();
    let base = hist.len();
    let n = block.len();
    let min_len = match_min_len(kind);

    // W6: kind-specific residual cost — Binary/Exec CM is typically weaker than
    // Text-after-BWT, so matches are worth taking slightly more aggressively.
    let avg_bits_per_byte: f64 = match kind {
        crate::classify::BlockKind::Binary | crate::classify::BlockKind::Exec => 5.0,
        _ => 4.0,
    };
    // Match record overhead: 1 (flag) + 8 (len) + 24 (dist) = 33 bits.
    const MATCH_OVERHEAD_BITS: f64 = 33.0;
    let window = 8 * 1024 * 1024;

    // Combined stream: prior same-kind history || current block (W1). Match
    // positions are reported relative to the block start; distances may reach
    // into `hist` (decoder keeps the same prefix).
    let mut combined = Vec::with_capacity(base + n);
    combined.extend_from_slice(hist);
    combined.extend_from_slice(block);

    // Phase 1: pre-compute the best match at every position in the block.
    let mut best_match: Vec<Option<(usize, usize)>> = vec![None; n];
    for i in 0..base + n {
        lzp.train_at(&combined, i);
        if i < base {
            continue;
        }
        let bi = i - base;
        if bi + 1 >= min_len && bi + min_len <= n {
            if let Some((len, dist)) = lzp.best_match(&combined, i) {
                let len = len.min(255).min(n - bi);
                if len >= min_len && dist > 0 && dist <= window {
                    best_match[bi] = Some((len, dist));
                }
            }
        }
    }

    // Phase 2: DP optimal parse over the current block only.
    let mut dp: Vec<f64> = vec![f64::INFINITY; n + 1];
    let mut choice: Vec<bool> = vec![false; n];
    dp[n] = 0.0;

    for i in (0..n).rev() {
        let literal_cost = avg_bits_per_byte + dp[i + 1];
        dp[i] = literal_cost;
        choice[i] = false;

        if let Some((len, _dist)) = best_match[i] {
            let match_cost = MATCH_OVERHEAD_BITS + dp[i + len];
            if match_cost < dp[i] {
                dp[i] = match_cost;
                choice[i] = true;
            }
        }
    }

    // Phase 3: backtrack to extract match runs (block-relative positions).
    let mut runs: Vec<MatchRun> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if choice[i] {
            if let Some((len, dist)) = best_match[i] {
                runs.push(MatchRun { pos: i, len, dist });
                i += len;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    runs
}

fn find_match_distance(data: &[u8], pos: usize, len: usize) -> usize {
    if pos < len || len == 0 {
        return 0;
    }
    let needle = &data[pos - len..pos];
    let window = 8 * 1024 * 1024;
    let start = pos.saturating_sub(window);
    for back in (start..pos - len + 1).rev() {
        if data[back..back + len] == *needle {
            return pos - back;
        }
    }
    0
}

/// Resolve a match byte using prior same-kind history (W1) plus bytes already
/// decoded in this block. `dist` is the LZ lookback from the current end of
/// `out` (same convention as in-block-only matches).
#[inline]
fn match_byte_from_hist(hist: &[u8], out: &[u8], dist: usize) -> u8 {
    if dist == 0 {
        return 0;
    }
    if dist <= out.len() {
        out[out.len() - dist]
    } else {
        let into_hist = dist - out.len();
        if into_hist <= hist.len() {
            hist[hist.len() - into_hist]
        } else {
            0
        }
    }
}

/// Decode a block.
///
/// Reads match side-stream (validates records), then rANS-decodes all bytes.
fn decode_block(
    comp: &[u8],
    orig_len: usize,
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    match_hist: &[u8],
) -> Result<Vec<u8>> {
    decode_block_with_matches(comp, orig_len, models, mixer, lzp_idx, match_hist)
}

fn decode_block_plain(
    comp: &[u8],
    orig_len: usize,
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
) -> Result<Vec<u8>> {
    let mut dec = BitDecoder::new(comp).map_err(|e| RcnError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;
    let mut cascade = SseApmCascade::new();

    while out.len() < orig_len {
        let mut byte = 0u8;
        for bit_idx in (0..8).rev() {
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
            let (p_mixer, pacc) = mixer.mix_acc(&probs[..n], bit_pos, lzp_conf);
            let p_refined = cascade.refine(p_mixer, bit_pos);
            let bit = dec
                .decode_bit(p_refined)
                .map_err(|e| RcnError::Entropy(e.to_string()))?;
            mixer.update_acc(&probs[..n], bit, bit_pos, pacc);
            cascade.update(bit, p_mixer, bit_pos);
            for m in models.iter_mut() {
                m.update(bit);
            }
            if bit {
                byte |= 1 << bit_idx;
            }
        }
        out.push(byte);
        cascade.set_context(byte);
        mixer.push_byte(byte);
    }
    Ok(out)
}

fn decode_block_with_matches(
    comp: &[u8],
    orig_len: usize,
    models: &mut [crate::model::stack_enum::StackModel],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    match_hist: &[u8],
) -> Result<Vec<u8>> {
    // Header: [num_runs:u32][flag:u8][payload_len:u32] = 9 bytes minimum.
    if comp.len() < 9 {
        return Err(RcnError::InvalidContainer(
            "match side-stream too short".into(),
        ));
    }
    let num_runs = u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]) as usize;
    let flag = comp[4];
    let payload_len = u32::from_le_bytes([comp[5], comp[6], comp[7], comp[8]]) as usize;
    let payload_start: usize = 9;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| RcnError::InvalidContainer("match side-stream length overflow".into()))?;
    if payload_end > comp.len() {
        return Err(RcnError::InvalidContainer(
            "match side-stream truncated".into(),
        ));
    }
    let raw_side = side_fse::unpack_side_stream(flag, &comp[payload_start..payload_end])
        .map_err(|()| RcnError::InvalidContainer("corrupt match side-stream entropy".into()))?;

    // Read all match records (varint-encoded delta positions).
    let mut runs: Vec<MatchRun> = Vec::with_capacity(num_runs);
    let mut prev_pos: usize = 0;
    let mut offset = 0usize;
    for _ in 0..num_runs {
        let (delta_pos, new_offset) = read_varint(&raw_side, offset).ok_or_else(|| {
            RcnError::InvalidContainer("truncated match record (delta_pos)".into())
        })?;
        offset = new_offset;
        let (len, new_offset) = read_varint(&raw_side, offset)
            .ok_or_else(|| RcnError::InvalidContainer("truncated match record (len)".into()))?;
        offset = new_offset;
        let (dist, new_offset) = read_varint(&raw_side, offset)
            .ok_or_else(|| RcnError::InvalidContainer("truncated match record (dist)".into()))?;
        offset = new_offset;
        let pos = prev_pos.wrapping_add(delta_pos);
        prev_pos = pos;
        runs.push(MatchRun { pos, len, dist });
    }

    let mut dec =
        BitDecoder::new(&comp[payload_end..]).map_err(|e| RcnError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;

    // Walk through the original block, interleaving matched and literal bytes.
    let mut match_idx = 0usize;
    let mut i = 0usize; // current position in the output (decoded) stream

    // Helper closure: feed a byte through models+mixer WITHOUT rANS decoding.
    macro_rules! skip_byte {
        ($byte:expr) => {{
            let byte = $byte;
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1u8 == 1u8;
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
                mixer.update(&probs[..n], bit, bit_pos, lzp_conf);
                for m in models.iter_mut() {
                    m.update(bit);
                }
            }
            mixer.push_byte(byte);
        }};
    }

    while i < orig_len {
        // Check if a match starts at the current position.
        if match_idx < runs.len() && runs[match_idx].pos == i {
            let run = runs[match_idx];
            // Copy `len` bytes from history: out.len() - dist .. out.len() - dist + len
            // But we need to be careful about overlapping copies.
            let dist = run.dist;
            for _j in 0..run.len {
                let byte = match_byte_from_hist(match_hist, &out, dist);
                skip_byte!(byte);
                out.push(byte);
            }
            i += run.len;
            match_idx += 1;
        } else {
            // Literal: rANS-decode a byte.
            let mut byte = 0u8;
            let mut cascade = SseApmCascade::new();
            for bit_idx in (0..8).rev() {
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
                let (p_mixer, pacc) = mixer.mix_acc(&probs[..n], bit_pos, lzp_conf);
                let p_refined = cascade.refine(p_mixer, bit_pos);
                let bit = dec
                    .decode_bit(p_refined)
                    .map_err(|e| RcnError::Entropy(e.to_string()))?;
                mixer.update_acc(&probs[..n], bit, bit_pos, pacc);
                cascade.update(bit, p_mixer, bit_pos);
                for m in models.iter_mut() {
                    m.update(bit);
                }
                if bit {
                    byte |= 1 << bit_idx;
                }
            }
            out.push(byte);
            cascade.set_context(byte);
            mixer.push_byte(byte);
            i += 1;
        }
    }
    Ok(out)
}

/// Decompress a `RCN1` container back to the original bytes.
///
/// v1 streams and v2 archives with exactly one member succeed. A v2 archive
/// with two or more members returns [`RcnError::InvalidContainer`]; use
/// [`decompress_archive`] instead.
///
/// # Errors
///
/// Returns [`RcnError`] on a malformed container, corrupt block, CRC mismatch,
/// or a multi-member v2 archive.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    let members = decompress_impl(data, &mut build_stack_for_kind)?;
    if members.len() > 1 {
        return Err(RcnError::InvalidContainer(
            "multi-member archive; use decompress_archive".into(),
        ));
    }
    Ok(members.into_iter().next().map(|(_, b)| b).unwrap_or_default())
}

/// Decompress a v1 stream or v2 archive into named members.
///
/// v1 containers have no stored name; the single member's name is empty.
///
/// # Errors
///
/// Returns [`RcnError`] on a malformed container, corrupt block, or CRC mismatch.
pub fn decompress_archive(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    decompress_impl(data, &mut build_stack_for_kind)
}

/// Decompress a `RCN1` container using a custom model-stack builder.
///
/// This mirrors [`compress_with`] on the decode side: the `build_stack` closure
/// must match the one used for compression, otherwise rANS decoding will produce
/// garbage and the CRC check will fail.
pub fn decompress_with<F>(data: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>),
{
    let members = decompress_impl(data, build_stack)?;
    if members.len() > 1 {
        return Err(RcnError::InvalidContainer(
            "multi-member archive; use decompress_archive".into(),
        ));
    }
    Ok(members.into_iter().next().map(|(_, b)| b).unwrap_or_default())
}

fn member_index_for_block(members: &[Member], bi: usize) -> Option<usize> {
    members.iter().position(|m| {
        let start = m.first_block as usize;
        let end = start.saturating_add(m.num_blocks as usize);
        bi >= start && bi < end
    })
}

fn split_archive_output(
    out: &[u8],
    entries: &[BlockEntry],
    members: &[Member],
) -> Result<Vec<(String, Vec<u8>)>> {
    let mut ends = Vec::with_capacity(entries.len());
    let mut pos = 0usize;
    for e in entries {
        pos = pos.saturating_add(e.orig_len as usize);
        ends.push(pos);
    }
    let mut result = Vec::with_capacity(members.len());
    for m in members {
        let start = m.first_block as usize;
        let n = m.num_blocks as usize;
        if start.saturating_add(n) > entries.len() {
            return Err(RcnError::InvalidContainer(format!(
                "member '{}' block range {}+{} overruns {} entries",
                m.name,
                m.first_block,
                m.num_blocks,
                entries.len()
            )));
        }
        if n == 0 {
            result.push((m.name.clone(), Vec::new()));
            continue;
        }
        let start_off = if start == 0 { 0 } else { ends[start - 1] };
        let end_off = ends[start + n - 1];
        result.push((m.name.clone(), out[start_off..end_off].to_vec()));
    }
    Ok(result)
}

fn decompress_impl<F>(data: &[u8], build_stack: &mut F) -> Result<Vec<(String, Vec<u8>)>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<crate::model::stack_enum::StackModel>, MixerBank, Option<usize>),
{
    use std::io::Cursor;
    let mut cur = Cursor::new(data);
    let header = Header::read(&mut cur).map_err(|e| RcnError::InvalidContainer(e.to_string()))?;
    if header.version != VERSION && header.version != VERSION_V2 {
        return Err(RcnError::InvalidContainer(format!(
            "unsupported version {}",
            header.version
        )));
    }
    // Consume the global XWRT dictionary (if present) from between the header
    // and the member table / BlockEntry table; every XWRT block refers back to it.
    let (global_dict_bytes, dict_end) =
        read_global_dict(data, cur.position() as usize, header.flags)?;
    let (members_meta, table_end) = if header.version == VERSION_V2 {
        read_members(data, dict_end)?
    } else {
        (Vec::new(), dict_end)
    };
    cur.set_position(table_end as u64);
    let global_dict = XwrtDictionary::from_bytes(&global_dict_bytes);

    // Bound entry-table size against remaining bytes (13 bytes each).
    let entry_start = cur.position() as usize;
    let entry_bytes = (header.num_blocks as usize).saturating_mul(13);
    if entry_start.saturating_add(entry_bytes) > data.len() {
        return Err(RcnError::InvalidContainer(format!(
            "num_blocks {} requires {} entry bytes past offset {}, only {} available",
            header.num_blocks,
            entry_bytes,
            entry_start,
            data.len().saturating_sub(entry_start)
        )));
    }

    let mut entries = Vec::with_capacity(header.num_blocks as usize);
    for _ in 0..header.num_blocks {
        entries.push(
            BlockEntry::read(&mut cur).map_err(|e| RcnError::InvalidContainer(e.to_string()))?,
        );
    }
    let payload_start = cur.position() as usize;
    let payloads = &data[payload_start..];

    for m in &members_meta {
        let start = m.first_block as usize;
        let n = m.num_blocks as usize;
        if start.saturating_add(n) > entries.len() {
            return Err(RcnError::InvalidContainer(format!(
                "member '{}' block range {}+{} overruns {} entries",
                m.name,
                m.first_block,
                m.num_blocks,
                entries.len()
            )));
        }
    }

    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut last_kind: Option<crate::classify::BlockKind> = None;
    let mut last_member: Option<usize> = None;
    let mut models: Vec<crate::model::stack_enum::StackModel> = Vec::new();
    let mut mixer = MixerBank::new(0);
    let mut lzp_idx: Option<usize> = None;
    let mut match_hist: Vec<u8> = Vec::new();
    let mut hist_kind: Option<crate::classify::BlockKind> = None;
    for (bi, entry) in entries.iter().enumerate() {
        if header.version == VERSION_V2 {
            let cur_m = member_index_for_block(&members_meta, bi);
            if cur_m != last_member {
                last_member = cur_m;
                last_kind = None;
                match_hist.clear();
                hist_kind = None;
            }
        }
        let comp_len = entry.comp_len as usize;
        if pos.saturating_add(comp_len) > payloads.len() {
            return Err(RcnError::TruncatedStream(bi));
        }
        let comp = &payloads[pos..pos + comp_len];
        pos += comp_len;

        let block = if entry.method == METHOD_COPY {
            match_hist.clear();
            hist_kind = None;
            comp.to_vec()
        } else if entry.method == METHOD_WIRE {
            match_hist.clear();
            hist_kind = None;
            crate::wire::decompress(comp, entry.orig_len as usize)?
        } else if matches!(
            entry.method,
            METHOD_BYTE_CM
                | METHOD_BYTE_TEXT
                | METHOD_BYTE_BWT_MTF_RLE
                | METHOD_BYTE_LZP_BWT_MTF
                | METHOD_BYTE_JSON_SPLIT
                | METHOD_BYTE_XWRT_BWT_MTF_RLE
                | METHOD_BYTE_CSV_SPLIT
                | METHOD_BYTE_XML_SPLIT
                | METHOD_BYTE_EXEC_E8E9
        ) {
            let use_hist = matches!(
                entry.method,
                METHOD_BYTE_CM | METHOD_BYTE_EXEC_E8E9
            );
            let deep = use_hist; // W2: deep models on Binary/Exec Fast only
            let hist = if use_hist { match_hist.as_slice() } else { &[] };
            let decoded = crate::bytecodec::decompress_block_with_matches(
                comp,
                entry.orig_len as usize,
                hist,
                deep,
            )
            .map_err(|e| match e {
                RcnError::CorruptBlock(s) => RcnError::CorruptBlock(s),
                other => other,
            })?;
            if entry.method == METHOD_BYTE_CM {
                if hist_kind != Some(crate::classify::BlockKind::Binary) {
                    match_hist.clear();
                    hist_kind = Some(crate::classify::BlockKind::Binary);
                }
                append_match_hist(&mut match_hist, &decoded);
            } else if entry.method == METHOD_BYTE_EXEC_E8E9 {
                if hist_kind != Some(crate::classify::BlockKind::Exec) {
                    match_hist.clear();
                    hist_kind = Some(crate::classify::BlockKind::Exec);
                }
                append_match_hist(&mut match_hist, &decoded);
            } else {
                match_hist.clear();
                hist_kind = None;
            }
            inverse_transform_for_method(
                entry.method,
                decoded,
                entry.orig_len as usize,
                global_dict.as_ref(),
            )?
        } else {
            let kind = kind_for_method(entry.method)?;
            if last_kind != Some(kind) {
                let (new_models, new_mixer, new_lzp_idx) = build_stack(kind);
                models = new_models;
                mixer = new_mixer;
                lzp_idx = new_lzp_idx;
                last_kind = Some(kind);
                match_hist.clear();
                hist_kind = Some(kind);
            }
            let hist = if matches!(
                kind,
                crate::classify::BlockKind::Binary | crate::classify::BlockKind::Exec
            ) {
                match_hist.as_slice()
            } else {
                &[]
            };
            let decoded = decode_block(
                comp,
                entry.orig_len as usize,
                &mut models,
                &mut mixer,
                lzp_idx,
                hist,
            )
            .map_err(|e| match e {
                RcnError::Entropy(s) => RcnError::CorruptBlock(s),
                other => other,
            })?;
            let restored = inverse_transform_for_method(
                entry.method,
                decoded.clone(),
                entry.orig_len as usize,
                global_dict.as_ref(),
            )?;
            if matches!(
                kind,
                crate::classify::BlockKind::Binary | crate::classify::BlockKind::Exec
            ) {
                // Entropy domain: decoded (pre-inverse). For Exec that's e8e9 space.
                append_match_hist(&mut match_hist, &decoded);
            } else {
                match_hist.clear();
                hist_kind = None;
            }
            restored
        };

        if crate::container::crc32(&block) != entry.crc32 {
            return Err(RcnError::CrcMismatch(
                bi,
                crate::container::crc32(&block),
                entry.crc32,
            ));
        }
        out.extend_from_slice(&block);

        // Decay at block boundaries (mirrors compressor).
        if entry.method != METHOD_COPY {
            mixer.decay(BLOCK_DECAY);
        }
    }
    if header.version == VERSION {
        return Ok(vec![("".into(), out)]);
    }
    split_archive_output(&out, &entries, &members_meta)
}

/// Reverse map: container method byte → `BlockKind`. Unknown methods error.
fn kind_for_method(method: u8) -> Result<crate::classify::BlockKind> {
    match method {
        METHOD_COPY => Ok(crate::classify::BlockKind::Random),
        METHOD_CM | METHOD_TEXT => Ok(crate::classify::BlockKind::Text),
        METHOD_BINARY => Ok(crate::classify::BlockKind::Binary),
        METHOD_EXEC => Ok(crate::classify::BlockKind::Exec),
        METHOD_BWT_MTF_RLE
        | METHOD_LZP_BWT_MTF
        | METHOD_JSON_SPLIT
        | METHOD_XWRT_BWT_MTF_RLE
        | METHOD_CSV_SPLIT
        | METHOD_XML_SPLIT => Ok(crate::classify::BlockKind::Text),
        METHOD_BYTE_CM
        | METHOD_BYTE_TEXT
        | METHOD_BYTE_BWT_MTF_RLE
        | METHOD_BYTE_LZP_BWT_MTF
        | METHOD_BYTE_JSON_SPLIT
        | METHOD_BYTE_XWRT_BWT_MTF_RLE
        | METHOD_BYTE_CSV_SPLIT
        | METHOD_BYTE_XML_SPLIT => Ok(crate::classify::BlockKind::Text),
        METHOD_BYTE_EXEC_E8E9 => Ok(crate::classify::BlockKind::Exec),
        METHOD_WIRE => Ok(crate::classify::BlockKind::Binary),
        _ => Err(RcnError::InvalidContainer(format!(
            "unknown method {}",
            method
        ))),
    }
}

/// Re-export so callers can build CRCs without reaching into the container module.
pub use crate::container::crc32;

#[cfg(test)]
mod tests {
    use super::*;

    /// A ~200 KB mixed fixture: text + JSON + a binary blob + an ELF-like byte pattern.
    fn mixed_fixture() -> Vec<u8> {
        let mut v = Vec::new();
        let text = b"the quick brown fox jumps over the lazy dog. \\\
            compression mixes many context models so that each bit is predicted well. ";
        for _ in 0..2000 {
            v.extend_from_slice(text);
        }
        let json = b"{\"name\":\"rcn\",\"level\":3,\"models\":[\"order0\",\"order1\",\"order2\",\"sparse\",\"exec\",\"lzp\"],\"ratio\":0.42}\n";
        for _ in 0..500 {
            v.extend_from_slice(json);
        }
        let mut x = 0x9E37_79B9u32;
        for _ in 0..40_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            v.push(x as u8);
            v.push((x >> 11) as u8);
        }
        for _ in 0..3000 {
            v.extend_from_slice(&[0x7f, b'E', b'L', b'F', 0x55, 0x89, 0xE5, 0xFF, 0xD0]);
        }
        v
    }

    #[test]
    fn compress_then_decompress_returns_original() {
        let original = mixed_fixture();
        assert!(original.len() > 200_000, "fixture should exceed 200 KB");
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "round-trip mismatch");
    }

    #[test]
    fn compressed_is_smaller_on_redundant_input() {
        let mut input = Vec::new();
        for _ in 0..50_000 {
            input.extend_from_slice(b"rcnrcnrcn");
        }
        let comp = compress(&input).expect("compress");
        assert!(
            comp.len() < input.len(),
            "redundant input should compress (got {} vs {})",
            comp.len(),
            input.len()
        );
    }

    #[test]
    fn empty_input_round_trips() {
        let comp = compress(&[]).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert!(back.is_empty());
    }

    #[test]
    fn random_block_is_stored_verbatim() {
        let mut buf = [0u8; 4096];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let kind = crate::classify::classify(&buf);
        assert_eq!(kind, crate::classify::BlockKind::Random);
        assert_eq!(method_for_kind(kind), METHOD_COPY);
    }

    #[test]
    fn text_block_uses_text_stack() {
        let text = b"the quick brown fox jumps over the lazy dog. the quick brown fox. ";
        let kind = crate::classify::classify(text);
        assert_eq!(kind, crate::classify::BlockKind::Text);
        assert_eq!(method_for_kind(kind), METHOD_TEXT);
    }

    #[test]
    fn binary_block_uses_binary_stack() {
        let mut buf = [0u8; 256];
        let mut x = 0x9E37_79B9u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let kind = crate::classify::classify(&buf);
        assert_eq!(kind, crate::classify::BlockKind::Binary);
        assert_eq!(method_for_kind(kind), METHOD_BINARY);
    }

    #[test]
    fn json_round_trips() {
        let json = b"{\"name\":\"rcn\",\"level\":3,\"models\":[\"order0\",\"order1\",\"order2\",\"sparse\",\"exec\",\"lzp\"],\"ratio\":0.42}\n";
        let original: Vec<u8> = std::iter::repeat(json.as_ref())
            .take(4000)
            .flatten()
            .copied()
            .collect();
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "JSON round-trip mismatch");
    }

    #[test]
    fn scan_matches_finds_repeats() {
        let data = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
        let runs = scan_matches(data, crate::classify::BlockKind::Text, &[]);
        assert!(
            !runs.is_empty(),
            "expected at least one match in repeated data"
        );
        assert!(runs[0].len >= match_min_len(crate::classify::BlockKind::Text));
    }

    #[test]
    fn scan_matches_empty_on_unique() {
        let mut data = vec![0u8; 256];
        let mut x = 0x1234_5678u32;
        for b in &mut data {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let runs = scan_matches(&data, crate::classify::BlockKind::Binary, &[]);
        assert!(runs.is_empty(), "expected no matches in random data");
    }

    #[test]
    fn find_match_distance_correct() {
        let data = b"abcabcabcabc";
        let d = find_match_distance(data, 6, 3);
        assert_eq!(d, 3, "expected distance 3, got {}", d);
    }

    #[test]
    fn compress_fast_then_decompress_returns_original() {
        let original = mixed_fixture();
        let comp = compress_fast(&original).expect("compress_fast");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "fast round-trip mismatch");
    }

    #[test]
    fn compress_hybrid_then_decompress_returns_original() {
        let original = mixed_fixture();
        let comp = compress_hybrid(&original).expect("compress_hybrid");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "hybrid round-trip mismatch");
    }

    #[test]
    fn compress_mode_fast_empty_input_round_trips() {
        let comp = compress_mode(&[], CodecMode::Fast).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert!(back.is_empty());
    }

    #[test]
    fn compress_mode_fast_json_round_trips() {
        let json = b"{\"name\":\"rcn\",\"level\":3,\"models\":[\"order0\",\"order1\"],\"ratio\":0.42}\n";
        let original: Vec<u8> = std::iter::repeat(json.as_ref())
            .take(4000)
            .flatten()
            .copied()
            .collect();
        let comp = compress_fast(&original).expect("compress_fast");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original);
    }

    #[test]
    fn compress_csv_round_trips() {
        let csv = b"name,age,city\nJohn,30,NYC\nAnna,28,LA\nBob,45,CHI\n";
        let original = csv.repeat(6000);
        assert!(original.len() >= 256 * 1024);
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original);
    }

    #[test]
    fn compress_mode_fast_csv_round_trips() {
        let csv = b"name,age,city\nJohn,30,NYC\nAnna,28,LA\nBob,45,CHI\n";
        let original = csv.repeat(6000);
        let comp = compress_fast(&original).expect("compress_fast");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original);
    }

    #[test]
    fn compress_xml_round_trips() {
        let xml = b"<catalog><book id=\"1\">Alpha</book><book id=\"2\">Beta</book></catalog>\n";
        let original = xml.repeat(5000);
        assert!(original.len() >= 256 * 1024);
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original);
    }

    #[test]
    fn compress_mode_fast_xml_round_trips() {
        let xml = b"<catalog><book id=\"1\">Alpha</book><book id=\"2\">Beta</book></catalog>\n";
        let original = xml.repeat(5000);
        let comp = compress_fast(&original).expect("compress_fast");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original);
    }

    #[test]
    fn method_label_covers_method_byte_exec_e8e9() {
        assert_ne!(method_label(METHOD_BYTE_EXEC_E8E9), "?");
        assert_eq!(method_label(METHOD_WIRE), "wire-LZ");
    }

    #[test]
    fn compress_wire_then_decompress_returns_original() {
        let original = mixed_fixture();
        let comp = compress_wire(&original).expect("compress_wire");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "wire round-trip mismatch");
    }

    #[test]
    fn compress_general_then_decompress_returns_original() {
        let original = mixed_fixture();
        let comp = compress_general(&original).expect("compress_general");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "general round-trip mismatch");
    }

    #[test]
    fn decompress_byte_text_two_blocks_does_not_reuse_hist() {
        // METHOD_BYTE_TEXT must not inherit Binary match history across blocks.
        let a = b"alpha alpha alpha alpha alpha alpha".repeat(40);
        let b = b"bravo bravo bravo bravo bravo bravo".repeat(40);
        let ca = crate::bytecodec::compress_block_with_matches(&a, &[], false);
        let cb = crate::bytecodec::compress_block_with_matches(&b, &[], false);
        let mut buf = Vec::new();
        Header {
            version: VERSION,
            flags: 0,
            block_size_log: 16,
            num_blocks: 2,
        }
        .write(&mut buf);
        BlockEntry {
            comp_len: ca.len() as u32,
            orig_len: a.len() as u32,
            method: METHOD_BYTE_TEXT,
            crc32: crc32(&a),
        }
        .write(&mut buf);
        BlockEntry {
            comp_len: cb.len() as u32,
            orig_len: b.len() as u32,
            method: METHOD_BYTE_TEXT,
            crc32: crc32(&b),
        }
        .write(&mut buf);
        buf.extend_from_slice(&ca);
        buf.extend_from_slice(&cb);
        let back = decompress(&buf).expect("decompress");
        let mut expect = a;
        expect.extend_from_slice(&b);
        assert_eq!(back, expect);
    }

    #[test]
    fn compress_level_archive_matches_hybrid() {
        let original = b"level archive should match hybrid default".repeat(200);
        let a = compress_level(&original, crate::level::Level::Archive).expect("archive");
        let b = compress_hybrid(&original).expect("hybrid");
        assert_eq!(a, b);
    }

    #[test]
    fn decompress_unknown_method_errors() {
        let mut buf = Vec::new();
        Header {
            version: VERSION,
            flags: 0,
            block_size_log: 16,
            num_blocks: 1,
        }
        .write(&mut buf);
        BlockEntry {
            comp_len: 3,
            orig_len: 3,
            method: 99,
            crc32: 0,
        }
        .write(&mut buf);
        buf.extend_from_slice(b"abc");
        let err = decompress(&buf).unwrap_err();
        assert!(matches!(err, RcnError::InvalidContainer(_)));
    }

    #[test]
    fn decompress_crc_mismatch_errors() {
        let original = b"hello world hello world hello world".repeat(100);
        let mut comp = compress_fast(&original).expect("compress_fast");
        // Locate first BlockEntry CRC (after magic+header+optional global dict).
        let mut cur = std::io::Cursor::new(comp.as_slice());
        let header = crate::container::Header::read(&mut cur).expect("header");
        let (_, dict_end) = crate::container::read_global_dict(
            &comp,
            cur.position() as usize,
            header.flags,
        )
        .expect("dict");
        // BlockEntry: comp_len(4)+orig_len(4)+method(1)+crc32(4); CRC starts at +9.
        let crc_off = dict_end + 9;
        comp[crc_off] ^= 0xFF;
        let err = decompress(&comp).unwrap_err();
        assert!(matches!(err, RcnError::CrcMismatch(_, _, _)));
    }

    #[test]
    fn decompress_rejects_comp_len_past_end_without_panic() {
        let mut buf = Vec::new();
        Header {
            version: VERSION,
            flags: 0,
            block_size_log: 16,
            num_blocks: 1,
        }
        .write(&mut buf);
        BlockEntry {
            comp_len: 1_000_000,
            orig_len: 4,
            method: METHOD_COPY,
            crc32: 0,
        }
        .write(&mut buf);
        buf.extend_from_slice(b"tiny");
        let err = decompress(&buf).unwrap_err();
        assert!(matches!(
            err,
            RcnError::TruncatedStream(_) | RcnError::InvalidContainer(_)
        ));
    }

    #[test]
    fn compress_archive_single_file_round_trips_name_and_bytes() {
        let original = b"named single file payload".repeat(80);
        let comp = compress_archive(&[("input.bin", original.as_slice())], CodecMode::Hybrid)
            .expect("compress_archive");
        let members = decompress_archive(&comp).expect("decompress_archive");
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "input.bin");
        assert_eq!(members[0].1, original);
        let via_decompress = decompress(&comp).expect("decompress single member");
        assert_eq!(via_decompress, original);
    }

    #[test]
    fn compress_archive_two_files_round_trips_and_decompress_errors() {
        let a = b"alpha file contents alpha file contents".repeat(40);
        let b = b"bravo file contents bravo file contents".repeat(40);
        let comp = compress_archive(
            &[("mydir/a.txt", a.as_slice()), ("mydir/b.txt", b.as_slice())],
            CodecMode::Hybrid,
        )
        .expect("compress_archive");
        let members = decompress_archive(&comp).expect("decompress_archive");
        assert_eq!(members.len(), 2);
        assert_eq!(members[0], ("mydir/a.txt".into(), a));
        assert_eq!(members[1], ("mydir/b.txt".into(), b));
        let err = decompress(&comp).unwrap_err();
        assert!(matches!(err, RcnError::InvalidContainer(_)));
        assert!(err.to_string().contains("multi-member"));
    }

    #[test]
    fn v1_decompress_archive_has_empty_name() {
        let original = b"anonymous v1 stream".repeat(50);
        let comp = compress(&original).expect("compress v1");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original);
        let members = decompress_archive(&comp).expect("decompress_archive");
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "");
        assert_eq!(members[0].1, original);
    }

    #[test]
    fn compress_archive_empty_file_round_trips() {
        let other = b"sibling".repeat(20);
        let comp = compress_archive(
            &[("empty.dat", &[] as &[u8]), ("other.txt", other.as_slice())],
            CodecMode::Wire,
        )
        .expect("compress_archive");
        let members = decompress_archive(&comp).expect("decompress_archive");
        assert_eq!(members[0], ("empty.dat".into(), Vec::new()));
        assert_eq!(members[1], ("other.txt".into(), other));
    }

    #[test]
    fn compress_archive_empty_member_list_errors() {
        let err = compress_archive(&[], CodecMode::Hybrid).unwrap_err();
        assert!(matches!(err, RcnError::InvalidContainer(_)));
        assert!(err.to_string().contains("nothing to compress"));
    }

    #[test]
    fn compress_archive_rejects_dotdot_name() {
        let err = compress_archive(&[("../x", b"data".as_slice())], CodecMode::Wire).unwrap_err();
        assert!(matches!(err, RcnError::InvalidContainer(_)));
    }
}
