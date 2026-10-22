# Rcn Optimization Opportunity Log

Last updated: 2026-09-12
Test status: see `cargo test --lib` (198+; with or without `no_avx2`)
Headline benches: W1–W7 on 2026-09-11; levels `-1`/`-3` + dual scorecards 2026-09-12 (README).

Human-readable TODO, A/B history, and CI notes: [DEVELOPMENT.md](DEVELOPMENT.md).
Product overview and headline benches: [README.md](README.md).

---

## External proposal triage (2026-09)

Ideas from an external list that **collide with closed ticket IDs**. Status vs this tree:

| Proposal idea | Verdict |
|---------------|---------|
| S8 interleaved rANS after SIMD walk_dist | **Done** — already wired in `bytecodec` |
| S9 libsais BWT | **Done** — `bwt_libsais` → `libsais-rs` (default still `divsufsort` O(n)) |
| S10 prefetch + stretch-in-`MixerAcc` | **Done as S11** — stretch LUT on `MixerAcc`; `_mm_prefetch` next bank |
| S11 bumpalo / per-block bank arena | **Rejected** — banks are flat SoA + decay in place, not 8192 `Vec`s/block |
| S12 parallel BWT trials | **Done as S12** — `parallel_map_sizes` covers RawCm / XWRT / JSON / CSV / XML trials |
| C6 XWRT-512 ESC | **Done** |
| C7 classifier DP thresholds | **Done** (Text 12; Binary/Exec 16 — ≥24 hurt mr) |
| C8 16k + Indirect default | **Banks done; Indirect shelved** |
| C9 Order-12 PPMd Text | **Tried as C10 — reverted** (gain ≪ 0.3pt) |
| C10 FSE side-stream | **Done** (repo C9) |
| C11 CSV/XML stream split | **Done as C11** — detectors + BWT trial wiring; fixture A/B kept |

---

## Compression Big Tickets (Ratio Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| C1 | Wire SSE/APM/APM2 cascade | Integrated in slow-path encoder/decoder. | -2-4pt on text | Completed |
| C2 | Re-benchmark fixed PPMd config | Order-8 PPMd with SEE + sparse de Bruijn. | +1.5-2pt on webster | Completed |
| C3 | XWRT dictionary before BWT | Per-block XWRT before BWT→MTF→RLE0→CM. | 2-4pt on text | Completed |
| C4 | Exec/Binary transforms | DP-LZP default; E8E9 kept. Delta/stride transform was never wired into the codec and was removed 2026-09-11. | 2-5pt on mr/nci | Completed (E8E9/DP) |
| C5 | Corpus-wide global XWRT-128 | Top-128 dict once in header. | −4-5pt on text | Completed |
| C6 | Global XWRT 128 → 512 with ESC | `0x80..=0xFE` ids 0..126; `0xFF\|\|u16` ids 127..511; ESC only for words len>3. | −1 to −2pt on text | Completed |
| C7 | Adaptive DP LZP threshold | Text ≥12; Binary/Exec stay 16 (≥24 regressed mr). | Help text; avoid mr hit | Completed |
| C8 | 16k banks + indirect | `NUM_MIXERS=16384`. Indirect **not** in default Text stack. | Banks kept; indirect shelved | Completed (banks) |
| C9 | FSE-family match side-stream | Order-0 byte rANS on varint blob when smaller. | ∼0.5–1pt | Completed |
| C10 | Order-12 PPMd for Text only | `with_max_order(12)` API kept; default Text stays order-8. | −0.5pt text | **Reverted** — 2MB dickens −0.003pt |
| C11 | CSV/XML stream splitting | Column / tag-attr-text splits like JSON; methods 15–18. | Structured-data ratio | Completed (fixtures kept) |
| R1 | Adaptive Hybrid mode | Fast Text/Random; Slow Binary/Exec; mixed methods. | Clear zstd -19 gate | **Completed** — CLI default |
| R2 | Fast nci ≤5.0% | Global XWRT on Fast trials + DP-LZP literal-skip on byte path. | ≥0.1pt on nci | **Completed** — fast/hybrid nci 4.97% |

---

## Speed Big Tickets (Throughput Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| S1 | SIMD `walk_dist` | AVX2 cumulative counts. | 5-10x fast path | Completed |
| S2 | SoA weight layout | Contiguous bank weights. | 10-20% slow | Completed |
| S3 | Stretch-value reuse | Precompute stretch in mix/update (within-call). | Small | Completed |
| S4 | Wider stride / context | More models / context bits. | Known gain | Completed |
| S5 | Parallel BWT trials | `rayon::join` path B/C. | Modest | Partial → see S12 |
| S6 / S9 | libsais BWT backend | Optional `bwt_libsais` (`libsais-rs`); default `divsufsort`. | BWT trial speed | Completed (optional) |
| S7 | mimalloc | Allocator. | Small | Completed |
| S8 | Interleaved rANS re-bench | Wired; re-benched after AVX2 walk_dist. | Confirm MB/s | Completed |
| S10 | Classify-ahead pipeline | `rayon::join` classify N+1 while encode N. | Overlap classify | Completed |
| S11 | Prefetch + stretch on `MixerAcc` | Carry stretch LUT across mix→update; prefetch next bank weights. | Ratio-neutral speed | Completed |
| S12 | Wider BWT trial parallelism | `parallel_map_sizes` for remaining size trials. | Text trial wall time | Completed |
| V1 | Cache BWT trial winner payload | Return encoded bytes from trial; no re-encode. | Encode wall-time | **Completed** |
| V2 | aarch64 NEON `walk_dist` | Bit-identical to scalar; parity tests. | Fast path on Apple Silicon | **Completed** |
| V3 | `release-prof` + enum `StackModel` | Symbol-friendly profile; monomorphize Slow stacks. | Slow bit-loop | **Completed** |

---

## Current Stack (What Works)

### Hybrid Path (`--mode hybrid`, default)

- Text/Random → Fast byte CM (global XWRT-1024 trials + DP-LZP side-stream);
  Fast Text/Random blocks encoded in parallel (W3)
- Binary/Exec → Slow bit CM + DP-LZP + E8E9 on Exec; 1 MiB blocks + match hist (W1)
- Clears headline set vs `zstd -19` (2026-09-11 W1–W7 measure)

### Slow Path (`--mode slow`)

- 8 bit models (Text) + two-level **16k**-bank mixer hierarchy + master mixer
- SSE/APM/APM2 cascade; cross-block decay 0.995; 32MB LZP window
- BWT text trial (incl. CSV/XML split) + global XWRT-1024 (ESC) + DP-LZP with adaptive thresholds
- Match side-stream: varint + optional order-0 rANS (C9); Binary/Exec 1 MiB + hist
- Classify-ahead Rayon overlap (S10); stretch-on-acc + bank prefetch (S11); wider BWT trial fan-out (S12)
- Enum `StackModel` (V3) instead of `Box<dyn BitModel>`; default `bwt_libsais` (W7)

### Fast Path (`--mode fast`)

- Orders 0-2 count models + **wired** 32-way interleaved byte rANS
- Binary/Exec: hashed order-3 + large o2 tables (W2)
- AVX2 (x86_64) / NEON (aarch64) SIMD walk_dist (scalar fallback + parity tests; CI uses `no_avx2`)
- Global XWRT on Text trials; DP-LZP literal-skip with match side-stream
- Parallel Text/Random block encode (W3)

### Correctness / container hardening (2026-09-11)

- Structured BWT decode (JSON/CSV/XML/XWRT) returns `RcnError` on truncated or invalid payloads (no silent empty `Vec`)
- Global-dict read and decompress payload/`num_blocks` bounds fail closed
- Shared `split_common` framing helpers; unified method↔pipeline maps in `codec`

### Default-mode gate

`--mode hybrid` is default after clearing `zstd -19` on **all** headline files
(2026-09-11 W1–W7 refresh): dickens 26.5%, webster 20.1%, nci 4.99%, mr 27.3%,
json 0.1%. Scorecard: [README.md](README.md#scorecards). Numbers: [BENCHMARKS.md](BENCHMARKS.md).

---

## Recommended Next Step

1. **Level `-9`:** keep hybrid default. Close the `mr` dual-axis split vs
   `zstd -19` (need ≤31.2% at ≫1 cmp MB/s — Fast Binary is 31.5% / ~4 MB/s).
2. **Level `-1`:** raise wire LZ ratio toward `zstd -1` and cmp speed toward
   `zstd -1` (parallel tokens, longer chain, SIMD match).
3. **Level `-3`:** beat `gzip -9` on both axes (BWT budget or stronger byte
   models without Slow).

Do **not** re-default Text Indirect, re-open Binary DP ≥24, or re-bump Text PPMd
order without a ≥0.3pt measure. A/B history: [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Levels (L1–L19, 2026-09-12)

| Ticket | Change | Status |
|--------|--------|--------|
| L1 | Hash-chain LZ wire engine (`METHOD_WIRE` 19); `--level 1` | Landed — dual-axis vs lz4/`zstd -1` still open |
| L3 | Byte CM, no BWT (`METHOD_BYTE_TEXT` 20); `--level 3` | Landed — dual-axis vs gzip-9 still open |
| L9 | Hybrid remains default (`--level 9`) | Landed — both-win vs `-19` on text/`nci`; `mr`/json split |
| L19 | Slow aliased as `--level 19` | Landed |
| D1 | Dual ratio+cmp+both W-L-T in `bench_vs_sota.sh` + README | Landed |

---

## W1–W7 (2026-09-11)

| Ticket | Change | Notes |
|--------|--------|-------|
| W1 | Binary/Exec block **1 MiB** + cross-block DP-LZP match history (4 MiB cap) | Encoder/decoder keep same-kind hist |
| W2 | Fast Binary/Exec hashed **order-3** + large o2 tables | Text Fast residuals unchanged |
| W3 | Parallel Fast Text/Random block encode (Rayon) | Ordered assemble; Slow stays serial |
| W4 | Global XWRT **1024** words | Was 512 |
| W5 | Binary-only Indirect A/B | **Killed** — mr 27.52% vs prior 27.4% |
| W6 | Kind-specific DP `avg_bits` (Binary/Exec 5.0, else 4.0) | No min-len change |
| W7 | Default `bwt_libsais`; PGO recipe in DEVELOPMENT.md | CI stays non-PGO |

---

## Hotspot note (2026-09-09)

macOS `sample` on release `rcn` (ARM64, stripped — no demangled frames):

1. **Slow encode:** one deep in-process call chain dominates; consistent with bit CM predict→mix→update. S11 `_mm_prefetch` does not appear as a separable hotspot.
2. **Fast encode:** wall time in transform trial + byte entropy; dickens full file ~seconds at ~26.5% ratio after R2.
3. **Next speed lever (done):** NEON `walk_dist` (V2); BWT winner cache (V1); enum stacks (V3). Profile with `cargo build --profile release-prof`.

---

## Test Log

| Date | Notes |
|------|-------|
| 2026-09-08 | C5 global XWRT-128 |
| 2026-09-09 | Docs truth; C6–C9 / S8–S10; triage S11/S12/C11; C10 reverted; hygiene + de-exp docs; C11 fixtures; `--verbose`; fast-default gate documented |
| 2026-09-11 | Hardening pass: fallible BWT/dict/decompress bounds; Fast/CSV/XML + corrupt-container tests; `split_common` + method-map dedup; orphan `delta.rs` removed; AVX2↔scalar parity tests; pre-commit mirrors CI `no_avx2` — **186/186** |
| 2026-09-11 | Full headline re-bench (slow + fast + peers) on `.work/bench5`; README tables refreshed; ratios unchanged vs prior stitch; speeds updated |
| 2026-09-12 | Levels `-1`/`-3`/`-9`/`-19`; dual-axis scorecards; RCN1 v1 documented as stable. **199/199** |
| 2026-09-11 | V1 BWT payload cache; R1 Hybrid; R2 Fast XWRT+DP-LZP (nci 4.97%); V2 NEON walk_dist; V3 `release-prof` + `StackModel`; **default → hybrid**; **187/187** |
| 2026-09-11 | **W1–W7:** Binary 1 MiB + match hist; Fast o3 (mr 35.1%→31.5%); parallel Fast Text; XWRT-1024; Binary Indirect A/B **killed**; kind DP costs; libsais default + PGO docs. Hybrid gate held (mr 27.33%, nci 4.99%). **187/187** |
